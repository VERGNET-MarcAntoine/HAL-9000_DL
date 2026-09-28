"""
Evaluates a Hal9000_2D model (deterministic actions): planets reached per episode, survival, causes of death.

By default, the episodes run on the Python simulation. With --server, they run on the Rust server at the
training speed (server started with --train). With --watch, a few ships fly on the Rust server at the
[simulation] speed (server started without --train), to be watched with `display_ship2D --server`, which
also shows their targets.
"""
import argparse
from dataclasses import replace

import numpy as np
import torch
from stable_baselines3 import PPO

from hal9000.config import load_config
from hal9000.model.core.loading import load_ppo
from hal9000.model.core.training import latest_model, make_env
from hal9000.sim.live import LivePublisher


def evaluate(model: PPO, env, episodes: int, publisher: LivePublisher | None = None) -> list[dict]:
    """
    Flies the model until every ship has finished the same number of episodes (at least `episodes` in
    total). Stopping at the first `episodes` finished episodes would over-represent the deaths, which end
    earlier than the episodes flown to the end.

    Returns:
        list[dict]: The statistics of each episode (targets, dead_sun, dead_out, length).
    """
    per_env = -(-episodes // env.num_envs)
    results: list[list[dict]] = [[] for _ in range(env.num_envs)]
    obs = env.reset()
    while min(len(r) for r in results) < per_env:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, infos = env.step(action)
        if publisher:
            publisher.publish_server_ships([info["ship"] for info in infos if "ship" in info])
        for i in np.flatnonzero(dones):
            if len(results[i]) < per_env:
                info = infos[i]
                results[i].append({**info["hal"], "length": info["episode"]["l"]})
                if publisher:
                    print(f"Ship {i + 1}, episode {len(results[i])}: {info['hal']['targets']} planets, "
                          f"{info['episode']['l']} steps"
                          + (", fell into the sun" if info["hal"]["dead_sun"] else "")
                          + (", lost in space" if info["hal"]["dead_out"] else ""))
    return [episode for r in results for episode in r]


def summary(results: list[dict]) -> str:
    targets = np.array([r["targets"] for r in results])
    dead_sun = np.mean([r["dead_sun"] for r in results])
    dead_out = np.mean([r["dead_out"] for r in results])
    return (f"{len(results)} episodes | planets/episode: {targets.mean():.2f} (median {np.median(targets):.0f}, "
            f"max {targets.max()}) | survival: {1 - dead_sun - dead_out:.0%} | sun: {dead_sun:.0%} | lost: {dead_out:.0%}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="model to evaluate (default: the latest saved one)")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--server", action="store_true", help="evaluate on the Rust server (started with --train)")
    parser.add_argument("--watch", action="store_true", help="a few ships on the Rust server, at the [simulation] speed, "
                        "to watch with display_ship2D --server")
    parser.add_argument("--ships", type=int, default=4, help="with --watch: number of ships")
    parser.add_argument("--url", help="address of the server (default: the one of config.toml)")
    parser.add_argument("--seed", type=int, default=0, help="random seed (starting situations)")
    parser.add_argument("--set", action="append", metavar="SECTION.KEY=VALUE", help="override a value of config.toml")
    args = parser.parse_args()

    torch.set_num_threads(1)
    config = load_config(overrides=args.set)
    if args.url:
        config = replace(config, websocket_url=args.url)
    model_path = args.model or latest_model()
    if model_path is None:
        raise SystemExit("No model in models/: train one with `uv run python -m hal9000.train`.")
    if args.watch:
        env = make_env(config, "server", n_envs=args.ships, seed=args.seed)
    elif args.server:
        env = make_env(config.for_training(), "server", seed=args.seed)
    else:
        env = make_env(config, "sim", n_envs=args.episodes, seed=args.seed)

    model = load_ppo(model_path, env, config, env.num_envs)
    print(f"Model: {model_path}")
    publisher = LivePublisher("watch") if args.watch else None
    print(summary(evaluate(model, env, args.episodes, publisher)))
    env.close()


if __name__ == "__main__":
    main()
