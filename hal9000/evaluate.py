"""
Évalue un modèle Hal9000_2D (actions déterministes) : planètes atteintes par épisode, survie, causes de mort.

Par défaut, les épisodes tournent sur la simulation Python. Avec --server, ils tournent sur le serveur
Rust à la vitesse d'entraînement (serveur lancé avec --train). Avec --watch, quelques vaisseaux volent sur
le serveur Rust à la vitesse de [simulation] (serveur lancé sans --train), pour les regarder avec
`display_ship2D --server`, qui affiche aussi leurs cibles.
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


def evaluate(model: PPO, env, episodes: int, publisher=None) -> list[dict]:
    """
    Fait voler le modèle jusqu'à ce que chaque vaisseau ait terminé le même nombre d'épisodes (au moins
    `episodes` au total). S'arrêter aux `episodes` premiers épisodes terminés surreprésenterait les morts,
    qui finissent plus tôt que les épisodes menés à leur terme.

    Returns:
        list[dict]: Les statistiques de chaque épisode (targets, dead_sun, dead_out, length).
    """
    per_env = -(-episodes // env.num_envs)
    results = [[] for _ in range(env.num_envs)]
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
                    print(f"Vaisseau {i + 1}, épisode {len(results[i])} : {info['hal']['targets']} planètes, {info['episode']['l']} steps"
                          + (", tombé dans le soleil" if info["hal"]["dead_sun"] else "")
                          + (", perdu dans l'espace" if info["hal"]["dead_out"] else ""))
    return [episode for r in results for episode in r]


def summary(results: list[dict]) -> str:
    targets = np.array([r["targets"] for r in results])
    dead_sun = np.mean([r["dead_sun"] for r in results])
    dead_out = np.mean([r["dead_out"] for r in results])
    return (f"{len(results)} épisodes | planètes/épisode : {targets.mean():.2f} (médiane {np.median(targets):.0f}, "
            f"max {targets.max()}) | survie : {1 - dead_sun - dead_out:.0%} | soleil : {dead_sun:.0%} | perdu : {dead_out:.0%}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="modèle à évaluer (par défaut : le dernier sauvegardé)")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--server", action="store_true", help="évaluer sur le serveur Rust (lancé avec --train)")
    parser.add_argument("--watch", action="store_true", help="quelques vaisseaux sur le serveur Rust, à la vitesse de [simulation], "
                        "à regarder avec display_ship2D --server")
    parser.add_argument("--ships", type=int, default=4, help="avec --watch : nombre de vaisseaux")
    parser.add_argument("--url", help="adresse du serveur (par défaut : celle de config.toml)")
    parser.add_argument("--seed", type=int, default=0, help="graine aléatoire (situations de départ)")
    parser.add_argument("--set", action="append", metavar="SECTION.CLÉ=VALEUR", help="surcharger une valeur de config.toml")
    args = parser.parse_args()

    torch.set_num_threads(1)
    config = load_config(overrides=args.set)
    if args.url:
        config = replace(config, websocket_url=args.url)
    if args.watch:
        env = make_env(config, "server", n_envs=args.ships, seed=args.seed)
    elif args.server:
        env = make_env(config.for_training(), "server", seed=args.seed)
    else:
        env = make_env(config, "sim", n_envs=args.episodes, seed=args.seed)

    model_path = args.model or latest_model()
    model = load_ppo(model_path, env, config, env.num_envs)
    print(f"Modèle : {model_path}")
    publisher = LivePublisher("watch") if args.watch else None
    print(summary(evaluate(model, env, args.episodes, publisher)))
    env.close()


if __name__ == "__main__":
    main()
