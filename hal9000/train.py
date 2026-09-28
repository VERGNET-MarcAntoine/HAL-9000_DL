"""Trains Hal9000_2D on the Python simulation (default) or on the Rust server (--server, server started with --train)."""
import argparse

from hal9000.config import load_config
from hal9000.model.core.training import train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", action="store_true", help="train on the Rust server (started with --train)")
    parser.add_argument("--steps", type=int, help="number of steps to train (default: total_timesteps)")
    parser.add_argument("--new", action="store_true", help="start from a new model")
    parser.add_argument("--model", help="model to resume (default: the latest saved under this name)")
    parser.add_argument("--name", default="Hal9000_2D", help="name of the model (checkpoints and TensorBoard curves)")
    parser.add_argument("--set", action="append", metavar="SECTION.KEY=VALUE",
                        help="override a value of config.toml, for example --set reward.death_penalty=20")
    args = parser.parse_args()

    config = load_config(overrides=args.set)
    train(config, "server" if args.server else "sim", args.steps or config.total_timesteps, args.new, args.model,
          args.name)


if __name__ == "__main__":
    main()
