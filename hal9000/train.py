"""Entraîne Hal9000_2D sur la simulation Python (par défaut) ou sur le serveur Rust (--server, serveur lancé avec --train)."""
import argparse

from hal9000.config import load_config
from hal9000.model.core.training import train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", action="store_true", help="entraîner sur le serveur Rust (lancé avec --train)")
    parser.add_argument("--steps", type=int, help="nombre de steps à entraîner (par défaut : total_timesteps)")
    parser.add_argument("--new", action="store_true", help="repartir d'un modèle neuf")
    parser.add_argument("--model", help="modèle à reprendre (par défaut : le dernier sauvegardé sous ce nom)")
    parser.add_argument("--name", default="Hal9000_2D", help="nom du modèle (sauvegardes et courbes TensorBoard)")
    parser.add_argument("--set", action="append", metavar="SECTION.CLÉ=VALEUR",
                        help="surcharger une valeur de config.toml, par exemple --set reward.death_penalty=20")
    args = parser.parse_args()

    config = load_config(overrides=args.set)
    train(config, "server" if args.server else "sim", args.steps or config.total_timesteps, args.new, args.model,
          args.name)


if __name__ == "__main__":
    main()
