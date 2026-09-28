"""Starts the Rust server with a simulation speed synchronized with config.toml."""
import argparse
import os
import subprocess
from urllib.parse import urlparse

from hal9000.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", action="store_true", help="use the training speed")
    args = parser.parse_args()

    config = load_config()
    if args.train:
        config = config.for_training()
    url = urlparse(config.websocket_url)

    env = {
        **os.environ,
        "SIMULATION_SLEEP_TIME_MICROSECONDS": str(config.simulation_sleep_us),
        "SERVER_SLEEP_TIME_MICROSECONDS": str(config.server_sleep_us),
        "WEBSOCKET_HOST": url.hostname or "127.0.0.1",
        "WEBSOCKET_PORT": str(url.port or 3012),
    }
    print(f"Simulation x{config.speedup:g}: a tick every {config.simulation_sleep_us} µs, "
          f"state sent every {config.server_sleep_us} µs, on {config.websocket_url}")

    try:
        subprocess.run(["cargo", "run", "--release"], cwd=config.server_path, env=env, check=True)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
