import tomllib
from dataclasses import dataclass, replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.toml"

# The Rust server advances the simulation by 1/60 s per tick
REAL_TIME_TICK_US = 1_000_000 / 60
# The server sends the state every N simulation ticks
TICKS_PER_SERVER_UPDATE = 4


@dataclass(frozen=True)
class Config:
    """
    Configuration of HAL-9000, read from config.toml.

    Attributes:
        speedup (float): The speed of the simulation compared to real time.
        training_speedup (float): The speed of the simulation during a training on the Rust server.
        n_envs (int): The number of ships trained in parallel on the Rust server.
        decision_interval (float): The simulated time between two decisions of the AI, in seconds.
        websocket_url (str): The URL of the WebSocket server.
        server_path (Path): The directory of the Rust server.
        sim_envs (int): The number of ships simulated in parallel on the numpy simulation.
        episode_time (int): The maximum duration of an episode, in simulated minutes.
        total_timesteps (int): The total number of training steps.
        save_every (int): The checkpoint frequency, in steps.
        reward (dict): The reward parameters ([reward] section).
        ppo (dict): The PPO hyperparameters ([ppo] section).
    """
    speedup: float
    training_speedup: float
    n_envs: int
    decision_interval: float
    websocket_url: str
    server_path: Path
    sim_envs: int
    episode_time: int
    total_timesteps: int
    save_every: int
    reward: dict
    ppo: dict

    def for_training(self) -> "Config":
        """Returns the configuration with the training speed."""
        return replace(self, speedup=self.training_speedup)

    @property
    def step_time(self) -> float:
        """Real time between two Python steps, in seconds."""
        return self.decision_interval / self.speedup

    @property
    def steps_per_episode(self) -> int:
        """Maximum number of steps per episode."""
        return round(self.episode_time * 60 / self.decision_interval)

    @property
    def simulation_sleep_us(self) -> int:
        """Pause between two ticks of the Rust server (SIMULATION_SLEEP_TIME_MICROSECONDS)."""
        return round(REAL_TIME_TICK_US / self.speedup)

    @property
    def server_sleep_us(self) -> int:
        """Pause between two state messages of the Rust server (SERVER_SLEEP_TIME_MICROSECONDS)."""
        return TICKS_PER_SERVER_UPDATE * self.simulation_sleep_us


def load_config(path: Path = CONFIG_PATH, overrides: list[str] | None = None) -> Config:
    """
    Loads the configuration from a TOML file.

    Args:
        path (Path): The path of the configuration file.
        overrides (list[str], optional): Overrides "section.key=value" (value in TOML syntax).

    Returns:
        Config: The configuration.
    """
    with open(path, "rb") as f:
        data = tomllib.load(f)
    for override in overrides or []:
        key, _, value = override.partition("=")
        section, _, name = key.partition(".")
        if section not in data or name not in data[section]:
            raise ValueError(f"Unknown override: {key}")
        data[section][name] = tomllib.loads(f"v = {value}")["v"]

    config = Config(
        speedup=float(data["simulation"]["speedup"]),
        training_speedup=float(data["training"]["speedup"]),
        n_envs=int(data["training"]["n_envs"]),
        decision_interval=float(data["simulation"]["decision_interval"]),
        websocket_url=data["server"]["websocket_url"],
        server_path=PROJECT_ROOT / data["server"]["path"],
        sim_envs=int(data["training"]["sim_envs"]),
        episode_time=int(data["training"]["episode_time"]),
        total_timesteps=int(data["training"]["total_timesteps"]),
        save_every=int(data["training"]["save_every"]),
        reward=dict(data["reward"]),
        ppo=dict(data["ppo"]),
    )
    if min(config.speedup, config.training_speedup, config.decision_interval, config.n_envs) <= 0:
        raise ValueError(f"{path}: speedup, decision_interval and n_envs must be positive")
    return config
