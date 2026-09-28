import tomllib
from dataclasses import dataclass, replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.toml"

# Le serveur Rust avance la simulation de 1/60 s par tick
REAL_TIME_TICK_US = 1_000_000 / 60
# Le serveur envoie l'état tous les N ticks de simulation
TICKS_PER_SERVER_UPDATE = 4


@dataclass(frozen=True)
class Config:
    """
    Configuration de HAL-9000, lue depuis config.toml.

    Attributes:
        speedup (float): L'accélération de la simulation par rapport au temps réel.
        training_speedup (float): L'accélération de la simulation pendant l'entraînement.
        n_envs (int): Le nombre de vaisseaux entraînés en parallèle.
        decision_interval (float): Le temps simulé entre deux décisions de l'IA, en secondes.
        websocket_url (str): L'URL du serveur WebSocket.
        server_path (Path): Le dossier du serveur Rust.
        sim_envs (int): Le nombre de vaisseaux simulés en parallèle sur la simulation numpy.
        episode_time (int): La durée maximale d'un épisode, en minutes simulées.
        total_timesteps (int): Le nombre total de steps d'entraînement.
        save_every (int): La fréquence de sauvegarde du modèle, en steps.
        reward (dict): Les paramètres de la récompense (section [reward]).
        ppo (dict): Les hyperparamètres PPO (section [ppo]).
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
        """Retourne la configuration avec l'accélération d'entraînement."""
        return replace(self, speedup=self.training_speedup)

    @property
    def step_time(self) -> float:
        """Temps réel entre deux steps Python, en secondes."""
        return self.decision_interval / self.speedup

    @property
    def steps_per_episode(self) -> int:
        """Nombre maximal de steps par épisode."""
        return round(self.episode_time * 60 / self.decision_interval)

    @property
    def simulation_sleep_us(self) -> int:
        """Pause entre deux ticks du serveur Rust (SIMULATION_SLEEP_TIME_MICROSECONDS)."""
        return round(REAL_TIME_TICK_US / self.speedup)

    @property
    def server_sleep_us(self) -> int:
        """Pause entre deux envois d'état du serveur Rust (SERVER_SLEEP_TIME_MICROSECONDS)."""
        return TICKS_PER_SERVER_UPDATE * self.simulation_sleep_us


def load_config(path: Path = CONFIG_PATH, overrides: list[str] | None = None) -> Config:
    """
    Charge la configuration depuis un fichier TOML.

    Args:
        path (Path): Le chemin du fichier de configuration.
        overrides (list[str], optional): Des surcharges "section.clé=valeur" (valeur en syntaxe TOML).

    Returns:
        Config: La configuration.
    """
    with open(path, "rb") as f:
        data = tomllib.load(f)
    for override in overrides or []:
        key, _, value = override.partition("=")
        section, _, name = key.partition(".")
        if section not in data or name not in data[section]:
            raise ValueError(f"Surcharge inconnue : {key}")
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
        raise ValueError(f"{path}: speedup, decision_interval et n_envs doivent être positifs")
    return config
