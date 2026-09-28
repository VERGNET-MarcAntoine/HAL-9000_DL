import glob
import os
import time
from dataclasses import replace
from functools import partial

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv, VecMonitor

from hal9000.config import Config
from hal9000.model.core.loading import load_ppo
from hal9000.model.core.ship2D import Ship2D
from hal9000.sim.live import LivePublisher
from hal9000.sim.vec_env import Hal9000SimVecEnv
from hal9000.websocket.websocket_client import SpaceshipWebSocketClient

LOG_DIR = "logs"
MODELS_DIR = "models"
MODEL_NAME = "Hal9000_2D"
# Écart toléré entre l'accélération mesurée du serveur et celle de la config
SPEEDUP_TOLERANCE = 0.2
# Le réseau est trop petit pour profiter du GPU : sur CPU avec un seul thread, chaque step
# et chaque mise à jour PPO sont ~4x plus rapides (moins de latence, donc une synchro tenable)
DEVICE = "cpu"


def _orbit_angle(state: dict) -> tuple[float, float]:
    """Angle et vitesse angulaire de la planète la plus lointaine autour du soleil."""
    position = np.array(state["planets"][-1][1][:2]) - np.array(state["planets"][0][1][:2])
    speed = np.array(state["planet_speeds"][-1][1][:2])
    return np.arctan2(position[1], position[0]), np.linalg.norm(speed) / np.linalg.norm(position)


def measure_server_speedup(config: Config, duration: float = 2.0) -> float:
    """
    Mesure l'accélération réelle du serveur à partir du mouvement orbital d'une planète.

    Args:
        config (Config): La configuration (pour l'URL du serveur).
        duration (float): La durée de la mesure, en secondes réelles.

    Returns:
        float: Le nombre de secondes simulées par seconde réelle.
    """
    client = SpaceshipWebSocketClient(config.websocket_url)
    client.connect()
    try:
        angle_start, angular_speed = _orbit_angle(client.get_state())
        start = time.perf_counter()
        time.sleep(duration)
        angle_end, _ = _orbit_angle(client.get_state())
        elapsed = time.perf_counter() - start
    finally:
        client.disconnect()
    return ((angle_end - angle_start) % (2 * np.pi)) / angular_speed / elapsed


def server_config(config: Config) -> Config:
    """
    Vérifie que le serveur tourne à la vitesse attendue et retourne la configuration cadencée sur
    sa vitesse mesurée (le serveur n'atteint jamais tout à fait la vitesse demandée).
    """
    measured_speedup = measure_server_speedup(config)
    if abs(measured_speedup / config.speedup - 1) > SPEEDUP_TOLERANCE:
        raise SystemExit(
            f"Le serveur tourne à x{measured_speedup:.0f} alors que x{config.speedup:g} est attendu.\n"
            "Lancez-le avec : uv run python -m hal9000.server" + (" --train" if config.speedup > 1 else ""))
    print(f"Serveur à x{measured_speedup:.1f}")
    return replace(config, speedup=measured_speedup)


def make_env(config: Config, backend: str, n_envs: int | None = None, seed: int | None = None) -> VecEnv:
    """
    Crée les vaisseaux d'entraînement ou d'évaluation.

    Args:
        config (Config): La configuration (à la vitesse voulue pour le serveur).
        backend (str): "sim" (simulation numpy) ou "server" (serveur Rust).
        n_envs (int, optional): Le nombre de vaisseaux (par défaut, celui de la config pour ce backend).
        seed (int, optional): La graine aléatoire.

    Returns:
        VecEnv: Les environnements, avec les statistiques d'épisode (VecMonitor).
    """
    if backend == "sim":
        return VecMonitor(Hal9000SimVecEnv(n_envs or config.sim_envs, config, seed))
    config = server_config(config)
    n_envs = n_envs or config.n_envs
    vec_env_class = DummyVecEnv if n_envs == 1 else SubprocVecEnv
    env = vec_env_class([partial(Ship2D, config) for _ in range(n_envs)])
    env.seed(seed)
    return VecMonitor(env)


def ppo_kwargs(config: Config, n_envs: int) -> dict:
    """Hyperparamètres PPO de la configuration, avec un rollout de même taille quel que soit n_envs."""
    kwargs = dict(config.ppo)
    rollout_size = kwargs.pop("rollout_size")
    net_arch = kwargs.pop("net_arch")
    return dict(n_steps=max(1, rollout_size // n_envs), policy_kwargs={"net_arch": list(net_arch)}, **kwargs)


def latest_model(name: str = MODEL_NAME) -> str | None:
    """Le dernier modèle sauvegardé sous ce nom, ou None."""
    models = glob.glob(os.path.join(MODELS_DIR, f"{name}_*.zip"))
    return max(models, key=os.path.getmtime) if models else None


class HalMetrics(BaseCallback):
    """
    Enregistre dans TensorBoard ce qui compte vraiment, indépendamment de la forme de la récompense :
    planètes atteintes par épisode, taux de survie et causes de mort.
    """

    def _on_rollout_start(self):
        self.episodes = []

    def _on_step(self) -> bool:
        self.episodes += [info["hal"] for info in self.locals["infos"] if "hal" in info]
        return True

    def _on_rollout_end(self):
        if not self.episodes:
            return
        targets = np.array([e["targets"] for e in self.episodes])
        dead_sun = np.array([e["dead_sun"] for e in self.episodes])
        dead_out = np.array([e["dead_out"] for e in self.episodes])
        self.logger.record("hal/targets_per_episode", targets.mean())
        self.logger.record("hal/survival_rate", 1 - (dead_sun | dead_out).mean())
        self.logger.record("hal/death_sun_rate", dead_sun.mean())
        self.logger.record("hal/death_out_rate", dead_out.mean())
        self.logger.record("hal/episodes", len(self.episodes))


def train(config: Config, backend: str, timesteps: int, new: bool = False, model_path: str | None = None,
          name: str = MODEL_NAME):
    """
    Entraîne Hal9000_2D, en reprenant le dernier modèle sauvegardé sauf si new est vrai.

    Args:
        config (Config): La configuration.
        backend (str): "sim" (simulation numpy) ou "server" (serveur Rust, lancé avec --train).
        timesteps (int): Le nombre de steps à entraîner.
        new (bool): Repartir d'un modèle neuf.
        model_path (str, optional): Le modèle à reprendre (par défaut, le dernier sauvegardé sous ce nom).
        name (str): Le nom du modèle (préfixe des sauvegardes et des courbes TensorBoard).
    """
    torch.set_num_threads(1)
    if backend == "server":
        config = config.for_training()
    n_envs = config.sim_envs if backend == "sim" else config.n_envs
    env = make_env(config, backend)
    if backend == "sim":
        # Visible en direct avec `display_ship2D --training`
        env.venv.publisher = LivePublisher(name)
    print(f"{n_envs} vaisseaux en parallèle ({backend})")

    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = None if new else (model_path or latest_model(name))
    if model_path:
        model = load_ppo(model_path, env, config, n_envs, device=DEVICE)
        print(f"Modèle repris : {model_path} ({model.num_timesteps} steps)")
    else:
        model = PPO("MlpPolicy", env, tensorboard_log=LOG_DIR, device=DEVICE, **ppo_kwargs(config, n_envs))
        print("Nouveau modèle créé.")

    name = f"{name}_{backend}"
    checkpoint = CheckpointCallback(save_freq=max(1, config.save_every // n_envs), save_path=MODELS_DIR, name_prefix=name)
    # Un nouveau modèle a son propre dossier de logs ; une reprise continue les courbes du dossier précédent
    model.learn(total_timesteps=timesteps, callback=[HalMetrics(), checkpoint],
                tb_log_name=name, reset_num_timesteps=model_path is None)
    final_path = os.path.join(MODELS_DIR, f"{name}_{model.num_timesteps}_steps.zip")
    model.save(final_path)
    print(f"Modèle sauvegardé : {final_path}")
    env.close()
