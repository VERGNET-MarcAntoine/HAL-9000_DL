import os
import time
from dataclasses import replace
from datetime import datetime
from functools import partial

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from hal9000.config import Config, load_config
from hal9000.model.core.loading import load_ppo
from hal9000.model.core.ship2D import Ship2D
from hal9000.websocket.websocket_client import SpaceshipWebSocketClient

LOG_DIR = "logs"
MODELS_DIR = "models"
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


def train(env_class: type[Ship2D], model_name: str):
    """
    Entraîne un modèle PPO sur plusieurs vaisseaux en parallèle, en reprenant le dernier
    modèle sauvegardé s'il existe.

    Args:
        env_class (type[Ship2D]): La classe d'environnement (qui définit la récompense).
        model_name (str): Le nom du modèle, préfixe des fichiers sauvegardés.
    """
    config = load_config().for_training()
    torch.set_num_threads(1)

    # Un serveur à la mauvaise vitesse fausse chaque step : on vérifie avant de lancer des heures de calcul
    measured_speedup = measure_server_speedup(config)
    if abs(measured_speedup / config.speedup - 1) > SPEEDUP_TOLERANCE:
        raise SystemExit(
            f"Le serveur tourne à x{measured_speedup:.0f} alors que l'entraînement attend x{config.speedup:g}.\n"
            "Lancez-le avec : uv run python -m hal9000.server --train")
    # Le serveur n'atteint jamais tout à fait la vitesse demandée (coût de chaque tick) :
    # on cadence Python sur la vitesse mesurée pour que chaque step dure bien decision_interval simulé
    config = replace(config, speedup=measured_speedup)
    print(f"Serveur à x{measured_speedup:.1f}, {config.n_envs} vaisseaux en parallèle")

    check_env(env_class(config))
    env = VecMonitor(SubprocVecEnv([partial(env_class, config) for _ in range(config.n_envs)]))

    # Créer le dossier des logs et des modèles s'ils n'existent pas
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    print(model_name)
    existing_models = [f for f in os.listdir(
        MODELS_DIR) if f.startswith(model_name) and f.endswith(".zip")]

    if existing_models:
        def extract_timestep(filename):
            try:
                return int(filename.split("_step")[1].split(".zip")[0])
            except (IndexError, ValueError):
                return 0

        existing_models.sort(key=extract_timestep)

        latest_model_path = os.path.join(MODELS_DIR, existing_models[-1])
        model = load_ppo(latest_model_path, env, device=DEVICE)
        print(f"Modèle existant chargé : {latest_model_path}")
        loaded_timesteps = extract_timestep(existing_models[-1])
    else:
        model = PPO("MultiInputPolicy", env, tensorboard_log=LOG_DIR, device=DEVICE)
        print("Nouveau modèle créé.")
        loaded_timesteps = 0

    start_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    log_name = f"{model_name}_{start_time}"  # Format des logs

    # Les timesteps sont comptés sur l'ensemble des vaisseaux
    timesteps_per_episode = config.steps_per_episode

    # Calculer le nombre de timesteps pour SAVE_NUMBER épisodes
    timesteps_for_save = timesteps_per_episode * config.save_number

    # Calculer le nombre total de timesteps à entraîner
    total_train_timesteps = config.number_episode * timesteps_per_episode

    # Entraîner et sauvegarder tous les SAVE_NUMBER épisodes
    current_timesteps = loaded_timesteps
    while current_timesteps < total_train_timesteps:
        # Entraîner pour SAVE_NUMBER épisodes
        model.learn(total_timesteps=timesteps_for_save,
                    reset_num_timesteps=False, tb_log_name=log_name)

        current_timesteps += timesteps_for_save
        new_model_path = os.path.join(
            MODELS_DIR, f"{log_name}_step{current_timesteps}.zip")

        print(new_model_path)
        model.save(new_model_path)

    env.close()
