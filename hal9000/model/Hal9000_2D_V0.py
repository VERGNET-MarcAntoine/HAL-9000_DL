import numpy as np
from hal9000.model.core.ship2D import Ship2D
from dotenv import load_dotenv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
from datetime import datetime

import os


class Hal9000_2D_V0(Ship2D):
    """
    A custom environment that inherits from Ship2D and defines a specific reward function.
    """

    def __init__(self, episode_time: int, step_time: float, websocket_url: str = "ws://127.0.0.1:3012"):
        """
        Initializes the MyShipEnv.
        """
        super().__init__(episode_time, step_time, websocket_url)
        # You can add any specific initialization code here if needed
        self.previous_distance_to_target = None  # To track progress towards the target

    def get_reward(self, previous_state: dict) -> tuple[float, bool]:
        """
        Calcule la récompense et détermine si l'épisode est terminé.
        Args:
            previous_state (dict): L'état précédent de l'environnement.

        Returns:
            tuple[float, bool]: La récompense et un booléen indiquant si l'épisode est terminé.
        """

        previous_ship_data = self.get_ship_data(previous_state)
        previous_target_data, previous_planet_data = self.get_planet_data(
            previous_state)

        ship_data = self.get_ship_data(previous_state)
        target_data, planet_data = self.get_planet_data(
            self.state)

        previous_distance_sun = np.linalg.norm(
            previous_ship_data[0:2] - previous_planet_data[0:2])
        distance_sun = np.linalg.norm(ship_data[0:2] - planet_data[0:2])

        if distance_sun > 20000 or distance_sun < 150:
            return -1000, True  # Mort fin de l'eposide avec pénalité sevére

        previous_distance_target = np.linalg.norm(
            previous_ship_data[0:2] - previous_target_data[0:2])
        distance_target = np.linalg.norm(ship_data[0:2] - target_data[0:2])

        delta_distance = previous_distance_target - distance_target

        # Récompense principale : réduction de la distance cible
        reward = (-distance_target / 20000)

        if delta_distance > 0:
            # Bonus si on se rapproche
            reward += (delta_distance / 500)

        # Pénalité progressive pour la proximité au soleil
        if distance_sun < 500:
            # Pénalité plus douce en s'approchant
            reward -= (500 - distance_sun) / 500
        elif distance_sun > 3000:
            # Pénalité croissante si trop loin
            reward -= (distance_sun - 3000) / 2000

        # Récompense progressive pour atteindre l’objectif
        if distance_target < 200:
            reward += 1000
            self.current_target += 1
            print(f"Score : {self.current_target}")
            if self.current_target == self.nb_planets:
                self.current_target = 0

        # Récompense basée sur l'accélération (direction vers la cible)
        acceleration = ship_data[2:4] - previous_ship_data[2:4]

        direction_to_target = target_data[0:2] - ship_data[0:2]
        # Normalisation
        direction_to_target /= np.linalg.norm(direction_to_target)
        alignment_reward = np.dot(acceleration, direction_to_target)
        # Modulation de la récompense d'alignement selon la distance
        # Diminue l'importance en se rapprochant
        weight = min(1, distance_target / 1000)

        if alignment_reward > 0:
            reward += weight * alignment_reward * 5  # Bonus si aligné
        else:
            reward += weight * alignment_reward * 2  # Pénalité si opposé

        speed_norm = np.linalg.norm(ship_data[2:4])
        if speed_norm < 100:  # Trop lent, risque de stagnation
            reward -= (100 - speed_norm) / 2  # Pénalité progressive
        elif speed_norm > 500:  # Trop rapide, risque de perte de contrôle
            reward -= (speed_norm - 500) / 10  # Pénalité progressive
        return reward, False


if __name__ == "__main__":
    load_dotenv()
    episode_time = int(os.getenv("EPISODE_TIME"))
    step_time = float(os.getenv("SLEEP_TIME"))

    hal9000 = Hal9000_2D_V0(episode_time, step_time)
    check_env(hal9000)
    logdir = "logs"
    models_dir = "models"

    # Créer le dossier des logs et des modèles s'ils n'existent pas
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    number_episode = int(os.getenv("NUMBER_EPISODE"))
    save_number = int(os.getenv("SAVE_NUMBER"))

    TIMESTEPS = (number_episode * episode_time * 60 * 4) // save_number
    model_name = "hal9000_2D_V0"  # Nom du modèle

    existing_models = [f for f in os.listdir(models_dir) if f.startswith(model_name) and f.endswith(".zip")]

    if existing_models:
        def extract_timestep(filename):
            try:
                return int(filename.split("_step")[1].split(".zip")[0])
            except (IndexError, ValueError):
                return 0

        existing_models.sort(key=extract_timestep)

        latest_model_path = os.path.join(models_dir, existing_models[-1])
        model = PPO.load(latest_model_path, env=hal9000)
        print(f"Modèle existant chargé : {latest_model_path}")
        loaded_timesteps = extract_timestep(existing_models[-1])
    else:

        model = PPO("MultiInputPolicy", hal9000, tensorboard_log=logdir)
        print("Nouveau modèle créé.")
        loaded_timesteps = 0

    start_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    log_name = f"{model_name}_{start_time}"  # Format des logs

    for i in range(save_number):
        model.learn(total_timesteps=TIMESTEPS,
                    reset_num_timesteps=False, tb_log_name=log_name)
        
        total_steps = loaded_timesteps + TIMESTEPS * (i + 1)
        new_model_path = os.path.join(models_dir, f"{log_name}_step{total_steps}.zip")

        print(new_model_path)
        model.save(new_model_path)

    hal9000.close()
