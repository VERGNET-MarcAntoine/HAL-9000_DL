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

        ship_data = self.get_ship_data(self.state)
        target_data, planet_data = self.get_planet_data(self.state)

        # print(f"""
        # --- ÉTAT PRÉCÉDENT ---
        # Vaisseau: {previous_ship_data}
        # Cible: {previous_target_data}
        # Planètes: {previous_planet_data}

        # --- ÉTAT ACTUEL ---
        # Vaisseau: {ship_data}
        # Cible: {target_data}
        # Planètes: {planet_data}
        # """)

        previous_ship_data = self.get_ship_data(previous_state)
        previous_target_data, previous_planet_data = self.get_planet_data(
            previous_state)

        ship_data = self.get_ship_data(self.state)
        target_data, planet_data = self.get_planet_data(self.state)

        # Calcul des distances
        previous_distance_sun = np.linalg.norm(
            previous_ship_data[0:2] - previous_planet_data[0:2])
        distance_sun = np.linalg.norm(ship_data[0:2] - planet_data[0:2])

        # Vérification des limites de la gravité et fin d'épisode si nécessaire
        if distance_sun > 20000 or distance_sun < 150:
            # Pénalité sévère et fin de l'épisode si trop loin ou trop près du soleil
            return -1000, True

        # Calcul des distances par rapport à la cible
        previous_distance_target = np.linalg.norm(
            previous_ship_data[0:2] - previous_target_data[0:2])
        distance_target = np.linalg.norm(ship_data[0:2] - target_data[0:2])

        # Calcul de la variation de la distance à la cible
        delta_distance = previous_distance_target - distance_target

        # Récompense pour la réduction de la distance cible
        reward = delta_distance / distance_target

        # Récompense pour accélérer vers la cible (calcul de l'accélération)
        acceleration = ship_data[2:4] - previous_ship_data[2:4]
        target_direction = target_data[0:2] - ship_data[0:2]
        # Normalisation pour obtenir une direction unitaire
        target_direction /= np.linalg.norm(target_direction)

        # Calcul du produit scalaire entre l'accélération et la direction de la cible (récompense si on accélère dans la bonne direction)
        acceleration_dot_target = np.dot(acceleration, target_direction)
        # Récompense proportionnelle à l'alignement de l'accélération avec la cible
        reward += 10 * acceleration_dot_target

        # Pénalisation pour accélérer dans la direction opposée à la cible
        if acceleration_dot_target < 0:
            reward -= 5  # Pénalité si l'accélération est dirigée à l'opposé de la cible

        # Récompense basée sur la vitesse (bonus pour une vitesse modérée et efficace)
        speed = np.linalg.norm(ship_data[2:4])
        if speed > 50:
            # Pénalisation si la vitesse est trop élevée (risque de dépassement ou de gaspillage de carburant)
            reward -= 1

        # Récompense pour atteindre l'objectif
        if distance_target < 200:
            reward += 1000  # Grande récompense lorsqu'on atteint la cible
            self.current_target += 1
            print(f"Score : {self.current_target}")
            if self.current_target == self.nb_planets:
                self.current_target = 0
            # Retourner la récompense et l'indication que l'épisode n'est pas encore terminé
            return reward, False

        # Retourner la récompense et indiquer que l'épisode continue
        return reward, False


if __name__ == "__main__":
    load_dotenv()
    episode_time = int(os.getenv("EPISODE_TIME"))
    step_time = float(os.getenv("SLEEP_TIME"))
    number_episode = int(os.getenv("NUMBER_EPISODE"))
    save_number = int(os.getenv("SAVE_NUMBER"))

    hal9000 = Hal9000_2D_V0(episode_time, step_time)
    check_env(hal9000)
    logdir = "logs"
    models_dir = "models"

    # Créer le dossier des logs et des modèles s'ils n'existent pas
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    model_name = __file__.split(
        "\\")[-1].split("/")[-1].split(".")[0]  # Nom du modèle
    print(model_name)
    existing_models = [f for f in os.listdir(
        models_dir) if f.startswith(model_name) and f.endswith(".zip")]

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

    # Calculer le nombre de timesteps par épisode
    timesteps_per_episode = episode_time * 60 * 4

    # Calculer le nombre de timesteps pour SAVE_NUMBER épisodes
    timesteps_for_save = timesteps_per_episode * save_number

    # Calculer le nombre total de timesteps à entraîner
    total_train_timesteps = number_episode * timesteps_per_episode

    # Entraîner et sauvegarder tous les SAVE_NUMBER épisodes
    current_timesteps = loaded_timesteps
    while current_timesteps < total_train_timesteps:
        # Entraîner pour SAVE_NUMBER épisodes
        model.learn(total_timesteps=timesteps_for_save,
                    reset_num_timesteps=False, tb_log_name=log_name)

        current_timesteps += timesteps_for_save
        new_model_path = os.path.join(
            models_dir, f"{log_name}_step{current_timesteps}.zip")

        print(new_model_path)
        model.save(new_model_path)

    hal9000.close()
