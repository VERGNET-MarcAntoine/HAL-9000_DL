import gymnasium as gym
import numpy as np
from gymnasium import spaces
from hal9000.websocket.websocket_client import SpaceshipWebSocketClient
import os

import time
from dotenv import load_dotenv
from stable_baselines3.common.env_checker import check_env


class Ship2D(gym.Env):
    """
    Classe de base pour l'environnement 2D du vaisseau spatial.

    Cette classe gère la communication avec un serveur WebSocket, la définition de l'espace d'action et d'observation,
    et la logique de base pour les étapes de l'environnement.

    Attributes:
        ws (str): L'URL du serveur WebSocket.
        max_step (int): Le nombre maximal d'étapes par épisode.
        step_time (float): Le temps d'attente entre chaque étape en secondes.
        client (SpaceshipWebSocketClient): Le client WebSocket pour la communication avec le serveur.
        nb_planets (int): Le nombre de planètes dans l'environnement.
        ship_data (np.ndarray): Les données du vaisseau (position, vitesse).
        planets_data (np.ndarray): Les données des planètes (position, vitesse).
        action_space (spaces.MultiBinary): L'espace d'action (haut, bas, gauche, droite).
        observation_space (spaces.Dict): L'espace d'observation (vaisseau, cible, planètes).
        target_ids (np.ndarray): Les identifiants des cibles à atteindre.
        current_target (int): L'indice de la cible actuelle.
        current_step (int): Le nombre d'étapes effectuées.
        score (float): Le score actuel.
        state (dict): L'état actuel de l'environnement.
    """

    def __init__(self, episode_time: int, step_time: float, websocket_url: str = "ws://127.0.0.1:3012"):
        """
        Initialise l'environnement Ship2D.

        Args:
            episode_time (int): La durée de l'épisode en minutes.
            step_time (float): Le temps d'attente entre chaque étape en secondes.
            websocket_url (str): L'URL du serveur WebSocket.
        """
        super().__init__()

        self.ws = websocket_url

        # On veut 4 frame par seconde et le temps episode est en minute
        self.max_step = episode_time*60*4
        self.step_time = step_time
        # Client WebSocket pour communiquer avec le serveur pour recuperer le nombre de planet
        self.client = SpaceshipWebSocketClient(self.ws)
        self.client.connect()
        state = self.client.get_state()
        self.client.disconnect()

        self.nb_planets = len(state["planets"])

        # les action possible pour le vaisseau haut bat gauche droit
        self.action_space = spaces.MultiBinary(4)

        self.observation_space = gym.spaces.Dict(
            {
                "ship": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64
                ),
                "target": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64
                ),
                "planets": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(4*self.nb_planets,), dtype=np.float64
                )
            }
        )

    def action_to_command(self, action: np.ndarray) -> tuple[dict[str, bool], dict[str, bool]]:
        """
        Convertit une action MultiBinary (tableau numpy) en dictionnaires séparés pour les moteurs et la rotation.

        Args:
            action (np.ndarray): L'action MultiBinary.

        Returns:
            tuple[dict[str, bool], dict[str, bool]]: Un tuple contenant les dictionnaires des moteurs et de la rotation.
        """
        engines = {
            "left": bool(action[0]),
            "right": bool(action[1]),
            "up": bool(action[2]),
            "down": bool(action[3]),
            "front": False,
            "back": False,
        }
        rotation = {
            "left": False,
            "right": False,
            "up": False,
            "down": False,
        }

        return engines, rotation

    def get_ship_data(self, state: dict) -> np.ndarray:
        """
        Extrait les données du vaisseau de l'état.

        Args:
            state (dict): L'état de l'environnement.

        Returns:
            np.ndarray: Les données du vaisseau (position, vitesse).
        """
        ship = state.get("ship", [])
        combined_data = []

        combined_data.extend(ship["position"][0:2])
        combined_data.extend(ship["speed"][0:2])

        return np.array(combined_data, dtype=np.float64)

    def get_planet_data(self, state: dict) -> tuple[np.ndarray, np.ndarray]:
        """
        Extrait les données des planètes et de la cible de l'état.

        Args:
            state (dict): L'état de l'environnement.

        Returns:
            tuple[np.ndarray, np.ndarray]: Un tuple contenant les données de la cible et des planètes.
        """
        planets = state.get("planets", [])
        planets_speeds = state.get("planet_speeds", [])

        target_data = []
        planet_data = []

        for i in range(self.nb_planets):
            if self.target_ids[self.current_target] == i:
                target_data.extend(planets[i][1][0:2])
                target_data.extend(planets_speeds[i][1][0:2])
            planet_data.extend(planets[i][1][0:2])
            planet_data.extend(planets_speeds[i][1][0:2])

        return np.array(target_data, dtype=np.float64), np.array(planet_data, dtype=np.float64)

    def get_observation(self, state: dict):  # -> dict[str, np.ndarray]:
        """
        Crée l'observation à partir de l'état.

        Args:
            state (dict): L'état de l'environnement.

        Returns:
            dict[str, np.ndarray]: Le dictionnaire d'observation.
        """
        ship_data = self.get_ship_data(state)
        target_data, planet_data = self.get_planet_data(state)
        return {
            "ship": ship_data,
            "target": target_data,
            "planets": planet_data
        }

    def reset(self, *, seed=None, options=None) -> tuple[dict[str, np.ndarray], None]:
        """
        Réinitialise l'environnement.

        Args:
            seed (int, optional): La graine aléatoire.
            options (dict, optional): Les options de réinitialisation.

        Returns:
            tuple[dict[str, np.ndarray], None]: L'observation initiale et les informations (None).
        """
        super().reset(seed=seed)
        if self.client:
            if self.client.connected:
                self.client.disconnect()

        self.client = SpaceshipWebSocketClient(self.ws)
        self.client.connect()

        time.sleep(0.01)

        # Initialisation du parcours du vaisseau
        self.target_ids = np.arange(2, self.nb_planets)

        # Initialisation variable a chaque rest
        self.current_target = 0
        self.current_step = 0
        self.score = 0

        self.state = self.client.get_state()
        observation = self.get_observation(self.state)
        info = {}
        return observation, info

    def get_reward(self, previous_state: dict) -> tuple[float, bool]:
        """
        Calcule la récompense et détermine si l'épisode est terminé.
        Args:
            previous_state (dict): L'état précédent de l'environnement.

        Returns:
            tuple[float, bool]: La récompense et un booléen indiquant si l'épisode est terminé.
        """
        reward = 1
        terminated = False
        return reward, terminated

    def step(self, action: np.ndarray) -> tuple[dict[str, np.ndarray], float, bool, bool, None]:
        """
        Effectue une étape dans l'environnement.

        Args:
            action (np.ndarray): L'action à effectuer.

        Returns:
            tuple[dict[str, np.ndarray], float, bool, bool, None]: L'observation, la récompense, si l'épisode est terminé,
                                                             si l'épisode est tronqué, et les informations (None).
        """
        terminated = False
        truncated = False

        # Gestion du nombre de step et stop quand max_step atteint
        self.current_step += 1
        if self.current_step > self.max_step:
            truncated = True

        engine_command, rotation_command = self.action_to_command(action)
        self.client.send_command(engine_command, rotation_command)
        time.sleep(self.step_time)

        previous_state = self.state
        self.state = self.client.get_state()

        reward, terminated = self.get_reward(previous_state)

        observation = self.get_observation(self.state)
        info = {}
        return observation, reward, terminated, truncated, info

    def close(self):
        """
        Ferme la connexion WebSocket.
        """
        if self.client:
            if self.client.connected:
                self.client.disconnect()


if __name__ == "__main__":
    load_dotenv()
    step_time = float(os.getenv("SLEEP_TIME"))
    episode_time = int(os.getenv("EPISODE_TIME"))

    env = Ship2D(episode_time, step_time)
    check_env(env)
