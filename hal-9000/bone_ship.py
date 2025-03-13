from stable_baselines3 import A2C
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from websocket_client import SpaceshipWebSocketClient
import time
import pprint
from stable_baselines3.common.env_checker import check_env


class BoneShip(gym.Env):
    """Custom Environment that follows gym interface."""

    def __init__(self, websocket_url: str = "ws://127.0.0.1:3012"):
        super().__init__()

        self.ws = websocket_url

        # Client WebSocket pour communiquer avec le serveur pour recuperer le nombre de planet
        client = SpaceshipWebSocketClient(websocket_url)

        client.connect()
        init_state = client.get_state()
        client.disconnect()

        self.client = None
        # Récuperer nb_planets automatiquement en regardant etat système -1 car le soleil est considerer comme une planete
        self.nb_planets = len(init_state["planets"])

        # initialisation des variable
        # Initialize with zeros, size 9
        self._ship_data = np.zeros(9, dtype=np.float64)
        # Initialize with zeros, size nb_planets * 6
        self._planets_data = np.zeros(self.nb_planets * 6, dtype=np.float64)

        # Espace d'action: 10 valeurs binaires (6 moteurs de translation + 4 moteurs de rotation)
        # [front, back, left, right, up, down, rot_left, rot_right, rot_up, rot_down]
        self.action_space = spaces.MultiBinary(10)

        # Espace d'observation:
        # Planet speeds (x, y, z) et positions (x, y, z)
        obs_planets_size = self.nb_planets * 6
        self.observation_space = gym.spaces.Dict(
            {
                "ship": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(9,), dtype=np.float64
                ),
                "planets": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(obs_planets_size,), dtype=np.float64
                )
            }
        )
        # Variables d'état
        # self.episode_steps = 0
        # self.max_episode_steps = 1000

        # Objectifs de mission
        # self.target_planet = None
        # self.target_altitude = 1  # en unités de distance

    def _action_to_command(self, action):
        """
        Converts a MultiBinary action (numpy array) to separate engine and rotation dictionaries.
        """
        engines = {
            "front": bool(action[0]),
            "back": bool(action[1]),
            "left": bool(action[2]),
            "right": bool(action[3]),
            "up": bool(action[4]),
            "down": bool(action[5])
        }
        rotation = {
            "left": bool(action[6]),
            "right": bool(action[7]),
            "up": bool(action[8]),
            "down": bool(action[9])
        }
        return engines, rotation

    def _get_planet_data(self):
        planets = self.state.get("planets", [])
        planet_speeds = self.state.get("planet_speeds", [])
        combined_data = []

        for i in range(self.nb_planets):
            combined_data.extend(planets[i][1])  # Add location
            combined_data.extend(planet_speeds[i][1])  # Add speed

        return np.array(combined_data)

    def _get_ship_data(self):
        ship = self.state.get("ship", [])
        combined_data = []

        combined_data.extend(ship["position"])
        combined_data.extend(ship["speed"])
        combined_data.extend(ship["direction"])

        return np.array(combined_data)

    def _get_obs(self):
        return {"ship": self._ship_data, "planets": self._planet_data}

    def _get_info(self):
        return {
            "target": self._target_planet
        }

    def reset(self, seed=None, options=None):
        """
        Réinitialise l'environnement au début d'un nouvel épisode.

        Returns:
            Tuple[np.ndarray, Dict]: Observation initiale et info supplémentaires
        """
        super().reset(seed=seed)

        if self.client:
            if self.client.connected:
                self.client.disconnect()

        self.client = SpaceshipWebSocketClient(self.ws)
        self.client.connect()
        time.sleep(0.1)

        self.state = self.client.get_state()

        self._ship_data = self._get_ship_data()
        self._planet_data = self._get_planet_data()

        observation = self._get_obs()

        # QUOI METTRE DANS INFO pour l'instant la target peut etre autre chose plus tard
        planets = self.state.get("planets", [])
        self._target_planet = planets[np.random.choice(len(planets)-1) + 1]

        info = self._get_info()

        return observation, info

    def step(self, action):
        command_engine, command_rotation = self._action_to_command(action)
        self.client.send_command(command_engine, command_rotation)
        self.state = self.client.get_state()

        self._ship_data = self._get_ship_data()
        self._planet_data = self._get_planet_data()

        terminated = False
        truncated = False
        reward = 1 if terminated else 0

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def close(self):
        ...


if __name__ == "__main__":

    env = BoneShip()
    model = A2C("MultiInputPolicy", env, verbose=1).learn(
        10000)    # env.reset()
    # It will check your custom environment and output additional warnings if needed
    # check_env(env)
