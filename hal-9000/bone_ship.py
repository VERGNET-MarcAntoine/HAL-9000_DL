import gymnasium as gym
import numpy as np
from gymnasium import spaces
from websocket_client import SpaceshipWebSocketClient
import time
import os
from dotenv import load_dotenv

load_dotenv()

sleep_time = float(os.getenv("SLEEP_TIME"))
episode_time = int(os.getenv("EPISODE_TIME"))


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

        self._max_step = episode_time*60*4

        # initialisation des variable
        # Initialize with zeros, size 9
        self._ship_data = np.zeros(9, dtype=np.float64)
        # Initialize with zeros, size nb_planets * 6
        self._planets_data = np.zeros(self.nb_planets * 6, dtype=np.float64)
        # Espace d'action: 10 valeurs binaires (6 moteurs de translation + 4 moteurs de rotation)
        # [front, back, left, right, up, down, rot_left, rot_right, rot_up, rot_down]
        # self.action_space = spaces.MultiBinary(10)
        self.action_space = spaces.MultiBinary(4)
        # Espace d'observation:
        # Planet speeds (x, y, z) et positions (x, y, z)
        obs_planets_size = self.nb_planets * 6
        self.observation_space = gym.spaces.Dict(
            {
                "ship": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(9,), dtype=np.float64
                ),
                "target": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64
                ),
                "planets": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(obs_planets_size,), dtype=np.float64
                )
            }
        )

    def _action_to_command(self, action):
        """
        Converts a MultiBinary action (numpy array) to separate engine and rotation dictionaries.
        """
        engines = {
            "front": False,  # bool(action[0]),
            "back": False,  # bool(action[1]),
            "left": bool(action[0]),
            "right": bool(action[1]),
            "up": bool(action[2]),
            "down": bool(action[3])
        }
        rotation = {
            "left": False,  # bool(action[6]),
            "right": False,  # bool(action[7]),
            "up": False,  # bool(action[8]),
            "down": False,  # bool(action[9])
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
        return {
            "ship": self._ship_data,
            "target": np.array(self._planet_data[self._target_ids[self._current_target]*6:self._target_ids[self._current_target]*6+3], dtype=np.float64),
            "planets": self._planet_data
        }

    def _get_info(self):
        return {
            "target": self._planet_data[self._target_ids[self._current_target]*6:self._target_ids[self._current_target]*6+3],
            "current_target/score": self._current_target,
            "step": self._num_step,
            "global_reward": self._global_reward
        }

    def _get_norme(self):
        return np.linalg.norm(self._planet_data[0:3] - self._planet_data[self._target_ids[self._current_target]*6:self._target_ids[self._current_target]*6+3])

    def _get_distance_target(self):
        return np.linalg.norm(self._ship_data[0:3] - self._planet_data[self._target_ids[self._current_target]*6:self._target_ids[self._current_target]*6+3])

    def _get_sun_distance(self):
        return np.linalg.norm(self._ship_data[0:3] - self._planet_data[0:3])

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
        # waiting the initialisation of the client
        time.sleep(0.1)

        self.state = self.client.get_state()

        self._ship_data = self._get_ship_data()
        self._planet_data = self._get_planet_data()

        self._target_ids = np.random.choice(
            np.arange(1, self.nb_planets), size=self.nb_planets - 1, replace=False)
        self._current_target = 0
        self._norme = self._get_norme()

        observation = self._get_obs()

        self._num_step = 0
        self._global_reward = 0
        info = self._get_info()

        self.score = 0
        return observation, info

    def step(self, action):
        # Increase the number of step
        terminated = False
        truncated = False
        self._num_step += 1

        command_engine, command_rotation = self._action_to_command(action)
        self.client.send_command(command_engine, command_rotation)

        time.sleep(sleep_time)  # the sleep between 2 frame

       # Get the new state after sending the command
        self.state = self.client.get_state()
        self._ship_data = self._get_ship_data()
        self._planet_data = self._get_planet_data()

        new_distance_sun = self._get_sun_distance()

        distance_target = self._get_distance_target()

        reward = - distance_target / 20000

        if self._num_step > self._max_step:
            truncated = True

        elif new_distance_sun < 200:
            reward = -1

        elif distance_target < 200:
            reward = 100
            self._current_target += 1
            print(f"Score : {self._current_target}")

        self._global_reward += reward

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def close(self):
        if self.client:
            if self.client.connected:
                self.client.disconnect()
