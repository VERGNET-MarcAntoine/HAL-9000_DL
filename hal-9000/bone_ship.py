from stable_baselines3 import A2C, PPO
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from websocket_client import SpaceshipWebSocketClient
import time
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
import pprint


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
        self._max_step = 60*10*4

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
        self._max_step = 60*10*4
        self._global_reward = 0
        info = self._get_info()

        self.score = 0
        return observation, info

    def step(self, action):
        # Increase the number of step
        self._num_step += 1

        distance_target = self._get_distance_target()
        distance_sun = self._get_sun_distance()

        # Send the command to the serveur
        command_engine, command_rotation = self._action_to_command(action)
        self.client.send_command(command_engine, command_rotation)

        time.sleep(0.005)  # the sleep between 2 frame

        command_engine = {
            "front": False,  # bool(action[0]),
            "back": False,  # bool(action[1]),
            "left": False,
            "right": False,
            "up": False,
            "down": False
        }
        command_rotation = {
            "left": False,  # bool(action[6]),
            "right": False,  # bool(action[7]),
            "up": False,  # bool(action[8]),
            "down": False,  # bool(action[9])
        }

        self.client.send_command(command_engine, command_rotation)

        # Get the new state after sending the command
        self.state = self.client.get_state()
        self._ship_data = self._get_ship_data()
        self._planet_data = self._get_planet_data()

        new_distance_target = self._get_distance_target()

        new_distance_sun = self._get_sun_distance()

        reward = (distance_sun - new_distance_sun)/new_distance_sun

        terminated = False
        truncated = False

        if self._num_step > self._max_step:
            truncated = True

        elif distance_sun < 500:
            reward = -1

        elif distance_sun < 2000:
            reward = 1
            self.score += 1

        elif distance_sun > 20000:
            reward = -1

        self._global_reward += reward

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def close(self):
        if self.client:
            if self.client.connected:
                self.client.disconnect()


if __name__ == "__main__":

    logdir = "logs"
    models_dir = "models"

    env = BoneShip()

    model = PPO("MultiInputPolicy", env, verbose=1,
                tensorboard_log=logdir, device="cpu")

    max_episode = env._max_step * 150

    # Entraîner le modèle avec le callback
    TIMESTEPS = max_episode / 10

    for i in range(10):
        model.learn(total_timesteps=TIMESTEPS,
                    reset_num_timesteps=False, tb_log_name="PPO")
        model.save(f"{models_dir}/{TIMESTEPS*i}")

    env.close()
