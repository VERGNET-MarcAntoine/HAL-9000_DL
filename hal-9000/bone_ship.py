from stable_baselines3 import A2C
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from websocket_client import SpaceshipWebSocketClient
import time
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback


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
        # self.action_space = spaces.MultiBinary(10)
        self.action_space = spaces.MultiBinary(6)
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
            "front": bool(action[0]),
            "back": bool(action[1]),
            "left": bool(action[2]),
            "right": bool(action[3]),
            "up": bool(action[4]),
            "down": bool(action[5])
        }
        rotation = {
            "left": False, #bool(action[6]),
            "right": False, #bool(action[7]),
            "up": False, #bool(action[8]),
            "down": False, #bool(action[9])
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
    
    def _custom_print_info(self, info):
        print("-" * 40)
        print("    Environment Information    ")
        print("-" * 40)
        print(f"  Target Position: {info['target']}")
        print(f"  Current Target/Score: {info['current_target/score']}")
        print(f"  Step: {info['step']}")
        print(f"  Global Reward: {info['global_reward']:.2f}") # format to 2 decimal places
        print("-" * 40)
    
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
        time.sleep(0.1)

        self.state = self.client.get_state()

        self._ship_data = self._get_ship_data()
        self._planet_data = self._get_planet_data()
        
        self._target_ids = np.random.choice(np.arange(1, self.nb_planets), size=self.nb_planets - 1, replace=False)
        self._current_target = 0
        self._norme = self._get_norme()

        
        observation = self._get_obs()

        self._num_step = 0
        self._max_step = 3600*10*20
        self._global_reward = 0
        info = self._get_info()

        return observation, info
    

    def step(self, action):
        # Increase the number of step  
        self._num_step += 1

        distance_target = self._get_distance_target()


        # Send the command to the serveur
        command_engine, command_rotation = self._action_to_command(action)
        self.client.send_command(command_engine, command_rotation)
        
        time.sleep(0.005) # We can add sleep here

        # Get the new state after sending the command
        self.state = self.client.get_state()
        self._ship_data = self._get_ship_data()
        self._planet_data = self._get_planet_data()

        new_distance_target = self._get_distance_target()

        distance_sun = self._get_sun_distance()


        reward = (distance_target - new_distance_target)/self._norme

        terminated = False
        truncated = False


        if self._num_step > self._max_step:
            print("terminated")
            print(f"Distance with target: {new_distance_target}")
            truncated = True

        
        elif  distance_sun < 100:
            print("SUNBURN")
            print(f"distance with the sun: {distance_sun}")
            print(f"distance with target: {new_distance_target}")
            reward = -self._max_step/self._num_step
            terminated = True

        elif  distance_sun > 10000:
            print("BYEBYE")
            print(f"distance with the sun: {distance_sun}")
            print(f"distance with target: {new_distance_target}")
            reward = -self._max_step/self._num_step
            terminated = True

        elif new_distance_target < 100:
            reward = 100
            self._current_target += 1

            self._norme = self._get_norme()

            # For the log
            planets = self.state.get("planets", [])
            print(f"new target: {planets[self._target_ids[self._current_target]]}")
            print(f"number of step: {self._num_step}")
            print(f"Current score {self._current_target}")
        


        self._global_reward += reward

        observation = self._get_obs()
        info = self._get_info()
        
        if self._num_step % (100*60) == 0 or terminated or truncated:
            self._custom_print_info(info)
            print(f"target id: {self._target_ids[self._current_target]}")
            print(f"Distance with target: {new_distance_target}")

        return observation, reward, terminated, truncated, info

    def close(self):
        if self.client:
            if self.client.connected:
                self.client.disconnect()


if __name__ == "__main__":

    env = BoneShip()
    
    check_env(env)

    model = A2C("MultiInputPolicy", env, verbose=1)

    # Entraîner le modèle avec le callback
    model.learn(total_timesteps=36000000)

    env.close()
    """
    # ce qui ce passe dans learn de maniere non optimiser

    total_timesteps = 100000
    obs, info = env.reset()  # récupération des informations de reset
    print(f"reset info: {info}")

    for timestep in range(total_timesteps):
        action, _states = model.predict(obs, deterministic=True)
        new_obs, reward, done, truncated, info = env.step(
            action)  # récupération des information de step
        # afficher les informations de reset et step.
        print(f"step info: {info}")
        if done or truncated:
            obs, info = env.reset()  # récupération des informations de reset
        else:
            obs = new_obs
    """
