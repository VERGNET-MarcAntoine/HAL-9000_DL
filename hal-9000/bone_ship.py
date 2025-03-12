import gymnasium as gym
import numpy as np
from gymnasium import spaces
from websocket_client import SpaceshipWebSocketClient
import time


class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    def __init__(self, nb_planets, websocket_url: str = "ws://127.0.0.1:3012"):
        super().__init__()
        # !!!! Récuperer nb_planets automatiquement en regardant etat système


        # Client WebSocket pour communiquer avec le serveur
        self.client = SpaceshipWebSocketClient(websocket_url)

        # Espace d'action: 10 valeurs binaires (6 moteurs de translation + 4 moteurs de rotation)
        # [front, back, left, right, up, down, rot_left, rot_right, rot_up, rot_down]
        self.action_space = spaces.MultiBinary(10)

        # Espace d'observation:
        # - position (x, y, z)
        # - vitesse (vx, vy, vz)
        # - direction (dx, dy, dz)
        # - vitesse angulaire (wx, wy, wz)
        # - distance à chaque planète (n_planets valeurs)
        # !!!! on l'a pas pour le moment - vitesse relative à chaque planète (n_planets valeurs)
        self.max_planets = nb_planets
        obs_space_size = 12 + 1 * self.max_planets
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_space_size,), dtype=np.float32
        )

        # Variables d'état
        self.episode_steps = 0
        self.max_episode_steps = 1000

        # Objectifs de mission
        self.target_planet = None
        self.target_altitude = 1  # en unités de distance

    def step(self, action):
        
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """
        Réinitialise l'environnement au début d'un nouvel épisode.
        
        Returns:
            Tuple[np.ndarray, Dict]: Observation initiale et info supplémentaires
        """
        super().reset(seed=seed)
        
        # Connecter le client WebSocket si nécessaire
        if not self.client.connected:
            self.client.connect()
        
        # Réinitialiser les compteurs d'épisode
        self.episode_steps = 0
        
        # Choisir une planète cible aléatoire
        state = self.client.get_state()
        planets = state.get("planets", [])
        if planets:
            self.target_planet = np.random.choice(len(planets))
        else:
            self.target_planet = 0
        
        # Attendre quelques frames pour que le vaisseau soit bien initialisé
        time.sleep(0.1)
        
        # Récupérer l'observation initiale
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info


    def close(self):
        ...