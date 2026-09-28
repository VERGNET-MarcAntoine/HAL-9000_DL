import time

import gymnasium as gym
import numpy as np
from stable_baselines3.common.env_checker import check_env

from hal9000.config import Config, load_config
from hal9000.model import Hal9000_2D as task
from hal9000.websocket.websocket_client import SpaceshipWebSocketClient

# Un step est en retard s'il dépasse step_time de plus de 10 %
OVERRUN_TOLERANCE = 1.1
# Fréquence (en steps) de la vérification du retard, et part de steps en retard tolérée
OVERRUN_CHECK_STEPS = 1000
OVERRUN_MAX_RATIO = 0.2
NO_ROTATION = {"left": False, "right": False, "up": False, "down": False}


def state_to_arrays(state: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extrait les données 2D d'un état du serveur, sous la forme attendue par la tâche (n = 1).

    Returns:
        tuple: Positions (1, 6, 2) et vitesses (1, 6, 2) des corps, position (1, 2) et vitesse (1, 2) du vaisseau.
    """
    planet_pos = np.array([p[1][:2] for p in state["planets"]])[None]
    planet_vel = np.array([p[1][:2] for p in state["planet_speeds"]])[None]
    ship_pos = np.array(state["ship"]["position"][:2])[None]
    ship_vel = np.array(state["ship"]["speed"][:2])[None]
    return planet_pos, planet_vel, ship_pos, ship_vel


class Ship2D(gym.Env):
    """
    Environnement de la tâche Hal9000_2D connecté au serveur Rust : chaque épisode fait apparaître un
    nouveau vaisseau (une connexion WebSocket), piloté au rythme de decision_interval simulé.

    Attributes:
        config (Config): La configuration (voir config.toml).
        max_step (int): Le nombre maximal d'étapes par épisode.
        step_time (float): Le temps réel entre chaque étape en secondes.
        client (SpaceshipWebSocketClient): Le client WebSocket du vaisseau courant.
        task (Hal9000Task): L'état de la tâche (cibles, potentiel) pour ce vaisseau.
        state (dict): Le dernier état reçu du serveur.
    """

    observation_space = task.observation_space
    action_space = task.action_space

    def __init__(self, config: Config | None = None):
        """
        Initialise l'environnement Ship2D.

        Args:
            config (Config, optional): La configuration (par défaut, celle de config.toml).
        """
        super().__init__()
        self.config = config or load_config()
        self.max_step = self.config.steps_per_episode
        self.step_time = self.config.step_time
        self.total_steps = 0
        self.overrun_steps = 0
        self.client = None
        self.task = task.Hal9000Task(1, np.random.default_rng(), self.config.reward)

    def _observe(self) -> np.ndarray:
        planet_pos, planet_vel, ship_pos, ship_vel = state_to_arrays(self.state)
        return self.task.observe(planet_pos, planet_vel, ship_pos, ship_vel)[0]

    def reset(self, *, seed=None, options=None) -> tuple[np.ndarray, dict]:
        """
        Réinitialise l'environnement : déconnecte le vaisseau précédent et en fait apparaître un nouveau.

        Args:
            seed (int, optional): La graine aléatoire (ordre des cibles).
            options (dict, optional): Les options de réinitialisation.

        Returns:
            tuple[np.ndarray, dict]: L'observation initiale et les informations.
        """
        super().reset(seed=seed)
        self.task.rng = self.np_random
        self.close()
        self.client = SpaceshipWebSocketClient(self.config.websocket_url)
        self.client.connect()

        self.current_step = 0
        self.state = self.client.get_state()
        self.last_step_time = time.perf_counter()
        planet_pos, _, ship_pos, _ = state_to_arrays(self.state)
        self.task.reset(np.array([0]), planet_pos, ship_pos)
        return self._observe(), {}

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Envoie la poussée choisie, attend decision_interval simulé et calcule la récompense.

        Args:
            action (int): L'action Discrete(9) (direction de poussée).

        Returns:
            tuple: L'observation, la récompense, si l'épisode est terminé (mort), s'il est tronqué
                (durée maximale), et les informations (statistiques en fin d'épisode).
        """
        self.client.send_command(task.thrust_to_engines(task.THRUSTS[int(action)]), NO_ROTATION)
        # On attend jusqu'à step_time après le step précédent (et non step_time après ce point) :
        # le temps de calcul Python est absorbé et chaque step dure bien decision_interval simulé
        elapsed = time.perf_counter() - self.last_step_time
        time.sleep(max(0.0, self.step_time - elapsed))
        self.check_overrun(elapsed)

        self.state = self.client.get_state()
        self.last_step_time = time.perf_counter()
        planet_pos, _, ship_pos, _ = state_to_arrays(self.state)
        reward, dead, events = self.task.transition(planet_pos, ship_pos)

        self.current_step += 1
        terminated = bool(dead[0])
        truncated = self.current_step >= self.max_step and not terminated
        # Le serveur ne connaît pas les cibles : on les transmet pour l'affichage (display_ship2D --server)
        info = {"ship": {"uuid": self.client.ship_uuid, "target": int(self.task.target()[0]),
                         "reached": int(self.task.index[0])}}
        if terminated or truncated:
            info["hal"] = {"targets": int(self.task.index[0]), "dead_sun": bool(events["dead_sun"][0]),
                           "dead_out": bool(events["dead_out"][0])}
        return self._observe(), float(reward[0]), terminated, truncated, info

    def check_overrun(self, elapsed: float):
        """
        Signale quand Python n'arrive pas à tenir le rythme demandé : les steps durent alors plus
        que decision_interval simulé et l'entraînement n'est plus synchronisé avec la simulation.

        Args:
            elapsed (float): Le temps réel écoulé depuis le step précédent, en secondes.
        """
        self.total_steps += 1
        if elapsed > self.step_time * OVERRUN_TOLERANCE:
            self.overrun_steps += 1
        if self.total_steps % OVERRUN_CHECK_STEPS == 0:
            ratio = self.overrun_steps / OVERRUN_CHECK_STEPS
            if ratio > OVERRUN_MAX_RATIO:
                print(f"Attention : {ratio:.0%} des steps dépassent {self.step_time * 1000:.1f} ms, "
                      "Python ne suit pas la simulation. Baissez [training] speedup ou n_envs dans config.toml.")
            self.overrun_steps = 0

    def close(self):
        """
        Ferme la connexion WebSocket (le serveur supprime alors le vaisseau).
        """
        if self.client and self.client.connected:
            self.client.disconnect()


if __name__ == "__main__":
    check_env(Ship2D())
