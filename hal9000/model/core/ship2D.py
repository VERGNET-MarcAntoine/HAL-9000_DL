import time

import gymnasium as gym
import numpy as np
from stable_baselines3.common.env_checker import check_env

from hal9000.config import Config, load_config
from hal9000.model import Hal9000_2D as task
from hal9000.websocket.websocket_client import SpaceshipWebSocketClient

# A step is late if it exceeds step_time by more than 10%
OVERRUN_TOLERANCE = 1.1
# Frequency (in steps) of the lateness check, and accepted share of late steps
OVERRUN_CHECK_STEPS = 1000
OVERRUN_MAX_RATIO = 0.2
NO_ROTATION = {"left": False, "right": False, "up": False, "down": False}


def state_to_arrays(state: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts the 2D data of a server state, in the form expected by the task (n = 1).

    Returns:
        tuple: Positions (1, 6, 2) and speeds (1, 6, 2) of the bodies, position (1, 2) and speed (1, 2) of the ship.
    """
    planet_pos = np.array([p[1][:2] for p in state["planets"]])[None]
    planet_vel = np.array([p[1][:2] for p in state["planet_speeds"]])[None]
    ship_pos = np.array(state["ship"]["position"][:2])[None]
    ship_vel = np.array(state["ship"]["speed"][:2])[None]
    return planet_pos, planet_vel, ship_pos, ship_vel


class Ship2D(gym.Env):
    """
    Environment of the Hal9000_2D task connected to the Rust server: each episode spawns a new ship
    (a WebSocket connection), flown at the rhythm of decision_interval simulated seconds.

    Attributes:
        config (Config): The configuration (see config.toml).
        max_step (int): The maximum number of steps per episode.
        step_time (float): The real time between two steps, in seconds.
        client (SpaceshipWebSocketClient): The WebSocket client of the current ship.
        task (Hal9000Task): The state of the task (targets, progress) for this ship.
        state (dict): The last state received from the server.
    """

    observation_space = task.observation_space
    action_space = task.action_space

    def __init__(self, config: Config | None = None):
        """
        Initializes the Ship2D environment.

        Args:
            config (Config, optional): The configuration (default: the one of config.toml).
        """
        super().__init__()
        self.config = config or load_config()
        self.max_step = self.config.steps_per_episode
        self.step_time = self.config.step_time
        self.total_steps = 0
        self.overrun_steps = 0
        self.current_step = 0
        self.last_step_time = 0.0
        self.state: dict = {}
        self.client: SpaceshipWebSocketClient | None = None
        self.task = task.Hal9000Task(1, np.random.default_rng(), self.config.reward)

    def _observe(self) -> np.ndarray:
        planet_pos, planet_vel, ship_pos, ship_vel = state_to_arrays(self.state)
        return self.task.observe(planet_pos, planet_vel, ship_pos, ship_vel)[0]

    def reset(self, *, seed=None, options=None) -> tuple[np.ndarray, dict]:
        """
        Resets the environment: disconnects the previous ship and spawns a new one.

        Args:
            seed (int, optional): The random seed (order of the targets).
            options (dict, optional): The reset options.

        Returns:
            tuple[np.ndarray, dict]: The initial observation and the information.
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
        Sends the chosen thrust, waits decision_interval simulated seconds and computes the reward.

        Args:
            action (int): The Discrete(9) action (thrust direction).

        Returns:
            tuple: The observation, the reward, whether the episode is terminated (death), whether it is
                truncated (maximum duration), and the information (statistics at the end of the episode).
        """
        assert self.client is not None, "reset() must be called before step()"
        self.client.send_command(task.thrust_to_engines(task.THRUSTS[int(action)]), NO_ROTATION)
        # Wait until step_time after the previous step (and not step_time after this point): the Python
        # computation time is absorbed and each step really lasts decision_interval simulated seconds
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
        # The server does not know the targets: they are sent for the display (display_ship2D --server)
        info: dict = {"ship": {"uuid": self.client.ship_uuid, "target": int(self.task.target()[0]),
                               "reached": int(self.task.index[0])}}
        if terminated or truncated:
            info["hal"] = {"targets": int(self.task.index[0]), "dead_sun": bool(events["dead_sun"][0]),
                           "dead_out": bool(events["dead_out"][0])}
        return self._observe(), float(reward[0]), terminated, truncated, info

    def check_overrun(self, elapsed: float):
        """
        Reports when Python cannot keep the requested rhythm: the steps then last more than
        decision_interval simulated seconds and the training is no longer synchronized with the simulation.

        Args:
            elapsed (float): The real time elapsed since the previous step, in seconds.
        """
        self.total_steps += 1
        if elapsed > self.step_time * OVERRUN_TOLERANCE:
            self.overrun_steps += 1
        if self.total_steps % OVERRUN_CHECK_STEPS == 0:
            ratio = self.overrun_steps / OVERRUN_CHECK_STEPS
            if ratio > OVERRUN_MAX_RATIO:
                print(f"Warning: {ratio:.0%} of the steps exceed {self.step_time * 1000:.1f} ms, "
                      "Python cannot keep up with the simulation. Lower [training] speedup or n_envs in config.toml.")
            self.overrun_steps = 0

    def close(self):
        """
        Closes the WebSocket connection (the server then removes the ship).
        """
        if self.client and self.client.connected:
            self.client.disconnect()


if __name__ == "__main__":
    check_env(Ship2D())
