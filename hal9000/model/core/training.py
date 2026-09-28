import glob
import os
import time
from dataclasses import replace
from functools import partial

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv, VecMonitor

from hal9000.config import Config
from hal9000.model.core.loading import load_ppo
from hal9000.model.core.ship2D import Ship2D
from hal9000.sim.live import LivePublisher
from hal9000.sim.vec_env import Hal9000SimVecEnv
from hal9000.websocket.websocket_client import SpaceshipWebSocketClient

LOG_DIR = "logs"
MODELS_DIR = "models"
MODEL_NAME = "Hal9000_2D"
# Accepted gap between the measured speed of the server and the speed of the configuration
SPEEDUP_TOLERANCE = 0.2
# The network is too small to benefit from the GPU: on CPU with a single thread, each step and each
# PPO update are ~4x faster (less latency, hence a synchronization that holds)
DEVICE = "cpu"


def _orbit_angle(state: dict) -> tuple[float, float]:
    """Angle and angular speed of the farthest planet around the sun."""
    position = np.array(state["planets"][-1][1][:2]) - np.array(state["planets"][0][1][:2])
    speed = np.array(state["planet_speeds"][-1][1][:2])
    return float(np.arctan2(position[1], position[0])), float(np.linalg.norm(speed) / np.linalg.norm(position))


def measure_server_speedup(config: Config, duration: float = 2.0) -> float:
    """
    Measures the actual speed of the server from the orbital motion of a planet.

    Args:
        config (Config): The configuration (for the URL of the server).
        duration (float): The duration of the measurement, in real seconds.

    Returns:
        float: The number of simulated seconds per real second.
    """
    client = SpaceshipWebSocketClient(config.websocket_url)
    client.connect()
    try:
        angle_start, angular_speed = _orbit_angle(client.get_state())
        start = time.perf_counter()
        time.sleep(duration)
        angle_end, _ = _orbit_angle(client.get_state())
        elapsed = time.perf_counter() - start
    finally:
        client.disconnect()
    return ((angle_end - angle_start) % (2 * np.pi)) / angular_speed / elapsed


def server_config(config: Config) -> Config:
    """
    Checks that the server runs at the expected speed and returns the configuration paced on its
    measured speed (the server never quite reaches the requested speed).
    """
    measured_speedup = measure_server_speedup(config)
    if abs(measured_speedup / config.speedup - 1) > SPEEDUP_TOLERANCE:
        raise SystemExit(
            f"The server runs at x{measured_speedup:.0f} while x{config.speedup:g} is expected.\n"
            "Start it with: uv run python -m hal9000.server" + (" --train" if config.speedup > 1 else ""))
    print(f"Server at x{measured_speedup:.1f}")
    return replace(config, speedup=measured_speedup)


def make_env(config: Config, backend: str, n_envs: int | None = None, seed: int | None = None,
             publisher: LivePublisher | None = None) -> VecEnv:
    """
    Creates the training or evaluation ships.

    Args:
        config (Config): The configuration (at the desired speed for the server).
        backend (str): "sim" (numpy simulation) or "server" (Rust server).
        n_envs (int, optional): The number of ships (default: the one of the configuration for this backend).
        seed (int, optional): The random seed.
        publisher (LivePublisher, optional): Live broadcast of the ships (simulation only).

    Returns:
        VecEnv: The environments, with the episode statistics (VecMonitor).
    """
    if backend == "sim":
        sim_env = Hal9000SimVecEnv(n_envs or config.sim_envs, config, seed)
        sim_env.publisher = publisher
        return VecMonitor(sim_env)
    config = server_config(config)
    n_envs = n_envs or config.n_envs
    vec_env_class = DummyVecEnv if n_envs == 1 else SubprocVecEnv
    env = vec_env_class([partial(Ship2D, config) for _ in range(n_envs)])
    env.seed(seed)
    return VecMonitor(env)


def ppo_kwargs(config: Config, n_envs: int) -> dict:
    """PPO hyperparameters of the configuration, with rollouts of the same size whatever n_envs."""
    kwargs = dict(config.ppo)
    rollout_size = kwargs.pop("rollout_size")
    net_arch = kwargs.pop("net_arch")
    return dict(n_steps=max(1, rollout_size // n_envs), policy_kwargs={"net_arch": list(net_arch)}, **kwargs)


def latest_model(name: str = MODEL_NAME) -> str | None:
    """The latest model saved under this name, or None."""
    models = glob.glob(os.path.join(MODELS_DIR, f"{name}_*.zip"))
    return max(models, key=os.path.getmtime) if models else None


class HalMetrics(BaseCallback):
    """
    Records in TensorBoard what really matters, whatever the shape of the reward: planets reached per
    episode, survival rate and causes of death.
    """

    def __init__(self):
        super().__init__()
        self.episodes: list[dict] = []

    def _on_rollout_start(self):
        self.episodes = []

    def _on_step(self) -> bool:
        self.episodes += [info["hal"] for info in self.locals["infos"] if "hal" in info]
        return True

    def _on_rollout_end(self):
        if not self.episodes:
            return
        targets = np.array([e["targets"] for e in self.episodes])
        dead_sun = np.array([e["dead_sun"] for e in self.episodes])
        dead_out = np.array([e["dead_out"] for e in self.episodes])
        self.logger.record("hal/targets_per_episode", targets.mean())
        self.logger.record("hal/survival_rate", 1 - (dead_sun | dead_out).mean())
        self.logger.record("hal/death_sun_rate", dead_sun.mean())
        self.logger.record("hal/death_out_rate", dead_out.mean())
        self.logger.record("hal/episodes", len(self.episodes))


def train(config: Config, backend: str, timesteps: int, new: bool = False, model_path: str | None = None,
          name: str = MODEL_NAME):
    """
    Trains Hal9000_2D, resuming the latest saved model unless new is true.

    Args:
        config (Config): The configuration.
        backend (str): "sim" (numpy simulation) or "server" (Rust server, started with --train).
        timesteps (int): The number of steps to train.
        new (bool): Start from a new model.
        model_path (str, optional): The model to resume (default: the latest saved under this name).
        name (str): The name of the model (prefix of the checkpoints and of the TensorBoard curves).
    """
    torch.set_num_threads(1)
    if backend == "server":
        config = config.for_training()
    n_envs = config.sim_envs if backend == "sim" else config.n_envs
    # On the simulation, the training can be watched live with `display_ship2D --training`
    env = make_env(config, backend, publisher=LivePublisher(name) if backend == "sim" else None)
    print(f"{n_envs} ships in parallel ({backend})")

    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = None if new else (model_path or latest_model(name))
    if model_path:
        model = load_ppo(model_path, env, config, n_envs, device=DEVICE)
        print(f"Model resumed: {model_path} ({model.num_timesteps} steps)")
    else:
        model = PPO("MlpPolicy", env, tensorboard_log=LOG_DIR, device=DEVICE, **ppo_kwargs(config, n_envs))
        print("New model created.")

    name = f"{name}_{backend}"
    checkpoint = CheckpointCallback(save_freq=max(1, config.save_every // n_envs), save_path=MODELS_DIR, name_prefix=name)
    # A new model gets its own log directory; a resumed one continues the curves of the previous directory
    model.learn(total_timesteps=timesteps, callback=[HalMetrics(), checkpoint],
                tb_log_name=name, reset_num_timesteps=model_path is None)
    final_path = os.path.join(MODELS_DIR, f"{name}_{model.num_timesteps}_steps.zip")
    model.save(final_path)
    print(f"Model saved: {final_path}")
    env.close()
