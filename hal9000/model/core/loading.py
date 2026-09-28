from stable_baselines3 import PPO

from hal9000.config import Config


def load_ppo(path: str, env, config: Config, n_envs: int, device: str = "cpu") -> PPO:
    """
    Loads a saved PPO model with the hyperparameters of the configuration.

    SB3 .zip files contain the lr_schedule/clip_range functions serialized with cloudpickle. Unpickling
    them with another Python version than the one used for training crashes the interpreter (segfault),
    and executes arbitrary code if the file is not trusted. Their values are therefore provided directly:
    SB3 rebuilds the schedules without unpickling them. Warning: the .zip still contains other pickled
    objects (spaces, policy class), only load trusted models.

    Args:
        path (str): The path of the model (.zip).
        env: The environment to attach to the model.
        config (Config): The configuration ([ppo] section).
        n_envs (int): The number of ships of env (for the size of the rollouts).
        device (str): The torch device ("cpu", "cuda" or "auto").

    Returns:
        PPO: The loaded model.
    """
    custom_objects = {
        "learning_rate": config.ppo["learning_rate"],
        "lr_schedule": config.ppo["learning_rate"],
        "clip_range": config.ppo["clip_range"],
        "n_steps": max(1, config.ppo["rollout_size"] // n_envs),
    }
    return PPO.load(path, env=env, device=device, custom_objects=custom_objects)
