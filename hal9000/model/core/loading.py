from stable_baselines3 import PPO

from hal9000.config import Config


def load_ppo(path: str, env, config: Config, n_envs: int, device: str = "cpu") -> PPO:
    """
    Charge un modèle PPO sauvegardé en reprenant les hyperparamètres de la configuration.

    Les .zip SB3 contiennent les fonctions lr_schedule/clip_range sérialisées avec cloudpickle.
    Les désérialiser sous une autre version de Python que celle de l'entraînement fait planter
    l'interpréteur (segfault), et exécute du code arbitraire si le fichier n'est pas de confiance.
    On fournit donc directement leurs valeurs : SB3 reconstruit les schedules sans les dépickler.
    Attention : le .zip contient encore d'autres objets picklés (espaces, classe de policy),
    ne charger que des modèles de confiance.

    Args:
        path (str): Le chemin du modèle (.zip).
        env: L'environnement à associer au modèle.
        config (Config): La configuration (section [ppo]).
        n_envs (int): Le nombre de vaisseaux de env (pour la taille des rollouts).
        device (str): Le device torch ("cpu", "cuda" ou "auto").

    Returns:
        PPO: Le modèle chargé.
    """
    custom_objects = {
        "learning_rate": config.ppo["learning_rate"],
        "lr_schedule": config.ppo["learning_rate"],
        "clip_range": config.ppo["clip_range"],
        "n_steps": max(1, config.ppo["rollout_size"] // n_envs),
    }
    return PPO.load(path, env=env, device=device, custom_objects=custom_objects)
