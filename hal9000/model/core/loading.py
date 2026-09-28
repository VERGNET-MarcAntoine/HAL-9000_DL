from stable_baselines3 import PPO

# Les .zip SB3 contiennent les fonctions lr_schedule/clip_range sérialisées avec cloudpickle.
# Les désérialiser sous une autre version de Python que celle de l'entraînement fait planter
# l'interpréteur (segfault), et exécute du code arbitraire si le fichier n'est pas de confiance.
# On fournit donc directement les valeurs utilisées à l'entraînement (défauts de PPO) :
# SB3 reconstruit les schedules à partir de ces valeurs sans dépickler les fonctions.
PPO_CUSTOM_OBJECTS = {
    "learning_rate": 3e-4,
    "lr_schedule": 3e-4,
    "clip_range": 0.2,
}


def load_ppo(path: str, env, device: str = "auto") -> PPO:
    """
    Charge un modèle PPO sauvegardé sans désérialiser ses schedules picklés.

    Attention : le .zip contient encore d'autres objets picklés (espaces, classe de policy),
    ne charger que des modèles de confiance.

    Args:
        path (str): Le chemin du modèle (.zip).
        env: L'environnement à associer au modèle.
        device (str): Le device torch ("cpu", "cuda" ou "auto").

    Returns:
        PPO: Le modèle chargé.
    """
    return PPO.load(path, env=env, device=device, custom_objects=PPO_CUSTOM_OBJECTS)
