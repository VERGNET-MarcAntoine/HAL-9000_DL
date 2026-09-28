import os
from hal9000.model.Hal9000_2D_PAT_V1 import Hal9000_2D_V0
from hal9000.model.core.loading import load_ppo
from hal9000.config import load_config


model = "Hal9000_2D_PAT_V1_2025_03_31_03_23_04_step6588000"


# Charger le modèle
models_dir = "models"
model_path = os.path.join(models_dir, model)

# Initialiser l'environnement
# Assurez-vous que l'environnement peut afficher les résultats
env = Hal9000_2D_V0(load_config())

# Charger le modèle entraîné
model = load_ppo(model_path, env)
print(f"load model {model_path}")
episodes = 10  # Nombre d'épisodes à tester

for episode in range(episodes):
    obs, info = env.reset()
    done = False
    total_reward = 0
    while not done:
        action, _states = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

    print(f"Épisode {episode + 1}: Récompense totale = {total_reward}")

env.close()
