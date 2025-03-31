import os
import torch
from hal9000.model.Hal9000_2D_MAGB_V0 import Hal9000_2D_V0
from hal9000.ship import Ship
from stable_baselines3 import PPO
from dotenv import load_dotenv


load_dotenv()

episode_time = int(os.getenv("EPISODE_TIME"))
step_time = float(os.getenv("SLEEP_TIME"))
model = "Hal9000_2D_MAGB_V0_2025_03_30_07_34_33_step13536000"


# Charger le modèle
models_dir = "models"
model_path = os.path.join(models_dir, model)

# Initialiser l'environnement
# Assurez-vous que l'environnement peut afficher les résultats
env = Hal9000_2D_V0(episode_time, step_time)

# Charger le modèle entraîné
model = PPO.load(model_path, env=env)
print(f"load model {model_path}")

episodes = 10  # Nombre d'épisodes à tester

for episode in range(episodes):
    obs, info = env.reset()
    done = False
    total_reward = 0
    count = 0
    while not done:
        action, _states = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)
        # count += 1
        # if count % 400 == 0:
        #     print(obs["target"])

        done = terminated or truncated
        total_reward += reward
    print(f"Épisode {episode + 1}: Récompense totale = {total_reward}")

env.close()
