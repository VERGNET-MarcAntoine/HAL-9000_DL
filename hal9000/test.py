import os
import torch
from hal9000.model.Hal9000_2D_V0 import Hal9000_2D_V0
from stable_baselines3 import PPO
from dotenv import load_dotenv

load_dotenv()


number_episode = int(os.getenv("NUMBER_EPISODE"))
episode_time = int(os.getenv("EPISODE_TIME"))
save_number = int(os.getenv("SAVE_NUMBER"))
model = str(os.getenv("MODEL"))

TIMESTEPS = (number_episode*episode_time*60*4)//save_number


# Charger le modèle
models_dir = "models"
model_path = os.path.join(models_dir, model)

# Initialiser l'environnement
# Assurez-vous que l'environnement peut afficher les résultats
env = Hal9000_2D_V0()

# Charger le modèle entraîné
model = PPO.load(model_path, env=env)

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
