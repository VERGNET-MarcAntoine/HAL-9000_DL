import os
from ship import Ship
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import A2C, PPO
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

number_episode = int(os.getenv("NUMBER_EPISODE"))
episode_time = int(os.getenv("EPISODE_TIME"))
save_number = int(os.getenv("SAVE_NUMBER"))
model = os.getenv("MODEL")  # Chemin du modèle à charger

TIMESTEPS = (number_episode * episode_time * 60 * 4) // save_number

logdir = "logs"
models_dir = "models"

model_path = os.path.join(models_dir, model)

# Utilisation de datetime pour formater la date et l'heure
start_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

env = Ship()

model_name = "PPO_BoneShip"  # Nom du modèle
log_name = f"{model_name}_{start_time}"  # Format des logs

# Créer le dossier des logs et des modèles s'ils n'existent pas
os.makedirs(logdir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

if model_path and os.path.exists(model_path + ".zip"):
    print(f"Reprise de l'entraînement du modèle depuis {model_path}")
    model = PPO.load(model_path, env=env, tensorboard_log=logdir, device="cpu")
else:
    print("Aucun modèle existant trouvé, création d'un nouveau modèle.")
    model = PPO("MultiInputPolicy", env, verbose=1,
                tensorboard_log=logdir, device="cpu")

for i in range(save_number):
    model.learn(total_timesteps=TIMESTEPS,
                reset_num_timesteps=False, tb_log_name=log_name)

    new_model_path = f"{models_dir}/{log_name}_step{TIMESTEPS * (i + 1)}"
    model.save(new_model_path)

env.close()
