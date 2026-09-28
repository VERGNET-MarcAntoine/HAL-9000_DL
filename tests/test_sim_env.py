"""Environnement de simulation (VecEnv SB3) et performances du modèle entraîné."""
from pathlib import Path

import numpy as np
import pytest
from stable_baselines3.common.vec_env import VecMonitor

from hal9000.config import load_config
from hal9000.sim.vec_env import Hal9000SimVecEnv

FINAL_MODEL = Path(__file__).parent.parent / "models" / "Hal9000_2D_final.zip"


def test_episodes_are_truncated_at_the_time_limit():
    env = Hal9000SimVecEnv(4, load_config(), seed=0)
    env.reset()
    env.max_steps = 5
    env.sim.ship_vel[:] = env.sim.planet_vel[:, 3]  # pas de chute dans le soleil en 5 steps
    for step in range(5):
        obs, reward, dones, infos = env.step(np.full(4, 4))
    assert dones.all()
    assert all(info["TimeLimit.truncated"] and "hal" in info and "terminal_observation" in info for info in infos)
    assert (env.steps == 0).all()


def test_shared_planets_stay_synchronized():
    """Avec shared_planets, tous les vaisseaux restent dans le même système solaire, même après un reset."""
    env = Hal9000SimVecEnv(3, load_config(), seed=0, shared_planets=True)
    env.reset()
    for _ in range(300):  # sans poussée, les vaisseaux finissent par tomber et sont réinitialisés
        env.step(np.full(3, 4))
    assert np.allclose(env.sim.planet_pos, env.sim.planet_pos[0])


@pytest.mark.skipif(not FINAL_MODEL.exists(), reason="modèle final absent")
def test_final_model_performance():
    """Garde-fou : le modèle final atteint plusieurs planètes par épisode et survit à la majorité des épisodes."""
    from hal9000.evaluate import evaluate
    from hal9000.model.core.loading import load_ppo

    config = load_config()
    env = VecMonitor(Hal9000SimVecEnv(16, config, seed=0))
    model = load_ppo(str(FINAL_MODEL), env, config, env.num_envs)
    results = evaluate(model, env, 16)
    assert np.mean([r["targets"] for r in results]) >= 6
    assert np.mean([not (r["dead_sun"] or r["dead_out"]) for r in results]) >= 0.5
