"""Chargement de config.toml, surcharges et synchronisation des vitesses."""
import pytest

from hal9000.config import REAL_TIME_TICK_US, TICKS_PER_SERVER_UPDATE, load_config


def test_overrides():
    config = load_config(overrides=["reward.death_penalty=25", "ppo.net_arch=[256, 256]", "training.sim_envs=8"])
    assert config.reward["death_penalty"] == 25
    assert config.ppo["net_arch"] == [256, 256]
    assert config.sim_envs == 8


def test_unknown_override_is_rejected():
    with pytest.raises(ValueError):
        load_config(overrides=["reward.unknown=1"])


def test_python_and_server_speeds_are_synchronized():
    """Chaque décision couvre decision_interval simulé, quelle que soit l'accélération."""
    for config in (load_config(), load_config().for_training()):
        # Un tick = 1/60 s simulée toutes les simulation_sleep_us microsecondes réelles
        simulated_seconds_per_real_second = REAL_TIME_TICK_US / config.simulation_sleep_us
        assert config.step_time * simulated_seconds_per_real_second == pytest.approx(config.decision_interval, rel=1e-3)
        assert config.server_sleep_us == TICKS_PER_SERVER_UPDATE * config.simulation_sleep_us


def test_decisions_arrive_before_the_server_cuts_the_engines():
    """Le serveur coupe les moteurs après 15 ticks (0.25 s) sans commande."""
    assert load_config().decision_interval * 60 < 15
