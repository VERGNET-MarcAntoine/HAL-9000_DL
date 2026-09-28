"""Observations, récompense et fin d'épisode de la tâche Hal9000_2D."""
import numpy as np
import pytest

from hal9000.config import load_config
from hal9000.model import Hal9000_2D as task
from hal9000.sim.solar_system import SHIP_SPAWN, SolarSystemSim

REWARD = load_config().reward


def make_task():
    """Une tâche sur un système solaire, avec la planète 3 (la Terre) comme première cible."""
    sim = SolarSystemSim(1, np.random.default_rng(0))
    hal = task.Hal9000Task(1, np.random.default_rng(0), REWARD)
    hal.reset(np.array([0]), sim.planet_pos, sim.ship_pos)
    hal.order[0] = [3, 1, 2, 4, 5]
    return sim, hal


def place_ship(sim, hal, position):
    """Place le vaisseau et remet le guidage à zéro à cette position."""
    sim.ship_pos[0] = position
    hal.previous_distance[:] = hal._target_distance(sim.planet_pos, sim.ship_pos)


def test_reaching_the_target_rewards_and_moves_to_the_next_one():
    sim, hal = make_task()
    target = sim.planet_pos[0, 3]
    place_ship(sim, hal, target + [500, 0])
    sim.ship_pos[0] = target + [100, 0]
    reward, dead, events = hal.transition(sim.planet_pos, sim.ship_pos)
    assert events["reached"][0] and not dead[0]
    assert reward[0] == pytest.approx(REWARD["target_reward"] + 400 / REWARD["shaping_scale"])
    assert hal.index[0] == 1 and hal.target()[0] == 1


@pytest.mark.parametrize("distance, cause", [(100, "dead_sun"), (task.MAX_DISTANCE + 100, "dead_out")])
def test_death(distance, cause):
    sim, hal = make_task()
    place_ship(sim, hal, np.array([distance, 0.0]))
    reward, dead, events = hal.transition(sim.planet_pos, sim.ship_pos)
    assert dead[0] and events[cause][0]
    assert reward[0] < -REWARD["death_penalty"] + 1


def test_circling_the_target_earns_nothing():
    """Le guidage récompense le progrès vers la cible : un tour complet autour d'elle ne rapporte rien."""
    sim, hal = make_task()
    target = sim.planet_pos[0, 3]
    angles = np.linspace(0, 2 * np.pi, 200)
    radii = 1000 + 400 * np.sin(3 * angles)  # en s'approchant et en s'éloignant
    positions = target + np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=-1)
    place_ship(sim, hal, positions[0])
    total = 0.0
    for position in positions[1:]:
        sim.ship_pos[0] = position
        reward, _, _ = hal.transition(sim.planet_pos, sim.ship_pos)
        total += reward[0]
    assert total == pytest.approx(0.0, abs=1e-9)


def test_observation_orbital_quantities():
    """Vaisseau immobile au point d'apparition : ni vitesse radiale ni tangentielle, gravité ~0.15 x poussée."""
    sim, hal = make_task()
    obs = hal.observe(sim.planet_pos, sim.planet_vel, sim.ship_pos, sim.ship_vel)[0]
    assert obs.shape == (task.OBSERVATION_SIZE,) and np.isfinite(obs).all()
    radial, tangential, circular, gravity = obs[12:]
    r = np.linalg.norm(SHIP_SPAWN)
    assert radial == pytest.approx(0) and tangential == pytest.approx(0)
    assert circular == pytest.approx(np.sqrt(task.SUN_GM / r) / task.SPEED_SCALE, rel=1e-5)
    assert gravity == pytest.approx(task.SUN_GM / r ** 2 / 10, rel=1e-5)


def test_thrust_to_engines():
    """Directions de poussée du serveur : left -> +x, right -> -x, up -> -y, down -> +y."""
    engines = task.thrust_to_engines(np.array([1.0, -1.0]))
    assert engines["left"] and engines["up"] and not engines["right"] and not engines["down"]
    assert all(isinstance(value, bool) for value in engines.values())  # sérialisable en JSON
    assert not any(task.thrust_to_engines(np.zeros(2)).values())
