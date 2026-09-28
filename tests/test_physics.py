"""La simulation Python doit reproduire exactement la physique du serveur Rust."""
import json
from pathlib import Path

import numpy as np
import pytest

from hal9000.sim.solar_system import G, MASSES, SolarSystemSim

CASES = json.loads((Path(__file__).parent / "data" / "server_trajectories.json").read_text())


@pytest.mark.parametrize("case", CASES, ids=[case["engine"] for case in CASES])
def test_matches_rust_server(case):
    """Trajectoires enregistrées sur le serveur (tests/record_server_trajectories.py), ~10 s simulées."""
    sim = SolarSystemSim(1, np.random.default_rng(0))
    start = {k: np.array(v) for k, v in case["start"].items()}
    sim.load(0, start["planet_pos"], start["planet_vel"], start["ship_pos"], start["ship_vel"])
    for _ in range(case["ticks"]):
        sim.tick(np.array([case["thrust"]], dtype=float))

    end = {k: np.array(v) for k, v in case["end"].items()}
    np.testing.assert_allclose(sim.planet_pos[0], end["planet_pos"], atol=1e-6)
    np.testing.assert_allclose(sim.ship_pos[0], end["ship_pos"], atol=1e-6)
    np.testing.assert_allclose(sim.ship_vel[0], end["ship_vel"], atol=1e-6)


def test_planets_stay_on_circular_orbits():
    """Les planètes démarrent sur des orbites circulaires et y restent (pas de dérive numérique)."""
    sim = SolarSystemSim(3, np.random.default_rng(1))
    radius = np.linalg.norm(sim.planet_pos[:, 1:], axis=-1)
    for _ in range(60 * 60):  # une minute simulée
        sim.tick(np.zeros((3, 2)))
    np.testing.assert_allclose(np.linalg.norm(sim.planet_pos[:, 1:], axis=-1), radius, rtol=1e-3)


def test_ship_falls_toward_the_sun_without_thrust():
    """Immobile à son point d'apparition, le vaisseau tombe vers le soleil avec l'accélération GM / r²."""
    sim = SolarSystemSim(1, np.random.default_rng(2))
    r = np.linalg.norm(sim.ship_pos[0])
    sim.tick(np.zeros((1, 2)))
    radial_speed = sim.ship_vel[0] @ (sim.ship_pos[0] / np.linalg.norm(sim.ship_pos[0]))
    assert radial_speed < 0
    assert abs(radial_speed) == pytest.approx(G * MASSES[0] / r ** 2 / 60, rel=0.05)
