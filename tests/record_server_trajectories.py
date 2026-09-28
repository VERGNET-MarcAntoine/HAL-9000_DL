"""
Records reference trajectories on the Rust server, for tests/test_physics.py.

For each thrust (4 engines and none), a state of the server is read, the thrust is held for ~2 s, then a
second state is read. The number of elapsed ticks is found by fitting the Python simulation to the motion
of the planets (which does not depend on the ships).

To run again if the physics of the server changes, with the server started:
    uv run python -m hal9000.server
    uv run python -m tests.record_server_trajectories
"""
import json
import threading
import time
from pathlib import Path

import numpy as np

from hal9000.model.core.ship2D import NO_ROTATION, state_to_arrays
from hal9000.sim.solar_system import SolarSystemSim
from hal9000.websocket.websocket_client import SpaceshipWebSocketClient

OUTPUT = Path(__file__).parent / "data" / "server_trajectories.json"
ENGINES = {"left": (1, 0), "right": (-1, 0), "up": (0, -1), "down": (0, 1), "none": (0, 0)}


def record(engine: str, duration: float = 2.0) -> dict:
    client = SpaceshipWebSocketClient()
    client.connect()
    stop = threading.Event()

    def keep_thrusting():
        while not stop.is_set():
            client.send_command({k: k == engine for k in ["left", "right", "up", "down", "front", "back"]}, NO_ROTATION)
            time.sleep(0.02)

    thread = threading.Thread(target=keep_thrusting)
    thread.start()
    time.sleep(0.5)
    start, end = client.get_state(), None
    time.sleep(duration)
    end = client.get_state()
    stop.set()
    thread.join()
    client.disconnect()

    start_arrays = [a[0] for a in state_to_arrays(start)]
    end_arrays = [a[0] for a in state_to_arrays(end)]
    sim = SolarSystemSim(1, np.random.default_rng(0))
    sim.load(0, *start_arrays)
    thrust = np.array([ENGINES[engine]], dtype=float)
    best_ticks, best_error = 0, np.inf
    for ticks in range(1, 60 * 60):
        sim.tick(thrust)
        error = np.abs(sim.planet_pos[0] - end_arrays[0]).max()
        if error < best_error:
            best_ticks, best_error = ticks, error
    return {
        "engine": engine,
        "thrust": ENGINES[engine],
        "ticks": best_ticks,
        "start": {k: v.tolist() for k, v in zip(["planet_pos", "planet_vel", "ship_pos", "ship_vel"], start_arrays, strict=True)},
        "end": {k: v.tolist() for k, v in zip(["planet_pos", "planet_vel", "ship_pos", "ship_vel"], end_arrays, strict=True)},
    }


if __name__ == "__main__":
    cases = [record(engine) for engine in ENGINES]
    OUTPUT.write_text(json.dumps(cases, indent=1))
    for case in cases:
        print(f"{case['engine']:5s} : {case['ticks']} ticks")
    print(f"Recorded: {OUTPUT}")
