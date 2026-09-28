"""
Live broadcast of the state of a training on the Python simulation, for `display_ship2D --training`.

The training regularly sends the position of a few of its ships over UDP on 127.0.0.1: if no display is
listening, the messages are simply lost, without slowing down the training.
"""
import json
import socket
import time

LIVE_ADDRESS = ("127.0.0.1", 47900)
LIVE_INTERVAL = 0.008  # minimum seconds between two messages (~1 message per training step)
LIVE_SHIPS = 4         # number of ships broadcast


class LivePublisher:
    """
    Sends the state of the first ships of a Hal9000SimVecEnv.

    Attributes:
        name (str): The name of the run (displayed, and used to choose a run among several).
    """

    def __init__(self, name: str):
        self.name = name
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setblocking(False)
        self.last_sent = 0.0
        self.steps = 0
        self.episodes: list[dict] = []

    def publish(self, env, finished: list[dict]):
        """
        Records the finished episodes and sends the state of the ships, at most every LIVE_INTERVAL.

        Args:
            env (Hal9000SimVecEnv): The training environment.
            finished (list[dict]): The statistics of the episodes finished at this step.
        """
        self.steps += env.num_envs
        self.episodes = (self.episodes + finished)[-200:]
        now = time.perf_counter()
        if now - self.last_sent < LIVE_INTERVAL:
            return
        self.last_sent = now
        n = min(LIVE_SHIPS, env.num_envs)
        targets = env.task.target()
        message = {
            "name": self.name,
            "steps": self.steps,
            "ships": [{
                "bodies": env.sim.planet_pos[i].round(1).tolist(),
                "ship": env.sim.ship_pos[i].round(1).tolist(),
                "target": int(targets[i]),
                "reached": int(env.task.index[i]),
            } for i in range(n)],
            "recent": {
                "episodes": len(self.episodes),
                "targets": sum(e["targets"] for e in self.episodes) / max(1, len(self.episodes)),
                "survival": sum(not (e["dead_sun"] or e["dead_out"]) for e in self.episodes) / max(1, len(self.episodes)),
            },
        }
        self._send(message)

    def publish_server_ships(self, ships: list[dict]):
        """
        Sends the target and score of ships flown on the Rust server (identified by their uuid), at most
        every LIVE_INTERVAL.

        Args:
            ships (list[dict]): For each ship: uuid, target (index of the planet), reached.
        """
        now = time.perf_counter()
        if now - self.last_sent < LIVE_INTERVAL:
            return
        self.last_sent = now
        self._send({"name": self.name, "server_ships": ships})

    def _send(self, message: dict):
        try:
            self.socket.sendto(json.dumps(message).encode(), LIVE_ADDRESS)
        except OSError:
            pass
