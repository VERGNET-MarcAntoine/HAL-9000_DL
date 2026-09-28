"""
Diffusion en direct de l'état d'un entraînement sur la simulation Python, pour `display_ship2D --training`.

L'entraînement envoie régulièrement la position de quelques-uns de ses vaisseaux en UDP sur 127.0.0.1 :
si aucun affichage n'écoute, les messages sont simplement perdus, sans ralentir l'entraînement.
"""
import json
import socket
import time

LIVE_ADDRESS = ("127.0.0.1", 47900)
LIVE_INTERVAL = 0.008  # secondes minimum entre deux messages (~1 message par step d'entraînement)
LIVE_SHIPS = 4         # nombre de vaisseaux diffusés


class LivePublisher:
    """
    Envoie l'état des premiers vaisseaux d'un Hal9000SimVecEnv.

    Attributes:
        name (str): Le nom du run (affiché, et permet de choisir un run parmi plusieurs).
    """

    def __init__(self, name: str):
        self.name = name
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setblocking(False)
        self.last_sent = 0.0
        self.steps = 0
        self.episodes = []

    def publish(self, env, finished: list[dict]):
        """
        Enregistre les épisodes terminés et envoie l'état des vaisseaux, au plus toutes les LIVE_INTERVAL.

        Args:
            env (Hal9000SimVecEnv): L'environnement d'entraînement.
            finished (list[dict]): Les statistiques des épisodes terminés à ce step.
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
        Envoie la cible et le score de vaisseaux pilotés sur le serveur Rust (identifiés par leur uuid),
        au plus toutes les LIVE_INTERVAL.

        Args:
            ships (list[dict]): Pour chaque vaisseau : uuid, target (indice de la planète), reached.
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
