"""
Affiche le système solaire et les vaisseaux en 2D.

Par défaut, fait voler le dernier modèle Hal9000_2D dans la simulation Python (aucun serveur nécessaire).
Avec --training, montre en direct un entraînement en cours sur la simulation Python.
Avec --server, observe le serveur Rust : tous les vaisseaux qui y sont connectés (entraînement,
évaluation) sont affichés.
"""
import argparse
from collections import deque
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation

from hal9000.config import load_config
from hal9000.model import Hal9000_2D as task

FRAME_INTERVAL = 0.02   # secondes entre deux images
TICKS_PER_SECOND = 60
TRAIL_TICKS = 300       # longueur des traînées (5 s simulées)
LIMIT = task.MAX_DISTANCE * 1.05
PLANET_NAMES = ["Soleil", "Mercure", "Vénus", "Terre", "Mars", "Jupiter"]


def setup_axes(title: str, ax=None):
    """Crée la figure (ou utilise ax) : limites du système, soleil, planètes et leurs noms."""
    fig, ax = plt.subplots(figsize=(8, 8)) if ax is None else (ax.figure, ax)
    ax.set_xlim(-LIMIT, LIMIT)
    ax.set_ylim(-LIMIT, LIMIT)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.add_patch(plt.Circle((0, 0), task.MAX_DISTANCE, fill=False, ls="--", color="grey", lw=0.8))
    sun = ax.scatter([], [], c="orange", s=120, zorder=3)
    planets = ax.scatter([], [], c="green", s=40, zorder=3)
    labels = [ax.text(0, 0, name, fontsize=8, color="dimgrey") for name in PLANET_NAMES]
    return fig, ax, sun, planets, labels


def draw_bodies(body_pos: np.ndarray, sun, planets, labels):
    """Met à jour le soleil, les planètes et leurs noms, body_pos (6, 2)."""
    sun.set_offsets(body_pos[:1])
    planets.set_offsets(body_pos[1:])
    for label, (x, y) in zip(labels, body_pos):
        label.set_position((x + 150, y + 150))


def run_sim(args):
    """Fait voler un modèle dans la simulation Python, tous les vaisseaux dans le même système solaire."""
    from hal9000.model.core.loading import load_ppo
    from hal9000.model.core.training import latest_model
    from hal9000.sim.vec_env import Hal9000SimVecEnv

    torch.set_num_threads(1)
    config = load_config()
    speed = args.speed or config.speedup
    env = Hal9000SimVecEnv(args.ships, config, seed=args.seed, shared_planets=True)
    env.record_ticks = True
    model_path = args.model or latest_model()
    if model_path is None:
        raise SystemExit("Aucun modèle dans models/ : entraînez-en un avec `uv run python -m hal9000.train`.")
    model = load_ppo(model_path, env, config, args.ships)

    fig, ax, sun, planets, labels = setup_axes(f"{model_path} — simulation Python x{speed:g}")
    colors = plt.cm.tab10(np.arange(args.ships) % 10)
    ships = ax.scatter(np.zeros(args.ships), np.zeros(args.ships), c=colors, s=25, zorder=4)
    trails = [ax.plot([], [], color=c, lw=0.8, alpha=0.6)[0] for c in colors]
    target_lines = [ax.plot([], [], color=c, lw=0.6, ls=":")[0] for c in colors]
    stats = ax.text(0.01, 0.99, "", transform=ax.transAxes, va="top", fontsize=8, family="monospace")

    state = {"obs": env.reset(), "queue": deque(), "ticks": 0.0, "episodes": [], "targets": env.task.target()}
    history = [deque(maxlen=TRAIL_TICKS) for _ in range(args.ships)]

    def update(frame):
        # Nombre de ticks de simulation à jouer pendant cette image
        state["ticks"] += speed * TICKS_PER_SECOND * FRAME_INTERVAL
        body_pos = ship_pos = None
        while state["ticks"] >= 1:
            if not state["queue"]:
                action, _ = model.predict(state["obs"], deterministic=True)
                state["obs"], _, dones, infos = env.step(action)
                state["queue"].extend(env.tick_history)
                state["targets"] = env.task.target()
                for i in np.flatnonzero(dones):
                    state["episodes"].append(infos[i]["hal"])
                    history[i].clear()
            body_pos, ship_pos = state["queue"].popleft()
            for trail, position in zip(history, ship_pos):
                trail.append(position)
            state["ticks"] -= 1
        if body_pos is None:
            return ()

        draw_bodies(body_pos, sun, planets, labels)
        ships.set_offsets(ship_pos)
        for i in range(args.ships):
            trail = np.array(history[i]) if history[i] else np.empty((0, 2))
            trails[i].set_data(trail[:, 0], trail[:, 1])
            target = body_pos[state["targets"][i]]
            target_lines[i].set_data([ship_pos[i, 0], target[0]], [ship_pos[i, 1], target[1]])

        lines = [f"vaisseau {i + 1} : {env.task.index[i]} planète(s)" for i in range(args.ships)]
        episodes = state["episodes"]
        if episodes:
            survived = np.mean([not (e["dead_sun"] or e["dead_out"]) for e in episodes])
            lines.append(f"{len(episodes)} épisode(s) terminé(s) : {np.mean([e['targets'] for e in episodes]):.1f} "
                         f"planètes/épisode, survie {survived:.0%}")
        stats.set_text("\n".join(lines))
        return ()

    return fig, update


def run_training(args):
    """Montre en direct les vaisseaux diffusés par un entraînement sur la simulation Python."""
    import json
    import socket

    from hal9000.sim.live import LIVE_ADDRESS, LIVE_SHIPS

    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.bind(LIVE_ADDRESS)
    receiver.setblocking(False)

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    panels = []
    for i, ax in enumerate(axes.flat[:LIVE_SHIPS]):
        _, ax, sun, planets, labels = setup_axes(f"vaisseau {i + 1}", ax)
        ship = ax.scatter([], [], c="blue", s=25, zorder=4)
        trail = ax.plot([], [], color="blue", lw=0.8, alpha=0.6)[0]
        target_line = ax.plot([], [], color="blue", lw=0.6, ls=":")[0]
        panels.append({"ax": ax, "sun": sun, "planets": planets, "labels": labels, "ship": ship,
                       "trail": trail, "target": target_line, "history": deque(maxlen=600)})
    title = fig.suptitle(f"En attente d'un entraînement ({LIVE_ADDRESS[0]}:{LIVE_ADDRESS[1]})…")

    def update(frame):
        # Tous les messages reçus depuis la dernière image allongent les traînées ; le dernier est affiché
        messages = []
        while True:
            try:
                data = json.loads(receiver.recv(65536))
            except BlockingIOError:
                break
            # Sans --name, on suit le premier run reçu (plusieurs entraînements peuvent diffuser en même temps)
            args.name = args.name or data["name"]
            if data["name"] == args.name:
                messages.append(data)
        if not messages:
            return ()
        message = messages[-1]
        for received in messages:
            for panel, ship in zip(panels, received["ships"]):
                position = np.array(ship["ship"])
                history = panel["history"]
                # Nouvel épisode : le vaisseau réapparaît ailleurs, on efface sa traînée
                if history and np.linalg.norm(position - history[-1]) > 1500:
                    history.clear()
                history.append(position)

        recent = message["recent"]
        title.set_text(f"Entraînement {message['name']} — {message['steps']:,} steps — {recent['episodes']} derniers "
                       f"épisodes : {recent['targets']:.1f} planètes/épisode, survie {recent['survival']:.0%}")
        for panel, ship in zip(panels, message["ships"]):
            bodies, position = np.array(ship["bodies"]), np.array(ship["ship"])
            history = panel["history"]
            draw_bodies(bodies, panel["sun"], panel["planets"], panel["labels"])
            panel["ship"].set_offsets(position[None])
            trail = np.array(history)
            panel["trail"].set_data(trail[:, 0], trail[:, 1])
            target = bodies[ship["target"]]
            panel["target"].set_data([position[0], target[0]], [position[1], target[1]])
            panel["ax"].set_title(f"{ship['reached']} planète(s) atteinte(s) dans l'épisode", fontsize=9)
        return ()

    return fig, update


def run_server(args):
    """
    Observe le serveur Rust : affiche tous les vaisseaux qui y sont connectés. Pour les vaisseaux pilotés
    par `evaluate --server --watch`, la cible, la traînée et le nombre de planètes atteintes sont aussi
    affichés (le serveur ne connaît pas les cibles : evaluate les diffuse en local).
    """
    import json
    import socket

    from hal9000.sim.live import LIVE_ADDRESS
    from hal9000.websocket.websocket_client import SpaceshipWebSocketClient

    config = load_config()
    if args.url:
        config = replace(config, websocket_url=args.url)
    client = SpaceshipWebSocketClient(config.websocket_url)
    client.connect()
    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.bind(LIVE_ADDRESS)
    receiver.setblocking(False)

    fig, ax, sun, planets, labels = setup_axes(f"Serveur Rust {config.websocket_url}")
    others = ax.scatter([], [], c="grey", s=20, zorder=4)
    palette = plt.cm.tab10(np.arange(10))
    # Vaisseaux pilotés, par uuid : couleur, traînée, trait vers la cible
    piloted = {}
    stats = ax.text(0.01, 0.99, "", transform=ax.transAxes, va="top", fontsize=8, family="monospace")
    known = {"ships": {}}

    def update(frame):
        while True:
            try:
                data = json.loads(receiver.recv(65536))
            except BlockingIOError:
                break
            if "server_ships" in data:
                known["ships"] = {ship["uuid"]: ship for ship in data["server_ships"]}

        state = client.get_state()
        bodies = np.array([p[1][:2] for p in state["planets"]])
        draw_bodies(bodies, sun, planets, labels)
        positions = {str(s["uuid"]): np.array(s["body"]["position"][:2]) for s in state.get("ships", [])}
        positions.pop(client.ship_uuid, None)  # le vaisseau (inactif) de cet affichage

        # Vaisseaux disparus (épisode terminé) : on retire leurs tracés
        for uuid in [u for u in piloted if u not in positions]:
            for artist in piloted.pop(uuid)["artists"]:
                artist.remove()
        lines = []
        for uuid, position in positions.items():
            info = known["ships"].get(uuid)
            if info is None:
                continue
            if uuid not in piloted:
                color = palette[len(piloted) % 10]
                artists = [ax.plot([], [], color=color, lw=0.8, alpha=0.6)[0], ax.plot([], [], color=color, lw=0.6, ls=":")[0],
                           ax.scatter([], [], color=color, s=25, zorder=5)]
                piloted[uuid] = {"artists": artists, "history": deque(maxlen=TRAIL_TICKS)}
            trail, target_line, marker = piloted[uuid]["artists"]
            history = piloted[uuid]["history"]
            history.append(position)
            points = np.array(history)
            trail.set_data(points[:, 0], points[:, 1])
            target = bodies[info["target"]]
            target_line.set_data([position[0], target[0]], [position[1], target[1]])
            marker.set_offsets(position[None])
            lines.append(f"vaisseau {len(lines) + 1} : {info['reached']} planète(s)")

        # Les autres vaisseaux (entraînement, autres clients) en gris
        rest = [p for u, p in positions.items() if u not in piloted]
        others.set_offsets(np.array(rest) if rest else np.empty((0, 2)))
        stats.set_text("\n".join(lines))
        return ()

    return fig, update


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", action="store_true", help="montrer en direct un entraînement sur la simulation Python")
    parser.add_argument("--name", help="avec --training : nom du run à suivre (par défaut : le premier reçu)")
    parser.add_argument("--server", action="store_true", help="observer le serveur Rust au lieu de la simulation")
    parser.add_argument("--url", help="adresse du serveur (par défaut : celle de config.toml)")
    parser.add_argument("--model", help="modèle à faire voler (par défaut : le dernier sauvegardé)")
    parser.add_argument("--ships", type=int, default=4, help="nombre de vaisseaux dans la simulation")
    parser.add_argument("--speed", type=float, help="accélération de la simulation (par défaut : [simulation] speedup)")
    parser.add_argument("--seed", type=int, help="graine aléatoire de la simulation")
    args = parser.parse_args()

    if args.training:
        fig, update = run_training(args)
    elif args.server:
        fig, update = run_server(args)
    else:
        fig, update = run_sim(args)
    # L'animation doit rester référencée tant que la fenêtre est ouverte
    animation = FuncAnimation(fig, update, interval=FRAME_INTERVAL * 1000, cache_frame_data=False)
    plt.show()


if __name__ == "__main__":
    main()
