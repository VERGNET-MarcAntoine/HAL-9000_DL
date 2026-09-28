"""
Displays the solar system and the ships in 2D.

By default, flies the latest Hal9000_2D model in the Python simulation (no server needed).
With --training, shows a training in progress on the Python simulation, live.
With --server, observes the Rust server: every ship connected to it (training, evaluation) is shown.
"""
import argparse
from collections import deque
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation
from matplotlib.figure import Figure
from matplotlib.patches import Circle

from hal9000.config import load_config
from hal9000.sim.solar_system import INITIAL_POSITIONS

FRAME_INTERVAL = 0.02      # seconds between two frames (on screen)
GIF_FRAME_INTERVAL = 0.04  # seconds between two frames of a GIF (lighter file)
GIF_DPI = 75               # 600 x 600 pixels
TICKS_PER_SECOND = 60
TRAIL_TICKS = 1800         # length of the trails (30 simulated seconds)
# View fitted to the orbits (Jupiter at ~7100); the death limit (10,000) is out of view
ORBIT_RADII = np.linalg.norm(INITIAL_POSITIONS[1:], axis=1)
LIMIT = ORBIT_RADII.max() * 1.12
PLANET_NAMES = ["Sun", "Mercury", "Venus", "Earth", "Mars", "Jupiter"]
PORT_IN_USE = "Another display (--training or --server) is already open and receives the live information"

# Themes: one color per body (sun, then planets) and distinct ship colors, contrasted on the theme background
THEMES = {
    "dark": {
        "background": "#0b0f1a", "text": "#c8d0e0", "muted": "#4a5368", "orbit_alpha": 0.25,
        "bodies": ["#ffd23f", "#a8a8a8", "#e8c07d", "#4f9dff", "#e0573a", "#d9a066"],
        "ships": ["#00e5ff", "#ff4fd8", "#7dff4f", "#ffffff", "#ff9f1c", "#b388ff"],
    },
    "light": {
        "background": "#ffffff", "text": "#2b2f38", "muted": "#8a90a0", "orbit_alpha": 0.35,
        "bodies": ["#f4a300", "#8d8d8d", "#c9a227", "#2f6fdb", "#c1440e", "#b07d48"],
        "ships": ["#0077b6", "#d6286f", "#2a9d3f", "#7b2cbf", "#f77f00", "#111111"],
    },
}
STYLE = dict(THEMES["dark"])


def ship_color(i: int) -> str:
    return STYLE["ships"][i % len(STYLE["ships"])]


def setup_axes(title: str, ax=None):
    """Creates the figure (or uses ax): limits of the system, sun, planets and their names."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
        fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.95)
    fig = ax.figure
    assert isinstance(fig, Figure)
    fig.patch.set_facecolor(STYLE["background"])
    ax.set_facecolor(STYLE["background"])
    ax.set_xlim(-LIMIT, LIMIT)
    ax.set_ylim(-LIMIT, LIMIT)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, color=STYLE["text"], fontsize=10)
    bodies = STYLE["bodies"]
    for radius, color in zip(ORBIT_RADII, bodies[1:], strict=True):
        ax.add_patch(Circle((0, 0), radius, fill=False, color=color, lw=0.5, alpha=STYLE["orbit_alpha"]))
    sun = ax.scatter([0], [0], c=bodies[:1], s=160, zorder=3)
    planets = ax.scatter(np.zeros(len(bodies) - 1), np.zeros(len(bodies) - 1), c=bodies[1:], s=45, zorder=3)
    labels = [ax.text(0, 0, name, fontsize=8, color=STYLE["muted"]) for name in PLANET_NAMES]
    return fig, ax, sun, planets, labels


def draw_bodies(body_pos: np.ndarray, sun, planets, labels):
    """Updates the sun, the planets and their names, body_pos (6, 2)."""
    sun.set_offsets(body_pos[:1])
    planets.set_offsets(body_pos[1:])
    for label, (x, y) in zip(labels, body_pos, strict=True):
        label.set_position((x + 150, y + 150))


def run_sim(args):
    """Flies a model in the Python simulation, every ship in the same solar system."""
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
        raise SystemExit("No model in models/: train one with `uv run python -m hal9000.train`.")
    model = load_ppo(model_path, env, config, args.ships)

    fig, ax, sun, planets, labels = setup_axes(f"HAL-9000 — {Path(model_path).stem} — simulation x{speed:g}")
    colors = [ship_color(i) for i in range(args.ships)]
    ships = ax.scatter(np.zeros(args.ships), np.zeros(args.ships), c=colors, s=25, zorder=4)
    trails = [ax.plot([], [], color=c, lw=0.8, alpha=0.6)[0] for c in colors]
    target_lines = [ax.plot([], [], color=c, lw=0.6, ls=":")[0] for c in colors]
    stats = ax.text(0.01, 0.99, "", transform=ax.transAxes, va="top", fontsize=8, family="monospace",
                    color=STYLE["text"])

    obs = env.reset()
    state = {"queue": deque(), "ticks": 0.0, "episodes": [], "targets": env.task.target()}
    history = [deque(maxlen=TRAIL_TICKS) for _ in range(args.ships)]

    def update(frame):
        nonlocal obs
        # Number of simulation ticks to play during this frame
        state["ticks"] += speed * TICKS_PER_SECOND * args.frame_interval
        body_pos = ship_pos = None
        while state["ticks"] >= 1:
            if not state["queue"]:
                action, _ = model.predict(obs, deterministic=True)
                step_obs, _, dones, infos = env.step(action)
                obs = np.asarray(step_obs)
                state["queue"].extend(env.tick_history)
                state["targets"] = env.task.target()
                for i in np.flatnonzero(dones):
                    state["episodes"].append(infos[i]["hal"])
                    history[i].clear()
            body_pos, ship_pos = state["queue"].popleft()
            for trail, position in zip(history, ship_pos, strict=True):
                trail.append(position)
            state["ticks"] -= 1
        if body_pos is None or ship_pos is None:
            return ()

        draw_bodies(body_pos, sun, planets, labels)
        ships.set_offsets(ship_pos)
        for i in range(args.ships):
            trail = np.array(history[i]) if history[i] else np.empty((0, 2))
            trails[i].set_data(trail[:, 0], trail[:, 1])
            target = body_pos[state["targets"][i]]
            target_lines[i].set_data([ship_pos[i, 0], target[0]], [ship_pos[i, 1], target[1]])

        lines = [f"ship {i + 1}: {env.task.index[i]} planet(s)" for i in range(args.ships)]
        episodes = state["episodes"]
        if episodes:
            survived = np.mean([not (e["dead_sun"] or e["dead_out"]) for e in episodes])
            lines.append(f"{len(episodes)} episode(s) finished: {np.mean([e['targets'] for e in episodes]):.1f} "
                         f"planets/episode, survival {survived:.0%}")
        stats.set_text("\n".join(lines))
        return ()

    return fig, update


def run_training(args):
    """Shows the ships broadcast by a training on the Python simulation, live."""
    import json
    import socket

    from hal9000.sim.live import LIVE_ADDRESS, LIVE_SHIPS

    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        receiver.bind(LIVE_ADDRESS)
    except OSError:
        raise SystemExit(f"{PORT_IN_USE}: close it first.") from None
    receiver.setblocking(False)

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    panels = []
    for i, ax in enumerate(axes.flat[:LIVE_SHIPS]):
        _, ax, sun, planets, labels = setup_axes(f"ship {i + 1}", ax)
        ship = ax.scatter([], [], c=ship_color(i), s=25, zorder=4)
        trail = ax.plot([], [], color=ship_color(i), lw=0.8, alpha=0.6)[0]
        target_line = ax.plot([], [], color=ship_color(i), lw=0.6, ls=":")[0]
        panels.append({"ax": ax, "sun": sun, "planets": planets, "labels": labels, "ship": ship,
                       "trail": trail, "target": target_line, "history": deque(maxlen=600)})
    title = fig.suptitle(f"Waiting for a training ({LIVE_ADDRESS[0]}:{LIVE_ADDRESS[1]})…", color=STYLE["text"])

    def update(frame):
        # Every message received since the last frame extends the trails; the last one is displayed
        messages = []
        while True:
            try:
                data = json.loads(receiver.recv(65536))
            except BlockingIOError:
                break
            # Without --name, follow the first run received (several trainings can broadcast at the same time)
            args.name = args.name or data["name"]
            if data["name"] == args.name:
                messages.append(data)
        if not messages:
            return ()
        message = messages[-1]
        for received in messages:
            # A training with fewer ships than panels broadcasts fewer ships
            for panel, ship in zip(panels, received["ships"], strict=False):
                position = np.array(ship["ship"])
                history = panel["history"]
                # New episode: the ship reappears elsewhere, its trail is cleared
                if history and np.linalg.norm(position - history[-1]) > 1500:
                    history.clear()
                history.append(position)

        recent = message["recent"]
        title.set_text(f"Training {message['name']} — {message['steps']:,} steps — last {recent['episodes']} "
                       f"episodes: {recent['targets']:.1f} planets/episode, survival {recent['survival']:.0%}")
        for panel, ship in zip(panels, message["ships"], strict=False):
            bodies, position = np.array(ship["bodies"]), np.array(ship["ship"])
            history = panel["history"]
            draw_bodies(bodies, panel["sun"], panel["planets"], panel["labels"])
            panel["ship"].set_offsets(position[None])
            trail = np.array(history)
            panel["trail"].set_data(trail[:, 0], trail[:, 1])
            target = bodies[ship["target"]]
            panel["target"].set_data([position[0], target[0]], [position[1], target[1]])
            panel["ax"].set_title(f"{ship['reached']} planet(s) reached in the episode", fontsize=9, color=STYLE["text"])
        return ()

    return fig, update


def run_server(args):
    """
    Observes the Rust server: shows every ship connected to it. For the ships flown by
    `evaluate --server --watch`, the target, the trail and the number of planets reached are also shown
    (the server does not know the targets: evaluate broadcasts them locally).
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
    try:
        receiver.bind(LIVE_ADDRESS)
        receiver.setblocking(False)
    except OSError:
        print(f"Warning: {PORT_IN_USE}. The targets of the ships will not be shown here.")
        receiver = None

    fig, ax, sun, planets, labels = setup_axes(f"Rust server {config.websocket_url}")
    others = ax.scatter([], [], c=STYLE["muted"], s=20, zorder=4)
    # Piloted ships, by uuid: color, trail, line to the target
    piloted = {}
    stats = ax.text(0.01, 0.99, "", transform=ax.transAxes, va="top", fontsize=8, family="monospace",
                    color=STYLE["text"])
    known = {"ships": {}}

    def update(frame):
        while receiver:
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
        if client.ship_uuid:
            positions.pop(client.ship_uuid, None)  # the (idle) ship of this display

        # Ships that disappeared (episode finished): remove their drawings
        for uuid in [u for u in piloted if u not in positions]:
            for artist in piloted.pop(uuid)["artists"]:
                artist.remove()
        lines = []
        for uuid, position in positions.items():
            info = known["ships"].get(uuid)
            if info is None:
                continue
            if uuid not in piloted:
                color = ship_color(len(piloted))
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
            lines.append(f"ship {len(lines) + 1}: {info['reached']} planet(s)")

        # The other ships (training, other clients) in grey
        rest = [p for u, p in positions.items() if u not in piloted]
        others.set_offsets(np.array(rest) if rest else np.empty((0, 2)))
        stats.set_text("\n".join(lines))
        return ()

    return fig, update


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", action="store_true", help="show a training on the Python simulation, live")
    parser.add_argument("--name", help="with --training: name of the run to follow (default: the first one received)")
    parser.add_argument("--server", action="store_true", help="observe the Rust server instead of the simulation")
    parser.add_argument("--url", help="address of the server (default: the one of config.toml)")
    parser.add_argument("--model", help="model to fly (default: the latest saved one)")
    parser.add_argument("--ships", type=int, default=4, help="number of ships in the simulation")
    parser.add_argument("--speed", type=float, help="speed of the simulation (default: [simulation] speedup)")
    parser.add_argument("--seed", type=int, help="random seed of the simulation")
    parser.add_argument("--theme", choices=THEMES, default="dark", help="colors: dark or light background")
    parser.add_argument("--save", metavar="FILE.gif", help="record the animation in a GIF instead of showing it")
    parser.add_argument("--frames", type=int, default=300, help="with --save: number of frames")
    args = parser.parse_args()
    args.frame_interval = GIF_FRAME_INTERVAL if args.save else FRAME_INTERVAL
    STYLE.update(THEMES[args.theme])
    if args.save:
        plt.switch_backend("Agg")

    if args.training:
        fig, update = run_training(args)
    elif args.server:
        fig, update = run_server(args)
    else:
        fig, update = run_sim(args)
    if args.save:
        from matplotlib.animation import PillowWriter
        animation = FuncAnimation(fig, update, frames=args.frames, cache_frame_data=False)
        animation.save(args.save, writer=PillowWriter(fps=round(1 / GIF_FRAME_INTERVAL)), dpi=GIF_DPI,
                       savefig_kwargs={"facecolor": STYLE["background"]})
        print(f"Animation saved: {args.save}")
        return
    # The animation must stay referenced while the window is open
    animation = FuncAnimation(fig, update, interval=FRAME_INTERVAL * 1000, cache_frame_data=False)  # noqa: F841
    plt.show()


if __name__ == "__main__":
    main()
