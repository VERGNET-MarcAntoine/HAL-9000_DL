from websocket_client import SpaceshipWebSocketClient

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import numpy as np

websocket_url = "ws://127.0.0.1:3012"
client = SpaceshipWebSocketClient(websocket_url)
client.connect()

# Création des deux sous-graphes pour X-Y et Y-Z
fig, (ax_xy, ax_yz) = plt.subplots(1, 2, figsize=(12, 6))

# === Graphique X-Y (déjà existant) ===
ax_xy.set_xlim(-6000, 6000)
ax_xy.set_ylim(-6000, 6000)
ax_xy.set_title("Vue X-Y")

ships_scatter_xy = ax_xy.scatter([], [], c='blue', label="Vaisseaux")
planets_scatter_xy = ax_xy.scatter([], [], c='green', label="Planètes")
sun_scatter_xy = ax_xy.scatter([], [], c='red', label="Soleil")
ax_xy.legend()

# === Graphique Y-Z (nouveau) ===
ax_yz.set_xlim(-6000, 6000)
ax_yz.set_ylim(-6000, 6000)
ax_yz.set_title("Vue Y-Z")

ships_scatter_yz = ax_yz.scatter([], [], c='blue', label="Vaisseaux")
planets_scatter_yz = ax_yz.scatter([], [], c='green', label="Planètes")
sun_scatter_yz = ax_yz.scatter([], [], c='red', label="Soleil")
ax_yz.legend()


def update(frame):
    state = client.get_state()
    planets = state.get("planets", [])
    ships = state.get("ships", [])

    # Récupération des positions (X, Y, Z)
    ship_positions = np.array([ship["body"]["position"][0:3] for ship in ships])
    planet_positions = np.array([planet[1][0:3] for planet in planets[1:-1]])
    sun_positions = np.array([planets[0][1][0:3]])

    # === Mise à jour Graphique X-Y ===
    if ship_positions.size > 0:
        ships_scatter_xy.set_offsets(ship_positions[:, :2])  # (X, Y)
    else:
        ships_scatter_xy.set_offsets(np.empty((0, 2)))

    if planet_positions.size > 0:
        planets_scatter_xy.set_offsets(planet_positions[:, :2])
    else:
        planets_scatter_xy.set_offsets(np.empty((0, 2)))

    if sun_positions.size > 0:
        sun_scatter_xy.set_offsets(sun_positions[:, :2])
    else:
        sun_scatter_xy.set_offsets(np.empty((0, 2)))

    # === Mise à jour Graphique Y-Z ===
    if ship_positions.size > 0:
        ships_scatter_yz.set_offsets(ship_positions[:, 1:])  # (Y, Z)
    else:
        ships_scatter_yz.set_offsets(np.empty((0, 2)))

    if planet_positions.size > 0:
        planets_scatter_yz.set_offsets(planet_positions[:, 1:])
    else:
        planets_scatter_yz.set_offsets(np.empty((0, 2)))

    if sun_positions.size > 0:
        sun_scatter_yz.set_offsets(sun_positions[:, 1:])
    else:
        sun_scatter_yz.set_offsets(np.empty((0, 2)))

    return ships_scatter_xy, planets_scatter_xy, ships_scatter_yz, planets_scatter_yz


# Animation
ani = FuncAnimation(fig, update, interval=100)  # Mise à jour toutes les 100ms
plt.tight_layout()
plt.show()
