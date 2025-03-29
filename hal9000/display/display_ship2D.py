from hal9000.websocket.websocket_client import SpaceshipWebSocketClient

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import numpy as np
import time

websocket_url = "ws://127.0.0.1:3012"
client = SpaceshipWebSocketClient(websocket_url)
client.connect()


# Configuration du graphique
fig, ax = plt.subplots()
ax.set_xlim(-20000, 20000)  # Ajuste selon la taille de ton univers
ax.set_ylim(-20000, 20000)
ax.set_title("Simulation en temps réel")

# Points pour vaisseaux et planètes
ships_scatter = ax.scatter([], [], c='blue', label="Vaisseaux")
planets_scatter = ax.scatter([], [], c='green', label="Planètes")
sun_scatter = ax.scatter([], [], c='red', label="Soleil")
ax.legend()


def update(frame):
    state = client.get_state()
    planets = state.get("planets", [])
    ships = state.get("ships", [])
    # Récupération des positions
    ship_positions = np.array(
        [ship["body"]["position"][0:2] for ship in ships])
    # Vérifie que les données sont bien en (x, y)
    planet_positions = np.array([planet[1][0:2] for planet in planets[1::]])

    sun_positions = np.array([planets[0][1][0:2]])
    # Mise à jour des données
    if ship_positions.size > 0:
        ships_scatter.set_offsets(ship_positions)
    else:
        ships_scatter.set_offsets(np.empty((0, 2)))

    if planet_positions.size > 0:
        planets_scatter.set_offsets(planet_positions)
    else:
        planets_scatter.set_offsets(np.empty((0, 2)))

    if sun_positions.size > 0:
        sun_scatter.set_offsets(sun_positions)
    else:
        sun_scatter.set_offsets(np.empty((0, 2)))

    return ships_scatter, planets_scatter


# Animation
ani = FuncAnimation(fig, update, interval=100)  # Mise à jour toutes les 100ms
plt.show()
