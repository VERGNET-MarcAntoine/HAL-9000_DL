"""
Réplique vectorisée (numpy) de la physique du serveur Rust (branche deep_learning), pour entraîner
beaucoup plus vite que via le serveur. Chaque tick reproduit SolarSystem::update, dans le même ordre :
  1. gravité du soleil sur les planètes (vitesses)
  2. gravité de tous les corps sur les vaisseaux (vitesses, positions des planètes pas encore mises à jour)
  3. déplacement des planètes
  4. poussée des moteurs puis déplacement des vaisseaux
"""
import numpy as np

G = 6.67430e-11
TICK = 1 / 60

# rust-server/src/body/solar_system.rs : (masse, position initiale), le soleil en premier
BODIES = [
    (1.989e17, (0.0, 0.0)),        # Sun
    (3.285e13, (8.0e2, 5.0e2)),    # Mercury
    (4.867e14, (-1.25e3, 1.0e3)),  # Venus
    (5.972e14, (1.8e3, -1.8e3)),   # Earth
    (6.39e14, (2.0e3, 3.0e3)),     # Mars
    (1.898e15, (-5.0e3, -5.0e3)),  # Jupiter
]
N_BODIES = len(BODIES)
MASSES = np.array([m for m, _ in BODIES])
INITIAL_POSITIONS = np.array([p for _, p in BODIES])

# rust-server/src/body/ship.rs
SHIP_SPAWN = np.array([3000.0, 0.0])
ENGINE_ACCELERATION = 10000.0 / 1000.0  # puissance / masse du vaisseau


class SolarSystemSim:
    """
    n systèmes solaires indépendants, chacun avec un vaisseau.

    Attributes:
        planet_pos (np.ndarray): Positions des corps, (n, N_BODIES, 2).
        planet_vel (np.ndarray): Vitesses des corps, (n, N_BODIES, 2).
        ship_pos (np.ndarray): Positions des vaisseaux, (n, 2).
        ship_vel (np.ndarray): Vitesses des vaisseaux, (n, 2).
    """

    def __init__(self, n: int, rng: np.random.Generator):
        self.n = n
        self.rng = rng
        self.planet_pos = np.zeros((n, N_BODIES, 2))
        self.planet_vel = np.zeros((n, N_BODIES, 2))
        self.ship_pos = np.zeros((n, 2))
        self.ship_vel = np.zeros((n, 2))
        self.reset(np.arange(n))

    def reset(self, idx: np.ndarray):
        """
        Réinitialise les systèmes idx : chaque planète est placée à un point aléatoire de son orbite
        circulaire (comme un serveur qui tourne depuis un temps quelconque), le vaisseau à son point
        d'apparition, immobile.
        """
        k = len(idx)
        radius = np.linalg.norm(INITIAL_POSITIONS, axis=1)
        # Vitesse circulaire initiale (Planet::initial_speed), nulle pour le soleil
        speed = np.sqrt(G * MASSES[0] / np.where(radius > 0, radius, 1.0)) * (radius > 0)
        base_angle = np.arctan2(INITIAL_POSITIONS[:, 1], INITIAL_POSITIONS[:, 0])
        angle = base_angle + self.rng.uniform(0, 2 * np.pi, (k, N_BODIES))
        self.planet_pos[idx] = radius[:, None] * np.stack([np.cos(angle), np.sin(angle)], axis=-1)
        self.planet_vel[idx] = speed[:, None] * np.stack([-np.sin(angle), np.cos(angle)], axis=-1)
        self.ship_pos[idx] = SHIP_SPAWN
        self.ship_vel[idx] = 0.0

    def load(self, i: int, planet_pos: np.ndarray, planet_vel: np.ndarray, ship_pos: np.ndarray, ship_vel: np.ndarray):
        """Copie un état (par exemple reçu du serveur Rust) dans le système i."""
        self.planet_pos[i], self.planet_vel[i] = planet_pos, planet_vel
        self.ship_pos[i], self.ship_vel[i] = ship_pos, ship_vel

    def tick(self, thrust: np.ndarray):
        """
        Avance tous les systèmes d'un tick (1/60 s).

        Args:
            thrust (np.ndarray): Direction de poussée de chaque vaisseau, (n, 2), composantes dans {-1, 0, 1}.
        """
        sun = self.planet_pos[:, :1]
        # 1. Gravité du soleil sur les planètes
        d = self.planet_pos[:, 1:] - sun
        r3 = np.linalg.norm(d, axis=-1, keepdims=True) ** 3
        self.planet_vel[:, 1:] -= G * MASSES[0] * d / r3 * TICK
        # 2. Gravité de tous les corps sur les vaisseaux
        d = self.ship_pos[:, None, :] - self.planet_pos
        r3 = np.linalg.norm(d, axis=-1, keepdims=True) ** 3
        self.ship_vel -= (G * MASSES[None, :, None] * d / r3).sum(axis=1) * TICK
        # 3. Déplacement des planètes
        self.planet_pos += self.planet_vel * TICK
        # 4. Moteurs puis déplacement des vaisseaux
        self.ship_vel += thrust * ENGINE_ACCELERATION * TICK
        self.ship_pos += self.ship_vel * TICK
