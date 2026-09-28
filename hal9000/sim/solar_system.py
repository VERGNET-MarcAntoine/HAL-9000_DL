"""
Vectorized (numpy) replica of the physics of the Rust server (deep_learning branch), to train much faster
than through the server. Each tick reproduces SolarSystem::update, in the same order:
  1. gravity of the sun on the planets (speeds)
  2. gravity of every body on the ships (speeds, with the positions of the planets not yet updated)
  3. motion of the planets
  4. thrust of the engines, then motion of the ships
"""
import numpy as np

G = 6.67430e-11
TICK = 1 / 60

# rust-server/src/body/solar_system.rs: (mass, initial position), the sun first
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
ENGINE_ACCELERATION = 10000.0 / 1000.0  # engine power / mass of the ship


class SolarSystemSim:
    """
    n independent solar systems, each with one ship.

    Attributes:
        planet_pos (np.ndarray): Positions of the bodies, (n, N_BODIES, 2).
        planet_vel (np.ndarray): Speeds of the bodies, (n, N_BODIES, 2).
        ship_pos (np.ndarray): Positions of the ships, (n, 2).
        ship_vel (np.ndarray): Speeds of the ships, (n, 2).
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
        Resets the systems idx: each planet is placed at a random point of its circular orbit (like a
        server that has been running for some time), the ship at its spawn point, motionless.
        """
        k = len(idx)
        radius = np.linalg.norm(INITIAL_POSITIONS, axis=1)
        # Initial circular speed (Planet::initial_speed), zero for the sun
        speed = np.sqrt(G * MASSES[0] / np.where(radius > 0, radius, 1.0)) * (radius > 0)
        base_angle = np.arctan2(INITIAL_POSITIONS[:, 1], INITIAL_POSITIONS[:, 0])
        angle = base_angle + self.rng.uniform(0, 2 * np.pi, (k, N_BODIES))
        self.planet_pos[idx] = radius[:, None] * np.stack([np.cos(angle), np.sin(angle)], axis=-1)
        self.planet_vel[idx] = speed[:, None] * np.stack([-np.sin(angle), np.cos(angle)], axis=-1)
        self.ship_pos[idx] = SHIP_SPAWN
        self.ship_vel[idx] = 0.0

    def load(self, i: int, planet_pos: np.ndarray, planet_vel: np.ndarray, ship_pos: np.ndarray, ship_vel: np.ndarray):
        """Copies a state (for example received from the Rust server) into the system i."""
        self.planet_pos[i], self.planet_vel[i] = planet_pos, planet_vel
        self.ship_pos[i], self.ship_vel[i] = ship_pos, ship_vel

    def tick(self, thrust: np.ndarray):
        """
        Advances every system by one tick (1/60 s).

        Args:
            thrust (np.ndarray): Thrust direction of each ship, (n, 2), components in {-1, 0, 1}.
        """
        sun = self.planet_pos[:, :1]
        # 1. Gravity of the sun on the planets
        d = self.planet_pos[:, 1:] - sun
        r3 = np.linalg.norm(d, axis=-1, keepdims=True) ** 3
        self.planet_vel[:, 1:] -= G * MASSES[0] * d / r3 * TICK
        # 2. Gravity of every body on the ships
        d = self.ship_pos[:, None, :] - self.planet_pos
        r3 = np.linalg.norm(d, axis=-1, keepdims=True) ** 3
        self.ship_vel -= (G * MASSES[None, :, None] * d / r3).sum(axis=1) * TICK
        # 3. Motion of the planets
        self.planet_pos += self.planet_vel * TICK
        # 4. Engines, then motion of the ships
        self.ship_vel += thrust * ENGINE_ACCELERATION * TICK
        self.ship_pos += self.ship_vel * TICK
