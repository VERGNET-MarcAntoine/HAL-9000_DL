"""
Hal9000_2D task: reach the planets one after the other, in a random order, without falling into the sun
or leaving the solar system.

The logic is vectorized over n ships to serve both the numpy simulation (large n) and the environment
connected to the Rust server (n = 1), with exactly the same observations and rewards.
"""
import numpy as np
from gymnasium import spaces

from hal9000.sim.solar_system import ENGINE_ACCELERATION, MASSES, G

TARGETS = np.arange(1, 6)          # indices of the planets to reach (0 = sun)
TARGET_RADIUS = 200.0              # distance at which a planet is reached
SUN_RADIUS = 150.0                 # below: the ship falls into the sun
MAX_DISTANCE = 10000.0             # beyond: the ship is lost (Jupiter orbits at ~7100)

# Normalization scales of the observations
LENGTH_SCALE = 5000.0
SPEED_SCALE = 100.0
SUN_GM = G * MASSES[0]
# Cap of the gravity / thrust observation (it tends to infinity near the sun)
MAX_GRAVITY_RATIO = 10.0

# Reward (parameters in the [reward] section of config.toml): bonus per planet reached, death penalty,
# and guidance by the progress toward the target, (distance before - distance after) / shaping_scale.
# Over an episode, the guidance amounts to the total distance covered toward the targets: circling or
# oscillating earns nothing.
# (The potential-based form gamma * phi(s') - phi(s) was tried: its (1 - gamma) * distance term rewarded
# staying far from the target, and the model settled for surviving at a distance.)

# Action Discrete(9) -> thrust direction (x, y), each component in {-1, 0, 1}
THRUSTS = np.array([(x, y) for y in (-1, 0, 1) for x in (-1, 0, 1)], dtype=float)

OBSERVATION_SIZE = 16
observation_space = spaces.Box(-np.inf, np.inf, (OBSERVATION_SIZE,), np.float32)
action_space = spaces.Discrete(len(THRUSTS))


def thrust_to_engines(thrust: np.ndarray) -> dict[str, bool]:
    """
    Converts a thrust direction into engine commands of the Rust server
    (left pushes toward +x, right toward -x, up toward -y, down toward +y).
    """
    x, y = thrust
    return {"left": bool(x > 0), "right": bool(x < 0), "up": bool(y < 0), "down": bool(y > 0), "front": False, "back": False}


class Hal9000Task:
    """
    State of the task for n ships: order of the targets, current target, previous distance.

    Attributes:
        order (np.ndarray): Order in which each ship visits the planets, (n, 5).
        index (np.ndarray): Number of targets reached during the episode, (n,) (the current target is
            order[index % 5]: the tour starts again once every planet has been visited).
        previous_distance (np.ndarray): Distance to the target at the previous step, (n,).
    """

    def __init__(self, n: int, rng: np.random.Generator, reward: dict):
        self.n = n
        self.rng = rng
        self.reward = reward
        self.order = np.tile(TARGETS, (n, 1))
        self.index = np.zeros(n, dtype=int)
        self.previous_distance = np.zeros(n)

    def target(self, offset: int = 0) -> np.ndarray:
        """Index of the target planet (offset = 1 for the next one), (n,)."""
        return self.order[np.arange(self.n), (self.index + offset) % len(TARGETS)]

    def _target_distance(self, planet_pos: np.ndarray, ship_pos: np.ndarray) -> np.ndarray:
        return np.linalg.norm(planet_pos[np.arange(self.n), self.target()] - ship_pos, axis=-1)

    def reset(self, idx: np.ndarray, planet_pos: np.ndarray, ship_pos: np.ndarray):
        """Draws a new visiting order for the ships idx and initializes their distance to the target."""
        self.order[idx] = self.rng.permuted(np.tile(TARGETS, (len(idx), 1)), axis=1)
        self.index[idx] = 0
        self.previous_distance[idx] = self._target_distance(planet_pos, ship_pos)[idx]

    def observe(self, planet_pos, planet_vel, ship_pos, ship_vel) -> np.ndarray:
        """
        Relative and normalized observations, (n, 16): position of the ship relative to the sun, its
        speed, position and speed of the target relative to the ship, distances to the target and to the
        sun, position of the next target relative to the ship, then the orbital quantities: radial and
        tangential speed relative to the sun, circular orbital speed at this distance, and the sun's
        gravity compared to the thrust of an engine. Near the sun, gravity exceeds the thrust: only a
        sufficient tangential speed avoids the fall.
        """
        rows = np.arange(self.n)
        sun = planet_pos[:, 0]
        target = self.target()
        to_target = planet_pos[rows, target] - ship_pos
        target_vel = planet_vel[rows, target] - ship_vel
        to_next = planet_pos[rows, self.target(1)] - ship_pos
        from_sun = ship_pos - sun
        sun_distance = np.linalg.norm(from_sun, axis=-1, keepdims=True)
        radial = from_sun / sun_distance
        radial_speed = (ship_vel * radial).sum(axis=-1, keepdims=True)
        tangential_speed = radial[:, :1] * ship_vel[:, 1:] - radial[:, 1:] * ship_vel[:, :1]
        circular_speed = np.sqrt(SUN_GM / sun_distance)
        gravity_ratio = np.minimum(SUN_GM / sun_distance ** 2 / ENGINE_ACCELERATION, MAX_GRAVITY_RATIO)
        return np.concatenate([
            from_sun / LENGTH_SCALE,
            ship_vel / SPEED_SCALE,
            to_target / LENGTH_SCALE,
            target_vel / SPEED_SCALE,
            np.linalg.norm(to_target, axis=-1, keepdims=True) / LENGTH_SCALE,
            sun_distance / LENGTH_SCALE,
            to_next / LENGTH_SCALE,
            radial_speed / SPEED_SCALE,
            tangential_speed / SPEED_SCALE,
            circular_speed / SPEED_SCALE,
            gravity_ratio,
        ], axis=-1).astype(np.float32)

    def transition(self, planet_pos, ship_pos) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        """
        Computes the reward of the step that just happened and updates the target.

        Returns:
            tuple: The reward (n,), the end of the episode by death (n,), and the events of the step
                (reached, dead_sun, dead_out), each (n,).
        """
        distance = self._target_distance(planet_pos, ship_pos)
        sun_distance = np.linalg.norm(ship_pos - planet_pos[:, 0], axis=-1)
        reached = distance < TARGET_RADIUS
        dead_sun = sun_distance < SUN_RADIUS
        dead_out = sun_distance > MAX_DISTANCE
        dead = dead_sun | dead_out

        reward = (self.previous_distance - distance) / self.reward["shaping_scale"]
        reward += self.reward["target_reward"] * reached - self.reward["death_penalty"] * dead

        # New target: the progress is now measured toward it
        self.index += reached & ~dead
        self.previous_distance = np.where(reached, self._target_distance(planet_pos, ship_pos), distance)
        return reward, dead, {"reached": reached, "dead_sun": dead_sun, "dead_out": dead_out}
