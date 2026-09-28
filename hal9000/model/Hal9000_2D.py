"""
Tâche Hal9000_2D : rejoindre les planètes une à une, dans un ordre aléatoire, sans tomber dans le
soleil ni sortir du système.

La logique est vectorisée sur n vaisseaux pour servir à la fois à la simulation numpy (n grand) et à
l'environnement connecté au serveur Rust (n = 1), avec exactement les mêmes observations et récompenses.
"""
import numpy as np
from gymnasium import spaces

from hal9000.sim.solar_system import ENGINE_ACCELERATION, G, MASSES

TARGETS = np.arange(1, 6)          # indices des planètes à rejoindre (0 = soleil)
TARGET_RADIUS = 200.0              # distance à laquelle une planète est atteinte
SUN_RADIUS = 150.0                 # en dessous : le vaisseau tombe dans le soleil
MAX_DISTANCE = 10000.0             # au-delà : le vaisseau est perdu (Jupiter orbite à ~7100)

# Échelles de normalisation des observations
LENGTH_SCALE = 5000.0
SPEED_SCALE = 100.0
SUN_GM = G * MASSES[0]
# Plafond de l'observation gravité / poussée (elle tend vers l'infini près du soleil)
MAX_GRAVITY_RATIO = 10.0

# Récompense (paramètres dans la section [reward] de config.toml) : bonus par planète atteinte, pénalité
# de mort, et guidage par le progrès vers la cible (distance avant - distance après) / shaping_scale.
# Sur un épisode, le guidage se résume à la distance
# totale parcourue vers les cibles : tourner autour ou osciller ne rapporte rien.
# (La forme « potential-based » gamma * phi(s') - phi(s) a été essayée : son terme (1 - gamma) * distance
# récompensait le fait de rester loin de la cible, et le modèle se contentait de survivre à distance.)
# Action Discrete(9) -> direction de poussée (x, y), chaque composante dans {-1, 0, 1}
THRUSTS = np.array([(x, y) for y in (-1, 0, 1) for x in (-1, 0, 1)], dtype=float)

OBSERVATION_SIZE = 16
observation_space = spaces.Box(-np.inf, np.inf, (OBSERVATION_SIZE,), np.float32)
action_space = spaces.Discrete(len(THRUSTS))


def thrust_to_engines(thrust: np.ndarray) -> dict[str, bool]:
    """
    Convertit une direction de poussée en commandes moteurs du serveur Rust
    (left pousse vers +x, right vers -x, up vers -y, down vers +y).
    """
    x, y = thrust
    return {"left": bool(x > 0), "right": bool(x < 0), "up": bool(y < 0), "down": bool(y > 0), "front": False, "back": False}


class Hal9000Task:
    """
    État de la tâche pour n vaisseaux : ordre des cibles, cible courante, distance précédente.

    Attributes:
        order (np.ndarray): Ordre de visite des planètes de chaque vaisseau, (n, 5).
        index (np.ndarray): Nombre de cibles atteintes pendant l'épisode, (n,) (la cible courante est
            order[index % 5] : le parcours recommence une fois toutes les planètes visitées).
        previous_distance (np.ndarray): Distance à la cible au step précédent, (n,).
    """

    def __init__(self, n: int, rng: np.random.Generator, reward: dict):
        self.n = n
        self.rng = rng
        self.reward = reward
        self.order = np.tile(TARGETS, (n, 1))
        self.index = np.zeros(n, dtype=int)
        self.previous_distance = np.zeros(n)

    def target(self, offset: int = 0) -> np.ndarray:
        """Indice de la planète cible (offset = 1 pour la suivante), (n,)."""
        return self.order[np.arange(self.n), (self.index + offset) % len(TARGETS)]

    def _target_distance(self, planet_pos: np.ndarray, ship_pos: np.ndarray) -> np.ndarray:
        return np.linalg.norm(planet_pos[np.arange(self.n), self.target()] - ship_pos, axis=-1)

    def reset(self, idx: np.ndarray, planet_pos: np.ndarray, ship_pos: np.ndarray):
        """Tire un nouvel ordre de visite pour les vaisseaux idx et initialise leur distance à la cible."""
        self.order[idx] = self.rng.permuted(np.tile(TARGETS, (len(idx), 1)), axis=1)
        self.index[idx] = 0
        self.previous_distance[idx] = self._target_distance(planet_pos, ship_pos)[idx]

    def observe(self, planet_pos, planet_vel, ship_pos, ship_vel) -> np.ndarray:
        """
        Observations relatives et normalisées, (n, 16) : position du vaisseau par rapport au soleil,
        sa vitesse, position et vitesse de la cible relatives au vaisseau, distances à la cible et au
        soleil, position de la cible suivante relative au vaisseau, puis les grandeurs orbitales :
        vitesses radiale et tangentielle par rapport au soleil, vitesse orbitale circulaire à cette
        distance, et gravité du soleil rapportée à la poussée d'un moteur. Près du soleil, la gravité
        dépasse la poussée : seule une vitesse tangentielle suffisante évite la chute.
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
        Calcule la récompense du step qui vient d'avoir lieu et met à jour la cible.

        Returns:
            tuple: La récompense (n,), la fin d'épisode par mort (n,), et les événements du step
                (reached, dead_sun, dead_out), chacun (n,).
        """
        distance = self._target_distance(planet_pos, ship_pos)
        sun_distance = np.linalg.norm(ship_pos - planet_pos[:, 0], axis=-1)
        reached = distance < TARGET_RADIUS
        dead_sun = sun_distance < SUN_RADIUS
        dead_out = sun_distance > MAX_DISTANCE
        dead = dead_sun | dead_out

        reward = (self.previous_distance - distance) / self.reward["shaping_scale"]
        reward += self.reward["target_reward"] * reached - self.reward["death_penalty"] * dead

        # Nouvelle cible : le progrès se mesure désormais par rapport à celle-ci
        self.index += reached & ~dead
        self.previous_distance = np.where(reached, self._target_distance(planet_pos, ship_pos), distance)
        return reward, dead, {"reached": reached, "dead_sun": dead_sun, "dead_out": dead_out}
