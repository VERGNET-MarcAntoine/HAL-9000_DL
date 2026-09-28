import numpy as np
from stable_baselines3.common.vec_env import VecEnv

from hal9000.config import Config
from hal9000.model import Hal9000_2D as task
from hal9000.sim.solar_system import SolarSystemSim


class Hal9000SimVecEnv(VecEnv):
    """
    n vaisseaux de la tâche Hal9000_2D sur la simulation numpy, sous forme de VecEnv SB3 : tout est
    vectorisé dans un seul processus, sans serveur ni communication réseau.
    """

    def __init__(self, n: int, config: Config, seed: int | None = None, shared_planets: bool = False):
        """
        Args:
            n (int): Le nombre de vaisseaux.
            config (Config): La configuration.
            seed (int, optional): La graine aléatoire.
            shared_planets (bool): Tous les vaisseaux dans le même système solaire, comme sur le serveur
                Rust (pour l'affichage). Sinon, chaque vaisseau a son propre système, ce qui diversifie
                les situations rencontrées à l'entraînement.
        """
        super().__init__(n, task.observation_space, task.action_space)
        self.shared_planets = shared_planets
        # Positions (soleil, planètes, vaisseaux) à chaque tick du dernier step, si record_ticks
        self.record_ticks = False
        self.tick_history = []
        # Diffusion en direct pour `display_ship2D --training` (voir hal9000/sim/live.py)
        self.publisher = None
        self.rng = np.random.default_rng(seed)
        self.sim = SolarSystemSim(n, self.rng)
        self.task = task.Hal9000Task(n, self.rng, config.reward)
        self.ticks_per_step = round(config.decision_interval * 60)
        self.max_steps = config.steps_per_episode
        self.steps = np.zeros(n, dtype=int)
        self.actions = np.zeros(n, dtype=int)

    def _observe(self) -> np.ndarray:
        s = self.sim
        return self.task.observe(s.planet_pos, s.planet_vel, s.ship_pos, s.ship_vel)

    def _reset(self, idx: np.ndarray):
        self.sim.reset(idx)
        others = np.setdiff1d(np.arange(self.num_envs), idx)
        if self.shared_planets and len(others):
            # Les planètes ne dépendent pas des vaisseaux : copiées d'un autre système, elles restent synchronisées
            self.sim.planet_pos[idx] = self.sim.planet_pos[others[0]]
            self.sim.planet_vel[idx] = self.sim.planet_vel[others[0]]
        elif self.shared_planets:
            self.sim.planet_pos[:] = self.sim.planet_pos[0]
            self.sim.planet_vel[:] = self.sim.planet_vel[0]
        self.task.reset(idx, self.sim.planet_pos, self.sim.ship_pos)
        self.steps[idx] = 0

    def reset(self) -> np.ndarray:
        self._reset(np.arange(self.num_envs))
        return self._observe()

    def step_async(self, actions: np.ndarray):
        self.actions = np.asarray(actions)

    def step_wait(self):
        thrust = task.THRUSTS[self.actions]
        self.tick_history = []
        for _ in range(self.ticks_per_step):
            self.sim.tick(thrust)
            if self.record_ticks:
                self.tick_history.append((self.sim.planet_pos[0].copy(), self.sim.ship_pos.copy()))

        reward, terminated, events = self.task.transition(self.sim.planet_pos, self.sim.ship_pos)
        self.steps += 1
        truncated = (self.steps >= self.max_steps) & ~terminated
        done = terminated | truncated
        obs = self._observe()

        infos = [{} for _ in range(self.num_envs)]
        finished = np.flatnonzero(done)
        for i in finished:
            infos[i] = {
                "terminal_observation": obs[i],
                "TimeLimit.truncated": bool(truncated[i]),
                "hal": {"targets": int(self.task.index[i]), "dead_sun": bool(events["dead_sun"][i]),
                        "dead_out": bool(events["dead_out"][i])},
            }
        if self.publisher:
            self.publisher.publish(self, [infos[i]["hal"] for i in finished])
        if len(finished):
            self._reset(finished)
            obs[finished] = self._observe()[finished]
        return obs, reward.astype(np.float32), done, infos

    def close(self):
        pass

    def seed(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)
        self.sim.rng = self.task.rng = self.rng
        return [seed] * self.num_envs

    def get_attr(self, attr_name, indices=None):
        return [getattr(self, attr_name)] * len(self._get_indices(indices))

    def set_attr(self, attr_name, value, indices=None):
        setattr(self, attr_name, value)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        raise NotImplementedError

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * len(self._get_indices(indices))
