import numpy as np
from hal9000.model.core.ship2D import Ship2D
from hal9000.model.core.training import train
from hal9000.config import Config
from pathlib import Path


class Hal9000_2D_V0(Ship2D):
    """
    A custom environment that inherits from Ship2D and defines a specific reward function.
    """

    def __init__(self, config: Config | None = None):
        """
        Initializes the MyShipEnv.
        """
        super().__init__(config)
        # You can add any specific initialization code here if needed
        self.previous_distance_to_target = None  # To track progress towards the target

    def get_reward(self, previous_state: dict) -> tuple[float, bool]:
        """
        Calcule la récompense et détermine si l'épisode est terminé.
        Args:
            previous_state (dict): L'état précédent de l'environnement.

        Returns:
            tuple[float, bool]: La récompense et un booléen indiquant si l'épisode est terminé.
        """

        previous_ship_data = self.get_ship_data(previous_state)
        previous_target_data, previous_planet_data = self.get_planet_data(
            previous_state)

        ship_data = self.get_ship_data(self.state)
        target_data, planet_data = self.get_planet_data(
            self.state)

        previous_distance_sun = np.linalg.norm(
            previous_ship_data[0:2] - previous_planet_data[0:2])
        distance_sun = np.linalg.norm(ship_data[0:2] - planet_data[0:2])

        if distance_sun > 20000 or distance_sun < 150:
            return -1000, True  # Mort fin de l'eposide avec pénalité sevére

        previous_distance_target = np.linalg.norm(
            previous_ship_data[0:2] - previous_target_data[0:2])
        distance_target = np.linalg.norm(ship_data[0:2] - target_data[0:2])

        delta_distance = previous_distance_target - distance_target

        # Récompense principale : réduction de la distance cible
        reward = (-distance_target / 20000)

        if delta_distance > 0:
            # Bonus si on se rapproche
            reward += (delta_distance / 500)

        # Pénalité progressive pour la proximité au soleil
        if distance_sun < 500:
            # Pénalité plus douce en s'approchant
            reward -= (500 - distance_sun) / 500
        elif distance_sun > 3000:
            # Pénalité croissante si trop loin
            reward -= (distance_sun - 3000) / 2000

        # Récompense progressive pour atteindre l’objectif
        if distance_target < 200:
            reward += 1000
            self.current_target += 1
            print(f"Score : {self.current_target}")
            if self.current_target >= len(self.target_ids):
                self.current_target = 0

        # Récompense basée sur l'accélération (direction vers la cible)
        acceleration = ship_data[2:4] - previous_ship_data[2:4]

        direction_to_target = target_data[0:2] - ship_data[0:2]
        # Normalisation
        direction_to_target /= max(np.linalg.norm(direction_to_target), 1e-8)
        alignment_reward = np.dot(acceleration, direction_to_target)
        # Modulation de la récompense d'alignement selon la distance
        # Diminue l'importance en se rapprochant
        weight = min(1, distance_target / 1000)

        if alignment_reward > 0:
            reward += weight * alignment_reward * 5  # Bonus si aligné
        else:
            reward += weight * alignment_reward * 2  # Pénalité si opposé

        speed_norm = np.linalg.norm(ship_data[2:4])
        if speed_norm < 5:  # Trop lent, risque de stagnation
            reward -= (5 - speed_norm) / 2  # Pénalité progressive
        elif speed_norm > 100:  # Trop rapide, risque de perte de contrôle
            reward -= (speed_norm - 100) / 10  # Pénalité progressive
        return reward, False


if __name__ == "__main__":
    train(Hal9000_2D_V0, Path(__file__).stem)
