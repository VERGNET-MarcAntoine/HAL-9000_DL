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
        target_data, planet_data = self.get_planet_data(self.state)

        # print(f"""
        # --- ÉTAT PRÉCÉDENT ---
        # Vaisseau: {previous_ship_data}
        # Cible: {previous_target_data}
        # Planètes: {previous_planet_data}

        # --- ÉTAT ACTUEL ---
        # Vaisseau: {ship_data}
        # Cible: {target_data}
        # Planètes: {planet_data}
        # """)

        previous_ship_data = self.get_ship_data(previous_state)
        previous_target_data, previous_planet_data = self.get_planet_data(
            previous_state)

        ship_data = self.get_ship_data(self.state)
        target_data, planet_data = self.get_planet_data(self.state)

        # Calcul des distances
        previous_distance_sun = np.linalg.norm(
            previous_ship_data[0:2] - previous_planet_data[0:2])
        distance_sun = np.linalg.norm(ship_data[0:2] - planet_data[0:2])

        # Vérification des limites de la gravité et fin d'épisode si nécessaire
        if distance_sun > 20000 or distance_sun < 150:
            # Pénalité sévère et fin de l'épisode si trop loin ou trop près du soleil
            return -1000, True

        # Calcul des distances par rapport à la cible
        previous_distance_target = np.linalg.norm(
            previous_ship_data[0:2] - previous_target_data[0:2])
        distance_target = np.linalg.norm(ship_data[0:2] - target_data[0:2])

        # Calcul de la variation de la distance à la cible
        delta_distance = previous_distance_target - distance_target

        # Récompense pour la réduction de la distance cible
        reward = delta_distance / max(distance_target, 1e-8)

        # Récompense pour accélérer vers la cible (calcul de l'accélération)
        acceleration = ship_data[2:4] - previous_ship_data[2:4]
        target_direction = target_data[0:2] - ship_data[0:2]
        # Normalisation pour obtenir une direction unitaire
        target_direction /= max(np.linalg.norm(target_direction), 1e-8)

        # Calcul du produit scalaire entre l'accélération et la direction de la cible (récompense si on accélère dans la bonne direction)
        acceleration_dot_target = np.dot(acceleration, target_direction)
        # Récompense proportionnelle à l'alignement de l'accélération avec la cible
        reward += 10 * acceleration_dot_target

        # Pénalisation pour accélérer dans la direction opposée à la cible
        if acceleration_dot_target < 0:
            reward -= 5  # Pénalité si l'accélération est dirigée à l'opposé de la cible

        # Récompense basée sur la vitesse (bonus pour une vitesse modérée et efficace)
        speed = np.linalg.norm(ship_data[2:4])
        if speed > 50:
            # Pénalisation si la vitesse est trop élevée (risque de dépassement ou de gaspillage de carburant)
            reward -= 1

        # Récompense pour atteindre l'objectif
        if distance_target < 200:
            reward += 1000  # Grande récompense lorsqu'on atteint la cible
            self.current_target += 1
            print(f"Score : {self.current_target}")
            if self.current_target >= len(self.target_ids):
                self.current_target = 0
            # Retourner la récompense et l'indication que l'épisode n'est pas encore terminé
            return reward, False

        # Retourner la récompense et indiquer que l'épisode continue
        return reward, False


if __name__ == "__main__":
    train(Hal9000_2D_V0, Path(__file__).stem)
