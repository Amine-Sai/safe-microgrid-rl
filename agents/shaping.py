import numpy as np


class PotentialBasedShaping:
    """
    Potential-based reward shaping: F(s, s') = gamma * Phi(s') - Phi(s).
    Phi(s) = kappa * SoC.
    Guarantees policy invariance under gamma-discounted return.
    Phi is NOT zeroed on truncation (time-limit 168h boundaries).
    """

    def __init__(self, gamma: float = 0.99, kappa: float = 1.0):
        self.gamma = gamma
        self.kappa = kappa

    def phi(self, state: np.ndarray) -> float:
        # state[0] is SoC
        soc = float(state[0])
        return self.kappa * soc

    def shape(self, state: np.ndarray, next_state: np.ndarray) -> float:
        phi_s = self.phi(state)
        phi_s_prime = self.phi(next_state)
        return float(self.gamma * phi_s_prime - phi_s)


class NonPotentialShapedBonus:
    """
    Non-potential shaped bonus: F(s') = kappa * SoC'.
    Direct heuristic bonus without telescoping potential difference.
    """

    def __init__(self, kappa: float = 1.0):
        self.kappa = kappa

    def shape(self, next_state: np.ndarray) -> float:
        soc_prime = float(next_state[0])
        return float(self.kappa * soc_prime)
