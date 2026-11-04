"""Potential-Based Reward Shaping (PBRS) for microgrid energy management.

Implements the potential-based reward transformation:
    F(s, s') = gamma * Phi(s') - Phi(s)
with Phi(s) = kappa * SoC(s), guaranteeing policy invariance under discounted return.
"""

from typing import Union
import numpy as np


class PotentialBasedShaping:
    r"""Potential-based reward shaping with battery state-of-charge potential.

    Based on Ng, Harada, and Russell (1999), "Policy invariance under reward transformations:
    Theory and application to reward shaping". Adding a potential-difference shaping term
    $F(s, s') = \gamma \Phi(s') - \Phi(s)$ preserves the set of optimal policies:

    $$\Pi^*_{(R + F)} = \Pi^*_R$$

    For an episode of length $T$ with discount factor $\gamma$:
    $$\sum_{t=0}^{T-1} \gamma^t F(s_t, s_{t+1}) = \gamma^T \Phi(s_T) - \Phi(s_0)$$

    When $\gamma = 1$, this telescopes directly to $\Phi(s_T) - \Phi(s_0)$.

    In the microgrid environment:
        $\Phi(s) = \kappa \cdot \text{SoC}(s)$
    where $s[0]$ is the battery state of charge $\in [0, 1]$.

    Note:
        At episode truncation (e.g. 168h rolling boundary), $\Phi(s_T)$ is NOT zeroed,
        preserving the true potential boundary condition across contiguous weeks.
    """

    def __init__(self, gamma: float = 0.99, kappa: float = 1.0) -> None:
        r"""Initialise potential-based shaping.

        Args:
            gamma: Discount factor $\gamma \in (0, 1]$.
            kappa: Potential scale multiplier $\kappa > 0$.
        """
        self.gamma: float = float(gamma)
        self.kappa: float = float(kappa)

    def phi(self, state: Union[np.ndarray, list]) -> float:
        r"""Compute the scalar potential function $\Phi(s) = \kappa \cdot \text{SoC}$.

        Args:
            state: Environment state observation array, where state[0] is SoC.

        Returns:
            Scalar potential value $\Phi(s)$.
        """
        soc = float(state[0])
        return float(self.kappa * soc)

    def shape(
        self,
        state: Union[np.ndarray, list],
        next_state: Union[np.ndarray, list],
    ) -> float:
        r"""Compute shaping reward $F(s, s') = \gamma \Phi(s') - \Phi(s)$.

        Args:
            state: Current state observation $s_t$.
            next_state: Next state observation $s_{t+1}$.

        Returns:
            Scalar shaping term $F(s_t, s_{t+1})$.
        """
        phi_s = self.phi(state)
        phi_s_prime = self.phi(next_state)
        return float(self.gamma * phi_s_prime - phi_s)
