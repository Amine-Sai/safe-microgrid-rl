"""Running statistics tracker using Welford's algorithm and Chan's parallel update.

Provides numerically stable, online estimation of running empirical mean and variance
for observation and return normalization in continuous RL algorithms.
"""

from typing import Tuple, Union
import numpy as np


class RunningMeanStd:
    r"""Tracks running mean and variance using Chan et al. (1979) parallel update.

    Given a running accumulator $A$ of count $n_A$, mean $\mu_A$, and variance $\sigma_A^2$,
    and a new batch $B$ of count $n_B$, mean $\mu_B$, and variance $\sigma_B^2$, the combined
    sample moments are updated via:

    $$\delta = \mu_B - \mu_A$$
    $$n_{\text{tot}} = n_A + n_B$$
    $$\mu_{\text{new}} = \mu_A + \delta \frac{n_B}{n_{\text{tot}}}$$
    $$M_{2, \text{new}} = n_A \sigma_A^2 + n_B \sigma_B^2 + \delta^2 \frac{n_A n_B}{n_{\text{tot}}}$$
    $$\sigma_{\text{new}}^2 = \frac{M_{2, \text{new}}}{n_{\text{tot}}}$$

    References:
        - Welford, B. P. (1962). "Note on a method for calculating corrected sums of squares
          and products". Technometrics 4(3):419-420.
        - Chan, T. F., Golub, G. H., & LeVeque, R. J. (1979). "Updating Formulae and a Pairwise
          Algorithm for Computing Sample Variances". Technical Report STAN-CS-79-773.
    """

    def __init__(self, shape: Tuple[int, ...] = (), epsilon: float = 1e-4) -> None:
        """Initialise running mean and variance accumulators.

        Args:
            shape: Shape tuple of the array to track, e.g. (state_dim,) or () for scalar.
            epsilon: Small positive float for numerical stability and initial pseudo-count.
        """
        self.shape: Tuple[int, ...] = shape
        self.epsilon: float = float(epsilon)
        self.mean: np.ndarray = np.zeros(shape, dtype=np.float32)
        self.var: np.ndarray = np.ones(shape, dtype=np.float32)
        self.count: float = float(epsilon)

    def update(self, x: Union[np.ndarray, float]) -> None:
        """Update running statistics with a new batch or single sample.

        Args:
            x: Input array of observations or returns to incorporate into running stats.
        """
        arr = np.asarray(x, dtype=np.float32)
        batch_mean = np.mean(arr, axis=0)
        batch_var = np.var(arr, axis=0)
        batch_count = arr.shape[0] if arr.ndim > len(self.shape) else 1

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * (batch_count / tot_count)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * (self.count * batch_count / tot_count)
        new_var = m_2 / tot_count

        self.mean = new_mean.astype(np.float32)
        self.var = new_var.astype(np.float32)
        self.count = float(tot_count)

    def normalize(self, x: Union[np.ndarray, float], clip_range: float = 10.0) -> np.ndarray:
        r"""Normalize an array using the current empirical mean and variance.

        $$\hat{x} = \operatorname{clip}\left(\frac{x - \mu}{\sqrt{\sigma^2} + \epsilon}, -c, c\right)$$

        Args:
            x: Input array or scalar to normalize.
            clip_range: Maximum absolute value for clipping normalized outputs.

        Returns:
            Normalized and clipped array with dtype np.float32.
        """
        arr = np.asarray(x, dtype=np.float32)
        norm_x = (arr - self.mean) / (np.sqrt(self.var) + self.epsilon)
        return np.clip(norm_x, -clip_range, clip_range).astype(np.float32)
