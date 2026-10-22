import numpy as np


class RunningMeanStd:
    """Tracks running mean and variance using Welford's algorithm."""

    def __init__(self, shape: tuple, epsilon: float = 1e-4):
        self.shape = shape
        self.epsilon = epsilon
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count = epsilon

    def update(self, x: np.ndarray):
        """Update running statistics with a batch of observations."""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0] if x.ndim > len(self.shape) else 1

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = m_2 / tot_count

        self.mean = new_mean.astype(np.float32)
        self.var = new_var.astype(np.float32)
        self.count = tot_count

    def normalize(self, x: np.ndarray, clip_range: float = 10.0) -> np.ndarray:
        """Normalize observation array using current running statistics."""
        norm_x = (x - self.mean) / (np.sqrt(self.var) + self.epsilon)
        return np.clip(norm_x, -clip_range, clip_range).astype(np.float32)
