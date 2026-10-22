import numpy as np


class RunningNormalizer:
    """Online running mean/std normalizer using Welford's algorithm.

    Normalizes observations to approximately zero mean, unit variance.
    Can be frozen (stop updating statistics) for evaluation.
    """

    def __init__(self, shape: tuple, clip: float = 10.0, epsilon: float = 1e-8):
        self.shape = shape
        self.clip = clip
        self.epsilon = epsilon

        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 0
        self._m2 = np.zeros(shape, dtype=np.float64)  # For Welford's
        self.frozen = False

    def update(self, x: np.ndarray) -> None:
        """Update running statistics with a new observation."""
        if self.frozen:
            return
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            self._update_single(x)
        else:
            for xi in x:
                self._update_single(xi)

    def _update_single(self, x: np.ndarray) -> None:
        """Welford's online algorithm for single observation."""
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self._m2 += delta * delta2
        if self.count > 1:
            self.var = self._m2 / (self.count - 1)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Normalize observation to ~N(0, 1), clipped."""
        std = np.sqrt(self.var + self.epsilon)
        normalized = (x - self.mean.astype(np.float32)) / std.astype(np.float32)
        return np.clip(normalized, -self.clip, self.clip).astype(np.float32)

    def freeze(self) -> None:
        """Stop updating statistics (for evaluation)."""
        self.frozen = True

    def unfreeze(self) -> None:
        """Resume updating statistics."""
        self.frozen = False

    def state_dict(self) -> dict:
        """Return state for serialization."""
        return {
            "mean": self.mean.copy(),
            "var": self.var.copy(),
            "count": self.count,
            "_m2": self._m2.copy(),
            "frozen": self.frozen,
        }

    def load_state_dict(self, state: dict) -> None:
        """Load state from serialized dict."""
        self.mean = state["mean"].copy()
        self.var = state["var"].copy()
        self.count = state["count"]
        self._m2 = state["_m2"].copy()
        self.frozen = state.get("frozen", False)


class RewardNormalizer:
    """Running reward normalizer using return-based variance estimation.

    Normalizes rewards by dividing by the running standard deviation.
    Does NOT subtract mean (to preserve reward sign and relative ordering).
    """

    def __init__(self, clip: float = 10.0, epsilon: float = 1e-8, gamma: float = 0.99):
        self.clip = clip
        self.epsilon = epsilon
        self.gamma = gamma

        self._ret = 0.0  # Running discounted return estimate
        self.mean = 0.0
        self.var = 1.0
        self.count = 0
        self._m2 = 0.0
        self.frozen = False

    def update(self, reward: float) -> None:
        """Update with a new reward value."""
        if self.frozen:
            return
        self._ret = self._ret * self.gamma + reward
        self.count += 1
        delta = self._ret - self.mean
        self.mean += delta / self.count
        delta2 = self._ret - self.mean
        self._m2 += delta * delta2
        if self.count > 1:
            self.var = self._m2 / (self.count - 1)

    def normalize(self, reward: float) -> float:
        """Normalize reward by running std (no mean subtraction)."""
        std = np.sqrt(self.var + self.epsilon)
        normalized = reward / std
        return float(np.clip(normalized, -self.clip, self.clip))

    def reset_return(self) -> None:
        """Reset the running return at episode boundary."""
        self._ret = 0.0

    def freeze(self) -> None:
        self.frozen = True

    def unfreeze(self) -> None:
        self.frozen = False

    def state_dict(self) -> dict:
        return {
            "mean": self.mean,
            "var": self.var,
            "count": self.count,
            "_m2": self._m2,
            "_ret": self._ret,
            "frozen": self.frozen,
        }

    def load_state_dict(self, state: dict) -> None:
        self.mean = state["mean"]
        self.var = state["var"]
        self.count = state["count"]
        self._m2 = state["_m2"]
        self._ret = state.get("_ret", 0.0)
        self.frozen = state.get("frozen", False)
