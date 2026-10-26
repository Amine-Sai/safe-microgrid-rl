import os
import numpy as np

from agents.buffer import ReplayBuffer
from agents.running_stat import RunningMeanStd
from agents.shaping import PotentialBasedShaping


def test_buffer_termination_separation():
    """Terminated and truncated must be stored as INDEPENDENT flags, not one `done`.

    Storage is asserted directly because sample_buffer() samples WITH replacement,
    so a 2-element buffer cannot be relied on to return both flags.
    """
    buf = ReplayBuffer(max_size=100, input_shape=(17,), n_actions=6)
    s = np.zeros(17, dtype=np.float32)
    a = np.zeros(6, dtype=np.float32)

    # Store a truncated (time-limit) transition
    buf.store_transition(s, a, 1.0, s, terminated=False, truncated=True, cost=0.5)
    # Store a terminated transition
    buf.store_transition(s, a, 2.0, s, terminated=True, truncated=False, cost=1.0)

    # Independent storage: row 0 truncated-only, row 1 terminated-only.
    assert bool(buf.terminated_memory[0]) is False and bool(buf.truncated_memory[0]) is True
    assert bool(buf.terminated_memory[1]) is True and bool(buf.truncated_memory[1]) is False
    assert buf.cost_memory[0] == 0.5 and buf.cost_memory[1] == 1.0

    # Sampling returns both arrays separately, with correct lengths and dtypes.
    np.random.seed(0)
    states, actions, rewards, next_states, terminated, truncated, costs = buf.sample_buffer(2)
    assert len(terminated) == 2 and len(truncated) == 2 and len(costs) == 2
    assert terminated.dtype == bool and truncated.dtype == bool
    assert set(np.unique(terminated).tolist()) <= {False, True}
    assert set(np.unique(truncated).tolist()) <= {False, True}
    assert set(np.unique(costs).tolist()) <= {0.5, 1.0}
    assert set(np.unique(rewards).tolist()) <= {1.0, 2.0}


def test_running_mean_std():
    rms = RunningMeanStd(shape=(3,))
    data = np.array([
        [1.0, 10.0, 100.0],
        [2.0, 20.0, 200.0],
        [3.0, 30.0, 300.0],
    ], dtype=np.float32)

    for row in data:
        rms.update(row)

    norm = rms.normalize(data[1])
    assert norm.shape == (3,)
    assert np.all(np.isfinite(norm))


def test_potential_based_shaping():
    shaping = PotentialBasedShaping(gamma=0.99, kappa=2.0)
    s = np.zeros(17)
    s[0] = 0.5  # SoC = 0.5
    s_prime = np.zeros(17)
    s_prime[0] = 0.6  # SoC = 0.6

    f = shaping.shape(s, s_prime)
    expected = 0.99 * (2.0 * 0.6) - (2.0 * 0.5)
    assert np.isclose(f, expected, atol=1e-5)


def test_potential_shaping_episode_telescoping_sum():
    """Requirement: with gamma = 1, shaping terms sum over an episode to Phi(s_T) - Phi(s_0)."""
    shaping = PotentialBasedShaping(gamma=1.0, kappa=1.5)
    T = 168
    # Generate an arbitrary trajectory of SoC states in [0.2, 0.9]
    rng = np.random.RandomState(42)
    soc_trajectory = rng.uniform(0.20, 0.90, size=T + 1)

    terms = []
    for t in range(T):
        s_t = np.zeros(17)
        s_t[0] = soc_trajectory[t]
        s_next = np.zeros(17)
        s_next[0] = soc_trajectory[t + 1]

        f_t = shaping.shape(s_t, s_next)
        terms.append(f_t)

    sum_terms = np.sum(terms)
    s_0 = np.zeros(17); s_0[0] = soc_trajectory[0]
    s_T = np.zeros(17); s_T[0] = soc_trajectory[T]
    phi_0 = shaping.phi(s_0)
    phi_T = shaping.phi(s_T)
    expected_sum = phi_T - phi_0

    assert np.isclose(sum_terms, expected_sum, atol=1e-6), f"Sum {sum_terms} != Phi(s_T) - Phi(s_0) ({expected_sum})"
