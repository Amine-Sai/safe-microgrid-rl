import os
import sys

import numpy as np
import pytest
import torch as th

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.ppo import PPOAgent, compute_gae, grid_only_action  # noqa: E402


# --------------------------------------------------------------------------- 1
def _gae_reference(rewards, values, dones, truncated, last_value, gamma, lam):
    """Independent, straightforward implementation — the oracle.

    Formula: delta_t = r_t + gamma * V_{t+1} * (1 - term_t) - V_t
             A_t     = delta_t + gamma * lam * (1 - term_t) * A_{t+1}
    At t = T-1: V_{t+1} = 0 if terminal else last_value.
    """
    T = len(rewards)
    adv = np.zeros(T)
    gae = 0.0
    for t in reversed(range(T)):
        term = 1.0 if dones[t] else 0.0
        if t == T - 1:
            nv = 0.0 if dones[t] else last_value
        else:
            nv = values[t + 1]
        delta = rewards[t] + gamma * nv * (1.0 - term) - values[t]
        gae = delta + gamma * lam * (1.0 - term) * gae
        adv[t] = gae
    return adv


def test_gae_matches_numpy_reference():
    """Hand-checkable 5-step case: terminal at step 3, truncation at step 1.

    gamma=0.5, lam=0.5 for easy manual verification.
    rewards = [1, 0, 2, 1, 3], values = [1, 2, 1, 3, 2], last_value = 4.

    Manual calculation (backwards):
      t=4: nt=1, nv=4,     delta = 3+0.5*4-2 = 3.0,   A=3.0
      t=3: nt=0, nv=v[4],  delta = 1+0-3     = -2.0,  A=-2.0 (resets)
      t=2: nt=1, nv=v[3],  delta = 2+1.5-1   = 2.5,   A=2.5+0.25*(-2)=2.0
      t=1: nt=1, nv=v[2],  delta = 0+0.5-2   = -1.5,  A=-1.5+0.25*2.0=-1.0
      t=0: nt=1, nv=v[1],  delta = 1+1-1     = 1.0,   A=1.0+0.25*(-1.0)=0.75
    """
    r = np.array([1.0, 0.0, 2.0, 1.0, 3.0])
    v = np.array([1.0, 2.0, 1.0, 3.0, 2.0])
    d = np.array([0, 0, 0, 1, 0], dtype=bool)      # terminal at step 3
    tr = np.array([0, 1, 0, 0, 0], dtype=bool)      # truncation at step 1
    expected = np.array([0.75, -1.0, 2.0, -2.0, 3.0])

    adv = compute_gae(r, v, d, tr, last_value=4.0, gamma=0.5, lam=0.5)[0]
    ref = _gae_reference(r, v, d, tr, 4.0, 0.5, 0.5)

    assert np.allclose(adv, expected, atol=1e-10), f"compute_gae: {adv}"
    assert np.allclose(ref, expected, atol=1e-10), f"reference: {ref}"
    assert np.allclose(adv, ref, atol=1e-10)


# --------------------------------------------------------------------------- 2
def test_truncation_bootstrapping():
    """A time-limit end must bootstrap V(s_T), a terminal end must not."""
    r = np.array([1.0, 1.0, 1.0])
    v = np.array([0.5, 0.5, 0.5])
    d_terminal = np.array([0, 0, 1], dtype=bool)
    d_trunc = np.array([0, 0, 0], dtype=bool)
    trunc_flag = np.array([0, 0, 1], dtype=bool)

    adv_trunc = compute_gae(r, v, d_trunc, trunc_flag, last_value=100.0,
                            gamma=0.99, lam=0.95)[0]
    adv_term = compute_gae(r, v, d_terminal, trunc_flag, last_value=100.0,
                           gamma=0.99, lam=0.95)[0]
    # the terminal episode gets NO bootstrap: its last advantage must be far smaller
    assert adv_trunc[-1] > adv_term[-1] + 50.0
    assert adv_term[-1] == pytest.approx(1.0 - 0.5, abs=1e-9)


# --------------------------------------------------------------------------- 3
def test_seed_determinism():
    """Two full seeded runs (rollout + finish_path + update) produce identical results."""
    results = []
    for _ in range(2):
        th.manual_seed(42)
        np.random.seed(42)
        ag = PPOAgent(state_dim=4, action_dim=2, lagrangian=True, cost_limit=0.05)

        actions, values = [], []
        states = np.random.randn(20, 4).astype(np.float32)
        for i, s in enumerate(states):
            a, lp, v_val, cv = ag.select_action(s)
            actions.append(a.copy())
            values.append(v_val)
            ag.store_transition(s, a, float(np.random.randn()), abs(float(np.random.randn())),
                                v_val, cv, lp, (i == 19), False)

        batch = ag.finish_path(0.0, 0.0)
        diag = ag.update(batch)
        with th.no_grad():
            final_mu = ag.policy.forward(th.zeros(1, 4)).numpy().ravel().copy()
        results.append((np.array(actions), np.array(values), diag, final_mu))

    assert np.allclose(results[0][0], results[1][0]), "actions differ"
    assert np.allclose(results[0][1], results[1][1]), "values differ"
    assert np.allclose(results[0][3], results[1][3], atol=1e-6), "final policy differs"


# --------------------------------------------------------------------------- 4
def test_lambda_nonnegative_and_rises_when_cost_exceeds_d():
    ag = PPOAgent(state_dim=4, action_dim=2, lagrangian=True, cost_limit=0.05,
                  lambda_lr_mult=0.5)
    lam0 = ag.lambda_
    assert lam0 >= 0.0

    for _ in range(10):
        ag.rollout_costs = [5.0, 5.0, 5.0]        # way above d_main = 0.05
        ag.end_episode(terminated=True)
        lam = ag.update_lambda_from_source()
        assert lam >= 0.0
    assert ag.lambda_ > lam0
    assert ag.lambda_ == max(0.0, ag.lambda_)

    ag2 = PPOAgent(state_dim=4, action_dim=2, lagrangian=True, cost_limit=100.0,
                   lambda_lr_mult=1.0)
    for _ in range(50):
        ag2.rollout_costs = [0.0]
        ag2.end_episode(terminated=True)
        assert ag2.update_lambda_from_source() >= 0.0


# --------------------------------------------------------------------------- 5
@pytest.mark.needs_real_data
def test_init_policy_reproduces_grid_only_val():
    """Init policy mean IS Grid-Only; untrained agent produces Val 45,337.92 to the cent.
    Also verifies grid_only_action tracks the env's mapping rule."""
    from env.smartgrid_env import SmartGridEnv

    # Part A: grid_only_action derivation matches env for each mapping
    for mapping in ("linear", "relu", "deadzone"):
        a = grid_only_action(mapping)
        env = SmartGridEnv({"split": "val", "shed_mapping": mapping})
        env.reset(seed=42)
        _, _, _, _, info = env.step(a)
        assert max(info["shed_fractions"]) == pytest.approx(0.0, abs=1e-9), \
            f"{mapping}: {info['shed_fractions']}"

    # Part B: grid_only_action("linear") produces exact Grid-Only Val cost
    a_linear = grid_only_action("linear")
    assert np.allclose(a_linear[1:], -1.0) and a_linear[0] == 0.0
    env = SmartGridEnv({"split": "val", "shed_mapping": "linear"})
    total = 0.0
    for ep in range(26):
        obs, _ = env.reset(options={"start_idx": ep * 168})
        for _ in range(168):
            _, _, _, _, info = env.step(a_linear)
            total += (info["energy_cost"] + info["unserved_cost"]
                      + info["battery_wear_cost"])
    assert abs(total - 45337.92170812376) < 0.01, total

    # Part C: policy network mean == grid_only_action (zero-init weights)
    ag = PPOAgent(state_dim=17, action_dim=6, shed_mapping="linear")
    with th.no_grad():
        mu = ag.policy.forward(th.zeros(1, 17)).numpy().ravel()
    assert np.allclose(mu, a_linear, atol=1e-6), (mu, a_linear)


# --------------------------------------------------------------------------- 6
def test_no_test_split_access_under_agents():
    for root, _, files in os.walk("agents"):
        for f in files:
            if not f.endswith(".py"):
                continue
            src = open(os.path.join(root, f), encoding="utf-8", errors="replace").read()
            hits = [ln for ln in src.splitlines()
                    if '"test"' in ln or "'test'" in ln]
            assert not hits, f"{f} references the test split: {hits[:2]}"


def test_checkpoint_round_trip(tmp_path):
    """save_models/load_models restore weights, normalisers, and lambda."""
    agent = PPOAgent(state_dim=17, action_dim=6, lagrangian=True)
    with th.no_grad():
        for p in agent.policy.parameters():
            p.add_(0.01)
        for p in agent.value_net.parameters():
            p.add_(0.01)
    agent.obs_normalizer.mean = np.arange(17, dtype=np.float64) + 0.5
    agent.obs_normalizer.var = np.arange(17, dtype=np.float64) + 1.5
    agent.obs_normalizer.count = 11
    agent.ret_normalizer.mean = np.array(1.25)
    agent.ret_normalizer.var = np.array(2.5)
    agent.ret_normalizer.count = 7
    agent.lambda_ = 0.37

    path = tmp_path / "ckpt.pt"
    agent.save_models(str(path))

    restored = PPOAgent(state_dim=17, action_dim=6, lagrangian=True)
    restored.load_models(str(path))

    for key, value in agent.policy.state_dict().items():
        assert th.equal(value, restored.policy.state_dict()[key]), key
    for key, value in agent.value_net.state_dict().items():
        assert th.equal(value, restored.value_net.state_dict()[key]), key
    assert np.allclose(restored.obs_normalizer.mean, agent.obs_normalizer.mean)
    assert np.allclose(restored.obs_normalizer.var, agent.obs_normalizer.var)
    assert restored.obs_normalizer.count == agent.obs_normalizer.count
    assert restored.ret_normalizer.mean == pytest.approx(agent.ret_normalizer.mean)
    assert restored.ret_normalizer.var == pytest.approx(agent.ret_normalizer.var)
    assert restored.lambda_ == pytest.approx(agent.lambda_)

    step = th.zeros(1, 17)
    assert th.equal(agent.policy(step), restored.policy(step))
