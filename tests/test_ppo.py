import os
import sys

import numpy as np
import pytest
import torch as th

from agents.ppo import PPOAgent, compute_gae, grid_only_action
from tests.conftest import load_frozen_baselines, load_project_constants

DEFAULT_STATE_DIM = 17
DEFAULT_ACTION_DIM = 6
HOURS_PER_WEEK = 168
VAL_WEEKS = 26


def _gae_reference(rewards, values, dones, truncated, last_value, gamma, lam):
    """Reference numpy GAE implementation."""
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
    """Verify GAE calculation against reference numpy implementation."""
    r = np.array([1.0, 0.0, 2.0, 1.0, 3.0])
    v = np.array([1.0, 2.0, 1.0, 3.0, 2.0])
    d = np.array([0, 0, 0, 1, 0], dtype=bool)      # terminal at step 3
    tr = np.array([0, 1, 0, 0, 0], dtype=bool)     # truncation at step 1
    expected = np.array([0.75, -1.0, 2.0, -2.0, 3.0])

    adv = compute_gae(r, v, d, tr, last_value=4.0, gamma=0.5, lam=0.5)[0]
    ref = _gae_reference(r, v, d, tr, 4.0, 0.5, 0.5)

    assert np.allclose(adv, expected, atol=1e-10), f"compute_gae: {adv}"
    assert np.allclose(ref, expected, atol=1e-10), f"reference: {ref}"
    assert np.allclose(adv, ref, atol=1e-10)


def test_truncation_bootstrapping():
    """A time-limit end must bootstrap V(s_T), a terminal end must not."""
    r = np.array([1.0, 1.0, 1.0])
    v = np.array([0.5, 0.5, 0.5])
    d_terminal = np.array([0, 0, 1], dtype=bool)
    d_trunc = np.array([0, 0, 0], dtype=bool)
    trunc_flag = np.array([0, 0, 1], dtype=bool)
    last_val = 100.0

    adv_trunc = compute_gae(r, v, d_trunc, trunc_flag, last_value=last_val,
                            gamma=0.99, lam=0.95)[0]
    adv_term = compute_gae(r, v, d_terminal, trunc_flag, last_value=last_val,
                           gamma=0.99, lam=0.95)[0]
    # the terminal episode gets NO bootstrap: its last advantage must be far smaller
    assert adv_trunc[-1] > adv_term[-1] + 50.0
    assert adv_term[-1] == pytest.approx(1.0 - 0.5, abs=1e-9)


def test_seed_determinism(project_constants):
    """Two full seeded runs (rollout + finish_path + update) produce identical results."""
    d_main = project_constants.get("d_main_kw_per_step", 0.05)
    results = []
    for _ in range(2):
        th.manual_seed(42)
        np.random.seed(42)
        ag = PPOAgent(state_dim=4, action_dim=2, lagrangian=True, cost_limit=d_main)

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


def test_lambda_nonnegative_and_rises_when_cost_exceeds_d(project_constants):
    d_main = project_constants.get("d_main_kw_per_step", 0.05)
    ag = PPOAgent(state_dim=4, action_dim=2, lagrangian=True, cost_limit=d_main,
                  lambda_lr_mult=0.5)
    lam0 = ag.lambda_
    assert lam0 >= 0.0

    high_cost_per_step = d_main * 100.0  # well above d_main to force multiplier rise
    for _ in range(10):
        ag.rollout_costs = [high_cost_per_step] * 3
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


@pytest.mark.needs_real_data
def test_init_policy_reproduces_grid_only_val(frozen_baselines):
    """Init policy mean IS Grid-Only; untrained agent produces Val cost matching frozen baseline.
    Also verifies grid_only_action tracks the env's mapping rule."""
    from env.smartgrid_env import SmartGridEnv

    # Part A: grid_only_action derivation matches env for each mapping
    for mapping in ("linear", "relu", "deadzone"):
        a = grid_only_action(mapping)
        env = SmartGridEnv({"split": "val", "shed_mapping": mapping})
        env.reset(seed=42)
        _, _, _, _, info = env.step(a)
        max_shed = max(info["shed_fractions"])
        assert max_shed == pytest.approx(0.0, abs=1e-9), \
            f"Mapping {mapping} produced non-zero shed fraction: {info['shed_fractions']}"

    # Part B: grid_only_action("linear") produces exact Grid-Only Val cost
    a_linear = grid_only_action("linear")
    assert np.allclose(a_linear[1:], -1.0) and a_linear[0] == 0.0
    env = SmartGridEnv({"split": "val", "shed_mapping": "linear"})
    total_val_cost = 0.0
    for ep in range(VAL_WEEKS):
        obs, _ = env.reset(options={"start_idx": ep * HOURS_PER_WEEK})
        for _ in range(HOURS_PER_WEEK):
            _, _, _, _, info = env.step(a_linear)
            total_val_cost += (info["energy_cost"] + info["unserved_cost"]
                               + info["battery_wear_cost"])
    expected_grid_only_cost = frozen_baselines["Grid-Only"]["val_cost_eur"]
    assert abs(total_val_cost - expected_grid_only_cost) < 0.05, (
        f"Simulated Grid-Only val cost ({total_val_cost:.2f} EUR) did not match "
        f"frozen baseline ({expected_grid_only_cost:.2f} EUR)"
    )

    # Part C: policy network mean == grid_only_action (zero-init weights)
    ag = PPOAgent(state_dim=DEFAULT_STATE_DIM, action_dim=DEFAULT_ACTION_DIM, shed_mapping="linear")
    with th.no_grad():
        mu = ag.policy.forward(th.zeros(1, DEFAULT_STATE_DIM)).numpy().ravel()
    assert np.allclose(mu, a_linear, atol=1e-6), f"Initial policy {mu} != grid_only_action {a_linear}"


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
    agent = PPOAgent(state_dim=DEFAULT_STATE_DIM, action_dim=DEFAULT_ACTION_DIM, lagrangian=True)
    with th.no_grad():
        for p in agent.policy.parameters():
            p.add_(0.01)
        for p in agent.value_net.parameters():
            p.add_(0.01)
    agent.obs_normalizer.mean = np.arange(DEFAULT_STATE_DIM, dtype=np.float64) + 0.5
    agent.obs_normalizer.var = np.arange(DEFAULT_STATE_DIM, dtype=np.float64) + 1.5
    agent.obs_normalizer.count = 11
    agent.ret_normalizer.mean = np.array(1.25)
    agent.ret_normalizer.var = np.array(2.5)
    agent.ret_normalizer.count = 7
    agent.lambda_ = 0.37

    path = tmp_path / "ckpt.pt"
    agent.save_models(str(path))

    restored = PPOAgent(state_dim=DEFAULT_STATE_DIM, action_dim=DEFAULT_ACTION_DIM, lagrangian=True)
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

    step = th.zeros(1, DEFAULT_STATE_DIM)
    assert th.equal(agent.policy(step), restored.policy(step))


def test_ppo_subclass_hierarchy():
    """Verify distinct subclass instantiation and initial policy mean behaviour."""
    from agents.ppo import (
        PPOAgent,
        PPOLagrangianAgent,
        PPOResidualAgent,
        PPOShieldedAgent,
        PPOPotentialShapingAgent,
        PPOLagrangianPotentialShapingAgent,
    )

    base = PPOAgent(state_dim=DEFAULT_STATE_DIM, action_dim=DEFAULT_ACTION_DIM)
    assert isinstance(base, PPOAgent)
    assert not base.lagrangian
    assert not base.pbrs

    lag = PPOLagrangianAgent(state_dim=DEFAULT_STATE_DIM, action_dim=DEFAULT_ACTION_DIM)
    assert isinstance(lag, PPOAgent)
    assert isinstance(lag, PPOLagrangianAgent)
    assert lag.lagrangian
    assert lag.cost_net is not None

    res = PPOResidualAgent(state_dim=DEFAULT_STATE_DIM, action_dim=DEFAULT_ACTION_DIM)
    assert isinstance(res, PPOLagrangianAgent)
    assert res.residual_policy is not None
    # Residual policy mean starts at 0 (zero correction)
    with th.no_grad():
        mu_res = res.policy.forward(th.zeros(1, DEFAULT_STATE_DIM)).numpy().ravel()
    assert np.allclose(mu_res, 0.0, atol=1e-6)

    shi = PPOShieldedAgent(state_dim=DEFAULT_STATE_DIM)
    assert isinstance(shi, PPOAgent)
    assert shi.action_dim == 1

    ps = PPOPotentialShapingAgent(state_dim=DEFAULT_STATE_DIM, action_dim=DEFAULT_ACTION_DIM)
    assert isinstance(ps, PPOAgent)
    assert ps.pbrs

    lag_ps = PPOLagrangianPotentialShapingAgent(state_dim=DEFAULT_STATE_DIM, action_dim=DEFAULT_ACTION_DIM)
    assert isinstance(lag_ps, PPOLagrangianAgent)
    assert isinstance(lag_ps, PPOPotentialShapingAgent)
    assert lag_ps.lagrangian and lag_ps.pbrs

    from agents.ppo import PPOAuxShieldLossAgent, PPODiffSafetyLayerAgent
    aux = PPOAgent(state_dim=DEFAULT_STATE_DIM, action_dim=1, method="ppo_shield_aux", aux_shield_coef=0.5)
    assert isinstance(aux, PPOAuxShieldLossAgent)
    assert aux.aux_shield_coef == 0.5

    diff = PPOAgent(state_dim=DEFAULT_STATE_DIM, action_dim=1, method="ppo_diff_shield")
    assert isinstance(diff, PPODiffSafetyLayerAgent)
    from agents.safety_layer import SafePolicyNetwork
    assert isinstance(diff.policy, SafePolicyNetwork)


def test_raw_action_storage_prevents_off_policy_start():
    """Caveat 1 & 2: Storing raw policy actions preserves ratio_off_at_start == 0."""
    from agents.ppo import PPOAgent, PPOResidualAgent

    for agent in (
        PPOAgent(state_dim=DEFAULT_STATE_DIM, action_dim=DEFAULT_ACTION_DIM, store_raw_action=True),
        PPOResidualAgent(state_dim=DEFAULT_STATE_DIM, action_dim=DEFAULT_ACTION_DIM, store_raw_action=True),
    ):
        states = np.random.randn(10, DEFAULT_STATE_DIM).astype(np.float32)
        for s in states:
            a, lp, v, cv = agent.select_action(s)
            agent.store_transition(s, a, 1.0, 0.0, v, cv, lp, False, False)
        batch = agent.finish_path(0.0, 0.0)
        diag = agent.update(batch)
        assert diag["ratio_off_at_start"] == pytest.approx(0.0, abs=1e-6)


def test_explained_variance_diagnostic_consistency():
    """Caveat 5: Explained variance uses denormalized value net predictions."""
    from agents.ppo import PPOAgent

    agent = PPOAgent(state_dim=4, action_dim=2, use_minibatches=False)
    states = np.random.randn(20, 4).astype(np.float32)
    for s in states:
        a, lp, v, cv = agent.select_action(s)
        agent.store_transition(s, a, 1.0, 0.0, v, cv, lp, False, False)
    batch = agent.finish_path(0.0, 0.0)
    diag = agent.update(batch)
    assert np.isfinite(diag["explained_variance"])


def test_from_checkpoint_restores_variant_and_residual(tmp_path):
    """Caveat 7: PPOAgent.from_checkpoint restores subclass, residual policy, and weights."""
    from agents.ppo import PPOAgent, PPOResidualAgent

    res_agent = PPOResidualAgent(state_dim=DEFAULT_STATE_DIM, action_dim=DEFAULT_ACTION_DIM)
    ckpt_path = tmp_path / "residual_ckpt.pt"
    res_agent.save_models(str(ckpt_path))

    restored = PPOAgent.from_checkpoint(str(ckpt_path))
    assert isinstance(restored, PPOResidualAgent)
    assert restored.residual_policy is not None
    assert restored.lagrangian is True

    test_obs = np.ones(DEFAULT_STATE_DIM, dtype=np.float32) * 0.5
    a_orig = res_agent.select_action(test_obs, evaluate=True)[0]
    a_rest = restored.select_action(test_obs, evaluate=True)[0]
    assert np.allclose(a_orig, a_rest, atol=1e-6)

    # Test restoring PPOAuxShieldLossAgent
    from agents.ppo import PPOAuxShieldLossAgent, PPODiffSafetyLayerAgent
    aux_agent = PPOAuxShieldLossAgent(state_dim=DEFAULT_STATE_DIM, action_dim=1, aux_shield_coef=0.75)
    ckpt_aux = tmp_path / "aux_ckpt.pt"
    aux_agent.save_models(str(ckpt_aux))
    restored_aux = PPOAgent.from_checkpoint(str(ckpt_aux))
    assert isinstance(restored_aux, PPOAuxShieldLossAgent)
    assert restored_aux.aux_shield_coef == 0.75

    # Test restoring PPODiffSafetyLayerAgent
    diff_agent = PPODiffSafetyLayerAgent(state_dim=DEFAULT_STATE_DIM, action_dim=1, p_max=200.0, p_cap=350.0)
    ckpt_diff = tmp_path / "diff_ckpt.pt"
    diff_agent.save_models(str(ckpt_diff))
    restored_diff = PPOAgent.from_checkpoint(str(ckpt_diff))
    assert isinstance(restored_diff, PPODiffSafetyLayerAgent)
    assert restored_diff.p_max == 200.0
    assert restored_diff.p_cap == 350.0
