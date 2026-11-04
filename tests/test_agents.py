import numpy as np
import pytest
import torch as th

from agents.ppo import PPOAuxShieldLossAgent, PPODiffSafetyLayerAgent
from agents.running_stat import RunningMeanStd
from agents.safety_layer import DifferentiableSafetyLayer
from agents.shaping import PotentialBasedShaping
from tests.conftest import load_project_constants

DEFAULT_STATE_DIM = 17


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
    gamma = 0.99
    kappa = 2.0
    shaping = PotentialBasedShaping(gamma=gamma, kappa=kappa)
    s = np.zeros(DEFAULT_STATE_DIM)
    s[0] = 0.5  # SoC = 0.5
    s_prime = np.zeros(DEFAULT_STATE_DIM)
    s_prime[0] = 0.6  # SoC = 0.6

    f = shaping.shape(s, s_prime)
    expected = gamma * (kappa * s_prime[0]) - (kappa * s[0])
    assert np.isclose(f, expected, atol=1e-5)


def test_potential_shaping_episode_telescoping_sum(project_constants):
    """Requirement: with gamma = 1, shaping terms sum over an episode to Phi(s_T) - Phi(s_0)."""
    gamma = 1.0
    kappa = 1.5
    shaping = PotentialBasedShaping(gamma=gamma, kappa=kappa)
    T = 168  # 1 week of hourly steps
    # Generate an arbitrary trajectory of SoC states within physical bounds [soc_min, soc_max]
    soc_min = project_constants.get("battery_soc_min", 0.10)
    soc_max = 0.90  # upper SoC bound
    rng = np.random.RandomState(42)
    soc_trajectory = rng.uniform(soc_min, soc_max, size=T + 1)

    terms = []
    for t in range(T):
        s_t = np.zeros(DEFAULT_STATE_DIM)
        s_t[0] = soc_trajectory[t]
        s_next = np.zeros(DEFAULT_STATE_DIM)
        s_next[0] = soc_trajectory[t + 1]

        f_t = shaping.shape(s_t, s_next)
        terms.append(f_t)

    sum_terms = np.sum(terms)
    s_0 = np.zeros(DEFAULT_STATE_DIM)
    s_0[0] = soc_trajectory[0]
    s_T = np.zeros(DEFAULT_STATE_DIM)
    s_T[0] = soc_trajectory[T]
    phi_0 = shaping.phi(s_0)
    phi_T = shaping.phi(s_T)
    expected_sum = phi_T - phi_0

    assert np.isclose(sum_terms, expected_sum, atol=1e-6), (
        f"Sum {sum_terms} != Phi(s_T) - Phi(s_0) ({expected_sum})"
    )


def test_ppo_aux_shield_loss():
    """Verify that PPOAuxShieldLossAgent computes auxiliary loss, backward passes without NaN, and updates policy weights."""
    agent = PPOAuxShieldLossAgent(state_dim=DEFAULT_STATE_DIM, action_dim=1, aux_shield_coef=0.5)

    # 1. Auxiliary loss computation and backward pass
    s_t = th.randn(4, DEFAULT_STATE_DIM)
    weights = th.tensor([[1.0], [0.0], [1.0], [0.0]])
    loss_aux = agent.compute_aux_loss(s_t, weights)
    assert loss_aux.item() >= 0.0
    assert th.isfinite(loss_aux)

    agent.policy_optimizer.zero_grad()
    loss_aux.backward()
    for name, p in agent.policy.named_parameters():
        if "mu" in name or "fc" in name:
            assert p.grad is not None, f"Parameter {name} has no gradient"
            assert th.all(th.isfinite(p.grad)), f"Parameter {name} gradient contains NaN/Inf"

    # 2. End-to-end rollout buffer storage and policy weight update
    agent.policy_optimizer.zero_grad()
    rng = np.random.RandomState(42)
    for i in range(30):
        s = rng.randn(DEFAULT_STATE_DIM).astype(np.float32)
        a = np.array([0.5], dtype=np.float32)
        shed_kw = 50.0 if i % 2 == 0 else 0.0
        agent.store_transition(
            state=s,
            action=a,
            reward=-1.0,
            cost=0.0,
            value=0.0,
            cost_value=0.0,
            log_prob=-0.5,
            terminated=False,
            truncated=False,
            shield_shed_kw=shed_kw,
        )

    batch = agent.finish_path(last_value=0.0)
    assert "shield_shed_kw" in batch
    assert np.any(batch["shield_shed_kw"] > 0)

    params_before = [p.clone() for p in agent.policy.parameters()]
    metrics = agent.update(batch)
    assert isinstance(metrics, dict)
    assert "pg_loss" in metrics

    # Verify parameters updated without NaN
    updated = False
    for p_new, p_old in zip(agent.policy.parameters(), params_before):
        assert th.all(th.isfinite(p_new)), "Policy parameters contain NaN/Inf after update"
        if not th.allclose(p_new, p_old):
            updated = True
    assert updated, "Policy weights were not updated during agent.update()"

    # 3. Minibatch update with N=150 > batch_size=64 to verify weight slicing
    agent_mb = PPOAuxShieldLossAgent(
        state_dim=DEFAULT_STATE_DIM,
        action_dim=1,
        aux_shield_coef=0.5,
        use_minibatches=True,
        batch_size=64,
        value_epochs=2,
    )
    for i in range(150):
        s = rng.randn(DEFAULT_STATE_DIM).astype(np.float32)
        a = np.array([0.2], dtype=np.float32)
        shed_kw = 25.0 if i % 3 == 0 else 0.0
        agent_mb.store_transition(
            state=s,
            action=a,
            reward=-0.5,
            cost=0.0,
            value=0.0,
            cost_value=0.0,
            log_prob=-0.2,
            terminated=False,
            truncated=(i == 149),
            shield_shed_kw=shed_kw,
        )
    batch_mb = agent_mb.finish_path(last_value=0.0)
    assert len(batch_mb["states"]) == 150
    metrics_mb = agent_mb.update(batch_mb)
    assert isinstance(metrics_mb, dict)
    assert "pg_loss" in metrics_mb


def test_ppo_diff_safety_layer(project_constants):
    """Verify that DifferentiableSafetyLayer projects unsafe charging actions to safe discharge,
    and that gradients flow back into policy parameters."""
    p_max = 250.0  # Battery maximum power (kW)
    p_cap = project_constants.get("grid_import_cap_kw", 368.15)
    layer = DifferentiableSafetyLayer(p_max=p_max, p_cap=p_cap)

    # 1. High net load: PV=0, zone loads sum to 450 kW -> P_net = 450 kW > P_cap (368.15 kW)
    s_high = th.zeros((1, DEFAULT_STATE_DIM))
    s_high[0, 1] = 0.0  # PV
    zone_loads = th.tensor([100.0, 100.0, 100.0, 50.0, 100.0])
    s_high[0, 2:7] = zone_loads
    p_net = float(zone_loads.sum())  # 450.0 kW

    # Unsafe charging command a_0 = 0.5 (+125 kW charging)
    a_unsafe = th.tensor([[0.5]])
    a_proj = layer(s_high, a_unsafe)

    # Expected analytical projection (Dalal et al., 2018):
    # g = p_net + p_max * a_val - p_cap > 0
    # correction = g / p_max
    # a_tilde = a_val - correction
    a_val = a_unsafe.item()
    constraint_violation = p_net + p_max * a_val - p_cap
    expected_a_tilde = a_val - (constraint_violation / p_max)
    assert a_proj.item() < 0.0, f"Unsafe charging action must be projected to discharge, got {a_proj.item()}"
    assert np.isclose(a_proj.item(), expected_a_tilde, atol=1e-4)

    # Resulting net grid import satisfies P_cap exactly
    resulting_import = p_net + p_max * a_proj.item()
    assert np.isclose(resulting_import, p_cap, atol=1e-4)

    # 2. Already safe state & action: PV=100 kW, loads=100 kW -> P_net = 0 kW
    s_low = th.zeros((1, DEFAULT_STATE_DIM))
    s_low[0, 1] = 100.0
    s_low[0, 2:7] = th.tensor([20.0, 20.0, 20.0, 20.0, 20.0])
    a_safe = th.tensor([[0.5]])
    a_safe_proj = layer(s_low, a_safe)
    assert th.allclose(a_safe_proj, a_safe), "Safe action should remain untouched"

    # 3. Gradient flow through PPODiffSafetyLayerAgent into policy network parameters
    agent = PPODiffSafetyLayerAgent(state_dim=DEFAULT_STATE_DIM, action_dim=1, p_max=p_max, p_cap=p_cap)
    s_batch = th.randn((8, DEFAULT_STATE_DIM))
    # Ensure state values have non-zero PV and load
    s_batch[:, 1] = 20.0
    s_batch[:, 2:7] = 60.0  # P_net = 280 kW

    projected_actions = agent.policy.forward(s_batch)
    loss = (projected_actions - (-1.0)).pow(2).mean()

    agent.policy_optimizer.zero_grad()
    loss.backward()

    grads_found = False
    for name, p in agent.policy.named_parameters():
        if p.grad is not None:
            assert th.all(th.isfinite(p.grad)), f"Gradient for {name} contains NaN/Inf"
            if p.grad.abs().sum() > 0:
                grads_found = True

    assert grads_found, "Gradients failed to flow back into policy parameters through DifferentiableSafetyLayer"

    # 4. SafePolicyNetwork attachment and checkpoint restoration
    from agents.safety_layer import SafePolicyNetwork
    from agents.ppo import PPOAgent
    assert isinstance(agent.policy, SafePolicyNetwork)
    assert agent.policy.safety_layer is agent.safety_layer
