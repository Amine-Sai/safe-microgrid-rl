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
    s_high[0, 0] = 0.50  # Battery SoC = 0.50
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
    s_low[0, 0] = 0.50  # Battery SoC = 0.50
    s_low[0, 1] = 100.0
    s_low[0, 2:7] = th.tensor([20.0, 20.0, 20.0, 20.0, 20.0])
    a_safe = th.tensor([[0.5]])
    a_safe_proj = layer(s_low, a_safe)
    assert th.allclose(a_safe_proj, a_safe), "Safe action should remain untouched"

    # 3. Gradient flow through PPODiffSafetyLayerAgent into policy network parameters under ACTIVE constraint (g > 0)
    agent = PPODiffSafetyLayerAgent(state_dim=DEFAULT_STATE_DIM, action_dim=1, p_max=p_max, p_cap=p_cap)
    s_batch = th.randn((8, DEFAULT_STATE_DIM))
    # Ensure state values have active constraint: P_net = 450 kW > p_cap = 368.15 kW (g > 0)
    s_batch[:, 0] = 0.50  # Battery SoC = 0.50
    s_batch[:, 1] = 0.0   # PV = 0.0 kW
    s_batch[:, 2:7] = 90.0  # Zone loads sum to 450 kW -> P_net = 450 kW > p_cap

    projected_actions = agent.policy.forward(s_batch)
    g_val = agent.safety_layer.constraint_val(s_batch, projected_actions)
    assert th.all(g_val <= 1e-4), "Projected actions must satisfy constraint"

    loss = (projected_actions - (-1.0)).pow(2).mean()

    agent.policy_optimizer.zero_grad()
    loss.backward()

    grads_found = False
    for name, p in agent.policy.named_parameters():
        if p.grad is not None:
            assert th.all(th.isfinite(p.grad)), f"Gradient for {name} contains NaN/Inf"
            if p.grad.abs().sum() > 0:
                grads_found = True

    assert grads_found, "Gradients failed to flow back into policy parameters when constraint is ACTIVE (g > 0)"

    # 4. SafePolicyNetwork attachment and checkpoint restoration
    from agents.safety_layer import SafePolicyNetwork
    assert isinstance(agent.policy, SafePolicyNetwork)
    assert agent.policy.safety_layer is agent.safety_layer


def test_diff_safety_layer_soc_bounds(project_constants):
    """Prove requirement (b): hard SoC min/max bound enforcement inside DifferentiableSafetyLayer."""
    p_max = 250.0
    p_cap = project_constants.get("grid_import_cap_kw", 368.15)
    layer = DifferentiableSafetyLayer(p_max=p_max, p_cap=p_cap)

    # 1. Empty battery at soc_min (0.10): discharge command must be clamped to >= 0.0
    s_empty = th.zeros((1, DEFAULT_STATE_DIM))
    s_empty[0, 0] = layer.soc_min  # 0.10
    s_empty[0, 1] = 100.0
    s_empty[0, 2:7] = 20.0  # P_net = 0 kW (no grid cap pressure)
    a_dis = th.tensor([[-0.8]])
    a_proj_empty = layer(s_empty, a_dis)
    assert np.isclose(a_proj_empty.item(), 0.0, atol=1e-5), (
        f"At soc_min, battery cannot discharge; expected 0.0, got {a_proj_empty.item()}"
    )

    # 2. Full battery at soc_max (0.90): charging command must be clamped to <= 0.0
    s_full = th.zeros((1, DEFAULT_STATE_DIM))
    s_full[0, 0] = layer.soc_max  # 0.90
    s_full[0, 1] = 100.0
    s_full[0, 2:7] = 20.0  # P_net = 0 kW
    a_ch = th.tensor([[0.8]])
    a_proj_full = layer(s_full, a_ch)
    assert np.isclose(a_proj_full.item(), 0.0, atol=1e-5), (
        f"At soc_max, battery cannot charge; expected 0.0, got {a_proj_full.item()}"
    )

    # 3. Intermediate SoC: check exact analytical bound clamping
    # At soc = 0.20, delta_soc = 0.10, energy = 50 kWh, max dis power = 50 * 0.95 / 1.0 = 47.5 kW
    # a0_min = -47.5 / 250 = -0.19
    s_inter = th.zeros((1, DEFAULT_STATE_DIM))
    s_inter[0, 0] = 0.20
    s_inter[0, 1] = 50.0
    s_inter[0, 2:7] = 10.0  # P_net = 0 kW
    a_excess_dis = th.tensor([[-0.9]])
    a_proj_inter = layer(s_inter, a_excess_dis)
    expected_min = -(0.20 - layer.soc_min) * layer.capacity_kwh * (layer.eta_discharge / layer.dt_hours) / layer.p_max
    assert np.isclose(a_proj_inter.item(), expected_min, atol=1e-4), (
        f"Expected clamping at {expected_min}, got {a_proj_inter.item()}"
    )


def test_diff_safety_layer_multidim_qp(project_constants):
    """Prove requirement (c): multi-dimensional (action_dim=6) joint KKT QP projection
    with analytical non-zero Jacobian across battery + shedding actions."""
    p_max = 250.0
    p_cap = project_constants.get("grid_import_cap_kw", 368.15)
    layer = DifferentiableSafetyLayer(p_max=p_max, p_cap=p_cap)

    # High net load state: PV=0, zone loads = [90, 90, 90, 90, 90] kW -> P_net = 450 kW > P_cap
    s = th.zeros((1, DEFAULT_STATE_DIM))
    s[0, 0] = 0.50  # Battery SoC = 0.50
    s[0, 1] = 0.0   # PV = 0.0 kW
    s[0, 2:7] = 90.0  # 5 zones of 90 kW each

    # 6D action: battery charging (+0.2 = +50 kW), low shedding (-0.5 = 25% shed)
    a = th.tensor([[0.2, -0.5, -0.5, -0.5, -0.5, -0.5]], requires_grad=True)

    g_before = layer.constraint_val(s, a)
    assert g_before.item() > 0.0, f"Initial action must violate constraint, got g={g_before.item()}"

    # Project via joint KKT QP
    a_proj = layer(s, a)

    # 1. Constraint satisfaction
    g_after = layer.constraint_val(s, a_proj)
    assert g_after.item() <= 1e-4, f"Projected action must satisfy constraint g <= 0, got g={g_after.item()}"

    # 2. Joint cooperation: battery charging decreased and load shedding increased
    assert a_proj[0, 0].item() < a[0, 0].item(), "Battery command must decrease (charge less or discharge)"
    for z in range(1, 6):
        assert a_proj[0, z].item() >= a[0, z].item(), f"Zone {z} shedding command must increase"

    # 3. Analytical non-zero Jacobian across battery AND shedding actions
    def qp_func(act_vec):
        return layer(s, act_vec.unsqueeze(0)).squeeze(0)

    a_1d = th.tensor([0.2, -0.5, -0.5, -0.5, -0.5, -0.5])
    J = th.autograd.functional.jacobian(qp_func, a_1d)

    assert J.shape == (6, 6)
    rank = th.linalg.matrix_rank(J)
    assert rank.item() >= 5, f"Jacobian rank must be at least m-1=5, got {rank.item()}"
    assert th.all(J.abs() > 0.0), f"All elements in analytical QP Jacobian must be non-zero, got min={J.abs().min()}"


def test_safe_policy_network_log_prob_consistency(project_constants):
    """Prove SafePolicyNetwork maintains exact Gaussian log_prob consistency and avoids double-projection bias."""
    p_max = 250.0
    p_cap = project_constants.get("grid_import_cap_kw", 368.15)
    agent = PPODiffSafetyLayerAgent(
        state_dim=DEFAULT_STATE_DIM,
        action_dim=1,
        p_max=p_max,
        p_cap=p_cap,
        store_raw_action=True,
    )

    s = np.zeros(DEFAULT_STATE_DIM, dtype=np.float32)
    s[0] = 0.50
    s[1] = 0.0
    s[2:7] = 90.0  # P_net = 450 kW > p_cap

    # 1. SafePolicyNetwork distribution uses unprojected raw mean for Gaussian log_prob
    agent.obs_normalizer.update(s[None, :])
    s_tensor = agent.to_tensor(agent.obs_normalizer.normalize(s[None, :]))
    raw_mu = agent.policy.forward_raw(s_tensor)
    proj_mu = agent.policy.forward(s_tensor)
    dist = agent.policy.distribution(s_tensor)

    assert th.allclose(dist.loc, raw_mu), "Policy distribution mean must equal raw unprojected mean"
    assert not th.allclose(raw_mu, proj_mu), "When constraint is active, raw_mu and proj_mu must differ"

    # 2. Select action returns projected safe action while buffering raw action
    action, logp, value, _ = agent.select_action(s, evaluate=False)
    assert agent.last_raw_action is not None
    # Action executed in env is safe
    s_norm = agent.to_tensor(agent.obs_normalizer.normalize(s[None, :]))
    a_th = th.as_tensor(action[None, :], dtype=th.float32)
    g_val = agent.safety_layer.constraint_val(s_norm, a_th)
    assert g_val.item() <= 1e-4, f"Action returned by select_action must be safe, got g={g_val.item()}"

    # 3. Store transition stores raw action and matches initial log_prob
    agent.store_transition(
        state=s,
        action=action,
        reward=-1.0,
        cost=0.0,
        value=value,
        cost_value=0.0,
        log_prob=logp,
        terminated=False,
        truncated=False,
    )
    assert np.allclose(agent.buffer.actions[0], agent.last_raw_action), "Stored action must be raw action"
    batch = agent.finish_path(last_value=value)
    assert batch is not None


def test_safe_policy_distribution_backward_calls_safety_layer(project_constants):
    """T1: SafePolicyNetwork.distribution() preserves forward_raw values and propagates STE gradients via safety_layer."""
    from agents.ppo import initial_grid_only_action_mean
    from agents.safety_layer import SafePolicyNetwork

    p_max = 250.0
    p_cap = project_constants.get("grid_import_cap_kw", 368.15)
    layer = DifferentiableSafetyLayer(p_max=p_max, p_cap=p_cap, surrogate_grad_scale=0.1)
    init_mean = initial_grid_only_action_mean("linear", 6)
    net = SafePolicyNetwork(state_dim=DEFAULT_STATE_DIM, action_dim=6, safety_layer=layer, init_mean=init_mean)

    calls = [0]
    orig_forward = layer.forward

    def spy_forward(state: th.Tensor, action: th.Tensor) -> th.Tensor:
        calls[0] += 1
        return orig_forward(state, action)

    layer.forward = spy_forward  # type: ignore[method-assign]

    s = th.zeros((4, DEFAULT_STATE_DIM), dtype=th.float32)
    s[:, 0] = 0.50
    s[:, 1] = 0.0
    s[:, 2:7] = 100.0  # P_net = 500 kW > p_cap -> active constraint g > 0

    with th.enable_grad():
        raw_mu = net.forward_raw(s)
        assert layer.constraint_val(s, raw_mu).min().item() > 0.0
        dist = net.distribution(s)
        assert calls[0] == 1, f"Expected safety_layer.forward to be called once, got {calls[0]}"
        assert th.allclose(dist.mean, raw_mu), "dist.mean must match forward_raw(s)"

        u = (dist.mean + 0.25).detach()
        net.zero_grad()
        (-dist.log_prob(u).sum()).backward()
        assert net.mu.weight.grad is not None
        assert net.mu.weight.grad.abs().sum().item() > 0.0, "Expected non-zero mu.weight.grad under active constraint"


def test_safety_layer_6d_industrial_piecewise_consistency(project_constants):
    """T1: 6-D DifferentiableSafetyLayer industrial shedding matches SmartGridEnv below threshold."""
    from env.smartgrid_env import shed_action_to_fractions

    p_max = 250.0
    p_cap = project_constants.get("grid_import_cap_kw", 368.153918)
    layer = DifferentiableSafetyLayer(
        p_max=p_max,
        p_cap=p_cap,
        motor_min_pct=70.0,
        industrial_shutdown_threshold=0.90,
    )

    zone_loads = np.array([100.0, 90.0, 80.0, 50.0, 150.0], dtype=np.float32)
    s = th.zeros((1, DEFAULT_STATE_DIM), dtype=th.float32)
    s[0, 0] = 0.50
    s[0, 1] = 0.0
    s[0, 2:7] = th.from_numpy(zone_loads)

    for a_ind in (-1.0, -0.5, 0.0, 0.5, 0.8):
        shed_act = np.array([-1.0, -0.5, 0.0, 0.2, a_ind], dtype=np.float32)
        act = th.tensor([[0.0, *shed_act]], dtype=th.float32)

        fracs_env = shed_action_to_fractions(
            shed_act,
            shed_mapping="linear",
            motor_min_pct=70.0,
            industrial_shutdown_threshold=0.90,
        )
        fracs_layer = layer._zone_shed_fractions(act[:, 1:])[0].detach().numpy()
        assert np.allclose(fracs_layer, fracs_env, atol=1e-6), f"Mismatch for a_ind={a_ind}: {fracs_layer} vs {fracs_env}"

        g_layer = float(layer.constraint_val(s, act).item())
        expected_g = float(zone_loads.sum()) - float(np.sum(zone_loads * fracs_env)) - p_cap
        assert abs(g_layer - expected_g) <= 1e-4, f"g mismatch for a_ind={a_ind}: {g_layer} vs {expected_g}"


