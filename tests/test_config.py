import os
import sys

import pytest
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments.train_core import load_config_with_base

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_PATH = os.path.join(REPO_ROOT, "configs", "config.yaml")

_STORE = yaml.safe_load(open(CONFIG_PATH, encoding="utf-8")) or {}
EXPERIMENTS = _STORE.get("experiments") or {}
BASE = load_config_with_base("configs/config.yaml")

# The 12 runs the audit reads. A row disappearing breaks the audit table.
AUDIT_RUNS = [
    "control_plain", "control_lagrangian", "p1", "p1_p2", "p1_p2_p3",
    "m_relu", "m_relu_t", "m_relu_ta", "e_linear_ta",
    "b2_scaled", "b2_warm", "b2_long",
]


def test_base_yaml_documents_every_key_with_a_default():
    """Every knob lives in the defaults block with default == current live behaviour."""
    assert BASE["shed_mapping"] == "linear"
    assert BASE["reward_scale"] == 1.0
    assert BASE["reward_scale_mode"] == "fixed"
    assert BASE["reward_scale_explicit"] is False
    assert BASE["target_entropy"] == -3.0
    assert BASE["entropy_clamp"] is None
    assert BASE["alpha_init"] == 0.2
    assert BASE["warmup_mode"] == "uniform"
    assert BASE["torch_threads"] == 1


def test_unknown_key_raises():
    """A typo must not be silently dropped."""
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False,
                                     dir=os.path.dirname(CONFIG_PATH),
                                     encoding="utf-8") as f:
        f.write("target_entropy_mistype: -6.0\n")
        path = f.name
    try:
        with pytest.raises(KeyError, match="unknown config key"):
            load_config_with_base(path)
    finally:
        os.unlink(path)


def test_every_experiment_id_builds():
    """One-step smoke: each id in the table resolves onto the defaults."""
    assert EXPERIMENTS, "configs/config.yaml has no experiments"
    for exp_id in EXPERIMENTS:
        cfg = load_config_with_base(exp_id)
        assert set(cfg) == set(BASE), f"{exp_id} introduced or dropped a key"
        for k, v in (EXPERIMENTS[exp_id] or {}).items():
            assert cfg[k] == v, f"{exp_id}.{k} did not resolve"


def test_all_audit_runs_present():
    missing = [r for r in AUDIT_RUNS if r not in EXPERIMENTS]
    assert not missing, f"audit runs missing from configs/config.yaml: {missing}"


def test_every_override_key_actually_moves_the_resolved_config():
    """For each override key the effective value must differ from the base default.

    This is the generic version of the guard: a key that is declared but
    resolves to the default is a no-op probe.
    """
    for exp_id, ov in EXPERIMENTS.items():
        cfg = load_config_with_base(exp_id)
        for k, v in (ov or {}).items():
            assert cfg[k] == v, f"{exp_id}: {k} failed to apply"
            # `p1_p2_p3` is deliberately byte-identical to `p1_p2` - the
            # row is kept so the no-op stays reproducible, so exempt it here.
            if exp_id != "p1_p2_p3":
                assert cfg[k] != BASE[k], (
                    f"{exp_id}: override {k}={v!r} equals the base default — "
                    f"this probe would change nothing"
                )


@pytest.mark.needs_real_data
def test_shed_mapping_really_moves_the_env():
    """Regression: `relu` must not silently fall through to `linear`."""
    import numpy as np
    from env.smartgrid_env import SmartGridEnv

    shed = {}
    for mapping in ("linear", "relu", "deadzone"):
        env = SmartGridEnv({"split": "val", "shed_mapping": mapping})
        env.reset(seed=42)
        _, _, _, _, info = env.step(np.array([0.0, 0, 0, 0, 0, 0], dtype=np.float32))
        shed[mapping] = float(info["shed_fractions"][0])
    assert shed["linear"] == pytest.approx(0.5)      # s = (a+1)/2 at a=0
    assert shed["relu"] == pytest.approx(0.0)        # s = max(0, a) at a=0
    assert shed["deadzone"] == pytest.approx(shed["relu"])   # documented alias
    # and every `relu` experiment must reach the relu branch
    assert load_config_with_base("m_relu")["shed_mapping"] == "relu"


@pytest.mark.needs_real_data
def test_constant_action_is_grid_only_under_both_mappings():
    """Frozen baseline: a no-op action must cost exactly Grid-Only either way."""
    import numpy as np
    from env.smartgrid_env import SmartGridEnv

    for mapping in ("linear", "relu"):
        env = SmartGridEnv({"split": "val", "shed_mapping": mapping})
        total = 0.0
        for ep in range(26):
            env.reset(options={"start_idx": ep * 168})
            for _ in range(168):
                _, _, _, _, info = env.step(
                    np.array([0.0, -1, -1, -1, -1, -1], dtype=np.float32))
                total += (info["energy_cost"] + info["unserved_cost"]
                          + info["battery_wear_cost"])
        assert abs(total - 45337.92170812376) < 0.01, f"{mapping} broke Grid-Only"


def test_explicit_reward_scale_beats_the_computed_default():
    """Regression: `reward_scale_explicit: true` must win over `grid_only`."""
    cfg = load_config_with_base("b2_scaled")
    assert cfg["reward_scale_explicit"] is True
    assert cfg["reward_scale"] == 0.096879          # operator's value, not recomputed

    cfg_p1 = load_config_with_base("p1")
    assert cfg_p1["reward_scale_explicit"] is False  # recomputation allowed

    # the loader honours it: base default is 1.0, so the override really applied
    assert BASE["reward_scale"] == 1.0
    assert cfg["reward_scale"] != BASE["reward_scale"]


def test_p1_p2_p3_is_a_documented_no_op():
    """The defect must stay visible: the two rows resolve identically."""
    a = load_config_with_base("p1_p2")
    b = load_config_with_base("p1_p2_p3")
    assert a == b
    assert "stabilizers" not in BASE and "stabilizers" not in (EXPERIMENTS["p1_p2_p3"] or {})


def test_agent_helper_matches_training_kwargs():
    """`load_agent_for_method` must pass the SAME kwargs as the training call.

    If this drifts, a restored checkpoint silently replays a differently-configured
    agent than the one that trained — the worst possible silent failure. Compare the
    keyword sets of both constructor calls in the source, plus live attribute checks.
    """
    import ast
    import inspect
    from experiments import train_core as pp

    def _kwargs_of(fn, cls_name):
        tree = ast.parse(inspect.getsource(fn))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and getattr(node.func, "id", None) == cls_name):
                return {kw.arg for kw in node.keywords}
        raise AssertionError(f"{cls_name}(...) not found in {fn.__name__}")

    for cls in ("SACAgent", "SACLagrangianAgent"):
        train = _kwargs_of(pp.train_single_method, cls)
        helper = _kwargs_of(pp.load_agent_for_method, cls)
        assert helper == train, (
            f"{cls} kwargs differ from the training path: "
            f"helper-only={helper - train}, train-only={train - helper}")

    # live check: every hyperparameter reaches the agent
    for exp in ("control_plain", "b2_scaled"):
        cfg = load_config_with_base(exp)
        a = pp.load_agent_for_method(cfg["method"], cfg)
        assert isinstance(a, (pp.SACAgent, pp.SACLagrangianAgent))
        assert abs(float(a.alpha) - cfg["alpha_init"]) < 1e-6   # float32 storage
        assert a.gamma == cfg["gamma"] and a.tau == cfg["tau"]
        assert a.batch_size == cfg["batch_size"]
        if a.__class__.__name__ == "SACLagrangianAgent":
            assert a.cost_limit == cfg["cost_limit"]
            assert abs(a.lr_dual - cfg["lr_dual"]) < 1e-12


@pytest.mark.needs_real_data
def test_entropy_resolution_is_shared():
    """target_entropy / entropy_clamp / reward_scale must resolve identically."""
    from experiments.train_core import grid_only_reward_scale
    cfg = load_config_with_base("p1")
    assert cfg["reward_scale_mode"] == "grid_only" and cfg["reward_scale_explicit"] is False
    c = grid_only_reward_scale(cfg, cfg["shed_mapping"], method_name="t")
    # The documented value was 0.096879 from 104 episodes; the fix makes the
    # code actually reproduce it (this is the assertion that used to fail at 0.101380).
    assert abs(c - 0.096879) < 1e-4, c
    # explicit value is respected verbatim
    cfg2 = load_config_with_base("b2_scaled")
    assert grid_only_reward_scale(cfg2, "linear", method_name="t") == 0.096879
