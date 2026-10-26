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

def test_base_yaml_documents_every_key_with_a_default():
    """Every knob lives in the defaults block with default == current live behaviour."""
    assert BASE["method"] == "ppo"
    assert BASE["shed_mapping"] == "linear"
    assert BASE["reward_scale"] == 1.0
    assert BASE["reward_scale_mode"] == "fixed"
    assert BASE["reward_scale_explicit"] is False
    assert BASE["ppo_clip_ratio"] == 0.2
    assert BASE["rollout_steps"] == 2048
    assert BASE["torch_threads"] == 1


def test_unknown_key_raises():
    """A typo must not be silently dropped."""
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False,
                                     dir=os.path.dirname(CONFIG_PATH),
                                     encoding="utf-8") as f:
        f.write("ppo_clip_ratio_mistype: 0.3\n")
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


def test_every_override_key_actually_moves_the_resolved_config():
    """Every override must actually change the resolved config."""
    for exp_id, ov in EXPERIMENTS.items():
        cfg = load_config_with_base(exp_id)
        for k, v in (ov or {}).items():
            assert cfg[k] == v, f"{exp_id}: {k} failed to apply"
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
    import tempfile
    from experiments.train_core import grid_only_reward_scale

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False,
                                     dir=os.path.dirname(CONFIG_PATH),
                                     encoding="utf-8") as f:
        f.write("reward_scale: 0.096879\n"
                "reward_scale_mode: grid_only\n"
                "reward_scale_explicit: true\n")
        path = f.name
    try:
        cfg = load_config_with_base(path)
    finally:
        os.unlink(path)

    assert cfg["reward_scale_explicit"] is True
    assert cfg["reward_scale"] == 0.096879          # operator's value, not recomputed
    assert grid_only_reward_scale(cfg, "linear") == 0.096879

    # the loader honours it: base default is 1.0, so the override really applied
    assert BASE["reward_scale"] == 1.0
    assert cfg["reward_scale"] != BASE["reward_scale"]


def test_agent_helper_matches_training_kwargs():
    """`load_agent_for_method` must build the same PPO agent as the trainer."""
    import ast
    import inspect
    from experiments import train_core as pp
    from agents.ppo import PPOAgent

    def _calls_kwargs_helper(fn):
        tree = ast.parse(inspect.getsource(fn))
        return any(
            isinstance(node, ast.Call) and getattr(node.func, "id", None) == "_ppo_kwargs"
            for node in ast.walk(tree))

    assert _calls_kwargs_helper(pp.train_single_method_ppo)
    assert _calls_kwargs_helper(pp.load_agent_for_method)

    # Non-residual configs build without the dataset (the residual base policy
    # reads the Train price quantile, so it is checked separately below).
    for exp in ("s3_ppo", "s3_ppo_lag", "s3_sh_ppo"):
        cfg = load_config_with_base(exp)
        a = pp.load_agent_for_method(cfg["method"], cfg)
        assert isinstance(a, PPOAgent)
        assert a.gamma == cfg["gamma"]
        assert a.gae_lambda == cfg["gae_lambda"]
        assert a.clip_ratio == cfg["ppo_clip_ratio"]
        assert a.cost_limit == cfg["cost_limit"]
        assert a.reward_scale == cfg["reward_scale"]
        assert a.lagrangian == ("lagrangian" in cfg["method"])
        assert a.pbrs == ("potential_shaping" in cfg["method"])
        assert a.use_minibatches == cfg["ppo_use_minibatches"]
        assert a.store_raw_action == cfg["ppo_store_raw_action"]
        assert a.residual_coef == cfg["residual_coef"]
        assert a.residual_policy is None


@pytest.mark.needs_real_data
def test_residual_helper_attaches_the_base_policy():
    """The residual agent must be built on top of the live PSM' heuristic."""
    import numpy as np
    from experiments import train_core as pp
    from agents.ppo import PPOAgent

    cfg = load_config_with_base("s3_ppo_res")
    agent = pp.load_agent_for_method(cfg["method"], cfg)
    assert isinstance(agent, PPOAgent)
    assert agent.residual_policy is not None
    assert agent.residual_policy.get_action(
        np.zeros(17, dtype=np.float32)).shape == (6,)


@pytest.mark.needs_real_data
def test_grid_only_reward_scale_matches_documented_value():
    """The Grid-Only reward scale must reproduce c = 0.096879 on the Train split."""
    from experiments.train_core import grid_only_reward_scale

    cfg = dict(BASE, reward_scale_mode="grid_only", reward_scale_explicit=False)
    c = grid_only_reward_scale(cfg, cfg["shed_mapping"], method_name="t")
    # The published value is 0.096879 from the full Train split.
    assert abs(c - 0.096879) < 1e-4, c

    # an explicit value is respected verbatim
    cfg2 = dict(cfg, reward_scale=0.096879, reward_scale_explicit=True)
    assert grid_only_reward_scale(cfg2, "linear", method_name="t") == 0.096879
