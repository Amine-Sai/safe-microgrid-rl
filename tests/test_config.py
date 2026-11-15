import hashlib
import json
import os
import sys

import pytest
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments.train_core import load_config_with_base, PPO_METHODS, _require_ppo

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_PATH = os.path.join(REPO_ROOT, "configs", "config.yaml")

# Published Grid-Only reward scale (1 / mean|r| on Train split)
PUBLISHED_GRID_ONLY_REWARD_SCALE = 0.096879

_STORE = yaml.safe_load(open(CONFIG_PATH, encoding="utf-8")) or {}
EXPERIMENTS = _STORE.get("experiments") or {}
BASE = load_config_with_base("configs/config.yaml")

def test_base_yaml_documents_every_key_with_a_default():
    """Every knob lives in the defaults block with default == current live behaviour."""
    from env.smartgrid_env import default_config

    assert BASE["method"] == "ppo"
    assert BASE["shed_mapping"] == "linear"
    assert BASE["reward_scale"] == 1.0
    assert BASE["reward_scale_mode"] == "fixed"
    assert BASE["reward_scale_explicit"] is False
    assert BASE["ppo_clip_ratio"] == 0.2
    assert BASE["rollout_steps"] == 2048
    assert BASE["torch_threads"] == 1
    assert BASE["charge_clamp"] is True

    env_defaults = default_config()
    for key in (
        "forecast_mode",
        "forecast_noise_rel_pv",
        "forecast_noise_rel_wind",
        "forecast_noise_rel_load",
        "demand_charge_eur_per_kw_week",
        "zone_voll_eur_kwh",
        "industrial_trip_penalty_eur",
    ):
        assert key in BASE, f"Missing T9 env key {key!r} in configs/config.yaml defaults"
        assert BASE[key] == env_defaults[key]


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


def test_require_ppo_accepts_only_known_methods():
    for method in PPO_METHODS:
        _require_ppo(method)
    for bad in ("plain_sac", "notppo", "ppo_extra"):
        with pytest.raises(ValueError, match="unsupported method"):
            _require_ppo(bad)


def test_lagrangian_key_is_checked_against_the_resolved_method(tmp_path):
    """A row without `method` inherits ppo, so Lagrangian-only keys must be rejected."""
    path = tmp_path / "lagrangian_keys.yaml"
    path.write_text("lr_dual: 0.1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Lagrangian-only"):
        load_config_with_base(str(path))

    path.write_text("method: ppo_lagrangian\nlr_dual: 0.1\n", encoding="utf-8")
    assert load_config_with_base(str(path))["lr_dual"] == 0.1


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


def test_resolved_config_charge_clamp_matches_env():
    """Gate 3: resolved_config["charge_clamp"] matches build_env charge_clamp."""
    from experiments.train_core import build_env
    for config_id, expected_clamp in (
        ("s3_ppo", False),
        ("s3v2_ppo", True),
        ("unclamped_ppo", False),
        ("clamped_ppo", True),
    ):
        cfg = load_config_with_base(config_id)
        env = build_env(cfg)
        assert env.charge_clamp is expected_clamp
        assert cfg["charge_clamp"] is expected_clamp


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
def test_constant_action_is_grid_only_under_both_mappings(frozen_baselines):
    """Frozen baseline: a no-op action must cost exactly Grid-Only either way."""
    import numpy as np
    from env.smartgrid_env import SmartGridEnv

    expected_cost = frozen_baselines["Grid-Only"]["val_cost_eur"]
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
        assert abs(total - expected_cost) < 0.05, (
            f"{mapping} broke Grid-Only: total={total:.2f}, expected={expected_cost:.2f}"
        )


def test_explicit_reward_scale_beats_the_computed_default():
    """Regression: `reward_scale_explicit: true` must win over `grid_only`."""
    import tempfile
    from experiments.train_core import grid_only_reward_scale

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False,
                                     dir=os.path.dirname(CONFIG_PATH),
                                     encoding="utf-8") as f:
        f.write(f"reward_scale: {PUBLISHED_GRID_ONLY_REWARD_SCALE}\n"
                "reward_scale_mode: grid_only\n"
                "reward_scale_explicit: true\n")
        path = f.name
    try:
        cfg = load_config_with_base(path)
    finally:
        os.unlink(path)

    assert cfg["reward_scale_explicit"] is True
    assert cfg["reward_scale"] == PUBLISHED_GRID_ONLY_REWARD_SCALE          # operator's value, not recomputed
    assert grid_only_reward_scale(cfg, "linear") == PUBLISHED_GRID_ONLY_REWARD_SCALE

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

    from agents.ppo import PPOAuxShieldLossAgent, PPODiffSafetyLayerAgent, PPOShieldedAgent
    from agents.safety_layer import SafePolicyNetwork
    cfg_aux = load_config_with_base("clamped_ppo_shield_aux")
    a_aux = pp.load_agent_for_method(cfg_aux["method"], cfg_aux)
    assert isinstance(a_aux, PPOAuxShieldLossAgent)
    assert type(a_aux) is PPOAuxShieldLossAgent
    assert a_aux.aux_shield_coef == 0.5

    cfg_diff = load_config_with_base("clamped_ppo_diff_shield")
    a_diff = pp.load_agent_for_method(cfg_diff["method"], cfg_diff)
    assert isinstance(a_diff, PPODiffSafetyLayerAgent)
    assert type(a_diff) is PPODiffSafetyLayerAgent
    assert isinstance(a_diff.policy, SafePolicyNetwork)

    # Regime 1 (causal23) and Regime 2 (ci) agents: 1D+CapShield vs 6D joint KKT QP
    for exp_id in ("causal23_ppo_shielded", "ci_ppo_shielded"):
        cfg_sh = load_config_with_base(exp_id)
        ag_sh = pp.load_agent_for_method(cfg_sh["method"], cfg_sh)
        assert type(ag_sh) is PPOShieldedAgent
        assert ag_sh.state_dim == 23
        assert ag_sh.action_dim == 1
        assert ag_sh.store_raw_action is True
        assert ag_sh.use_minibatches is True
        assert pp.is_shielded_run(cfg_sh["method"], cfg_sh) is True

    for exp_id in ("causal23_ppo_diff_shield", "ci_ppo_diff_shield"):
        cfg_d1 = load_config_with_base(exp_id)
        ag_d1 = pp.load_agent_for_method(cfg_d1["method"], cfg_d1)
        assert type(ag_d1) is PPODiffSafetyLayerAgent
        assert isinstance(ag_d1.policy, SafePolicyNetwork)
        assert ag_d1.state_dim == 23
        assert ag_d1.action_dim == 1
        assert ag_d1.store_raw_action is True
        assert pp.is_shielded_run(cfg_d1["method"], cfg_d1) is True

    cfg_d6 = load_config_with_base("ci_ppo_diff_6d")
    ag_d6 = pp.load_agent_for_method(cfg_d6["method"], cfg_d6)
    assert type(ag_d6) is PPODiffSafetyLayerAgent
    assert isinstance(ag_d6.policy, SafePolicyNetwork)
    assert ag_d6.state_dim == 23
    assert ag_d6.action_dim == 6
    assert ag_d6.store_raw_action is True
    assert cfg_d6["shielded"] is False
    assert pp.is_shielded_run(cfg_d6["method"], cfg_d6) is False


@pytest.mark.needs_real_data
def test_regime1_causal23_and_regime2_ci_env_and_scale(tmp_path):
    """Verify T9 env forwarding, 23-D obs, Regime 2 reward scaling, and 6D KKT QP rollout."""
    import numpy as np
    from experiments.train_core import (
        build_env,
        grid_only_reward_scale,
        load_agent_for_method,
        train_single_method_ppo,
    )

    cfg_r1 = load_config_with_base("causal23_ppo_residual")
    env_r1 = build_env(cfg_r1, split="val", seed=42)
    assert env_r1.obs_dim == 23
    assert env_r1.forecast_horizon_hours == 6
    assert env_r1.forecast_mode == "causal_noisy"
    assert env_r1.charge_clamp is True
    ag_res = load_agent_for_method(cfg_r1["method"], cfg_r1)
    assert ag_res.state_dim == 23
    obs_r1, _ = env_r1.reset(seed=42)
    assert obs_r1.shape == (23,)
    act_res, _, _, _ = ag_res.select_action(obs_r1, evaluate=True)
    assert act_res.shape == (6,)

    cfg_ci = load_config_with_base("ci_ppo_diff_6d")
    env_ci = build_env(cfg_ci, split="val", seed=42)
    assert env_ci.obs_dim == 23
    assert env_ci.forecast_mode == "causal_noisy"
    assert env_ci.demand_charge_eur_per_kw_week == pytest.approx(2.333333)
    assert env_ci.industrial_trip_penalty_eur == pytest.approx(50.0)
    assert np.allclose(env_ci.get_zone_voll(), [1.5, 1.2, 0.4, 0.2, 3.0])

    ag_d6 = load_agent_for_method(cfg_ci["method"], cfg_ci)
    obs_ci, _ = env_ci.reset(seed=42)
    act_d6, logp, val, _ = ag_d6.select_action(obs_ci, evaluate=False)
    assert act_d6.shape == (6,)
    assert np.isfinite(logp) and np.isfinite(val)
    ag_d6.store_transition(obs_ci, act_d6, -1.0, 0.0, val, 0.0, logp, False, False)
    assert ag_d6.buffer.actions[0].shape == (6,)

    # Regime 2 Grid-Only scale adapts below Regime 1's 0.096879 due to demand charges
    ci_scale = grid_only_reward_scale(cfg_ci, cfg_ci["shed_mapping"], method_name="ci_test")
    assert 0.0 < ci_scale < PUBLISHED_GRID_ONLY_REWARD_SCALE

    # Smoke-run trainer end-to-end on ci_ppo_diff_6d (6D KKT QP without CapShield)
    smoke_cfg = dict(cfg_ci, total_steps=200, eval_interval_steps=200, rollout_steps=200)
    res = train_single_method_ppo(
        smoke_cfg["method"], smoke_cfg, ckpt_dir=str(tmp_path), run_name="ci6d_smoke", allow_overwrite=True
    )
    assert res["resolved_config"]["reward_scale_effective"] == pytest.approx(ci_scale, rel=1e-6)
    fm = res["final_metrics"]
    assert fm["demand_charge_eur"] > 0.0
    assert fm["total_cost_eur"] == pytest.approx(
        fm["energy_purchase_eur"] + fm["unserved_penalty_eur"]
        + fm["battery_wear_eur"] + fm["demand_charge_eur"] + fm["industrial_trip_eur"]
    )


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
    assert abs(c - PUBLISHED_GRID_ONLY_REWARD_SCALE) < 1e-4, c

    # an explicit value is respected verbatim
    cfg2 = dict(cfg, reward_scale=PUBLISHED_GRID_ONLY_REWARD_SCALE, reward_scale_explicit=True)
    assert grid_only_reward_scale(cfg2, "linear", method_name="t") == PUBLISHED_GRID_ONLY_REWARD_SCALE


def test_reproducibility_artifacts():
    """Verify reproducibility artifacts: LICENSE, requirements.txt, and checksums."""
    # 1. LICENSE exists and is non-empty
    license_path = os.path.join(REPO_ROOT, "LICENSE")
    assert os.path.isfile(license_path), f"Missing LICENSE file at {license_path}"
    with open(license_path, "r", encoding="utf-8") as f:
        license_text = f.read().strip()
    assert len(license_text) > 0, "LICENSE file is empty"

    # 2. requirements.txt exists and every non-empty, non-comment line uses exact '==' pinning
    req_path = os.path.join(REPO_ROOT, "requirements.txt")
    assert os.path.isfile(req_path), f"Missing requirements.txt at {req_path}"
    with open(req_path, "r", encoding="utf-8") as f:
        req_lines = [line.strip() for line in f]

    pinned_count = 0
    for line in req_lines:
        if not line or line.startswith("#"):
            continue
        assert "==" in line, f"Line does not use exact '==' pinning: {line}"
        pkg, ver = line.split("==", 1)
        assert pkg.strip(), f"Missing package name in: {line}"
        assert ver.strip(), f"Missing version in: {line}"
        assert not any(op in ver for op in (">=", "<=", "~=", "!=", ">", "<")), (
            f"Relational operator detected in version: {line}"
        )
        pinned_count += 1
    assert pinned_count > 0, "No packages found in requirements.txt"

    # 3. data/checksums.json exists and contains SHA-256 hashes for opsd, pvgis, and openmeteo
    checksums_path = os.path.join(REPO_ROOT, "data", "checksums.json")
    assert os.path.isfile(checksums_path), f"Missing checksums file at {checksums_path}"
    with open(checksums_path, "r", encoding="utf-8") as f:
        checksums = json.load(f)
    for key in ("opsd", "pvgis", "openmeteo"):
        assert key in checksums, f"Missing '{key}' in data/checksums.json"
        val = checksums[key]
        assert isinstance(val, str) and len(val) == 64, (
            f"Expected 64-char SHA-256 hex string for '{key}', got: {val}"
        )
        int(val, 16)  # Verify valid hex string

    # 4. If data/processed/split_report.json exists, verify processed_sha256 matches disk
    report_path = os.path.join(REPO_ROOT, "data", "processed", "split_report.json")
    if os.path.exists(report_path):
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
        assert "processed_sha256" in report, "split_report.json missing 'processed_sha256'"
        processed_hashes = report["processed_sha256"]
        for split_file in ("train.parquet", "val.parquet", "test.parquet"):
            assert split_file in processed_hashes, (
                f"Missing '{split_file}' in processed_sha256"
            )
            file_path = os.path.join(REPO_ROOT, "data", "processed", split_file)
            if os.path.exists(file_path):
                with open(file_path, "rb") as f:
                    actual_sha = hashlib.sha256(f.read()).hexdigest()
                assert actual_sha == processed_hashes[split_file], (
                    f"Hash mismatch for {split_file}: expected {processed_hashes[split_file]}, "
                    f"got {actual_sha}"
                )
