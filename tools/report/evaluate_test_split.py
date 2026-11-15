#!/usr/bin/env python3
"""Locked one-shot evaluation on the closed Test split (2019 H2).

Requires `--unlock-test-split` to prevent premature execution.
Evaluates baselines (Grid-Only, PSM', MPC) and trained checkpoints on the full
24 Test weeks (4,032 hours) and on the 5 stress weeks (including the cap-binding week 2019-W49).
Includes tail-risk metrics (CVaR_0.95), maximum consecutive breach hours,
shield intervention rates, step latency tracking, and out-of-distribution (OOD) contingency stress suite.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import gymnasium as gym
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

from agents.heuristics import GridOnlyPolicy, PeakShaveMinPolicy
from agents.mpc import MPCPolicy
from env.cap_shield import CapShield
from env.smartgrid_env import SmartGridEnv
from experiments.train_core import load_agent_for_method, load_config_with_base


# =========================================================================
# === Centralized OOD Contingency Configuration ===========================
# =========================================================================

DEFAULT_OOD_CONTINGENCIES: Dict[str, Dict[str, Any]] = {
    "forecast_noise": {
        "noise_std": 0.20,
        "relative": True,
        "description": "Gaussian noise perturbation on lookahead forecast features",
    },
    "pv_inverter_trip": {
        "trip_start_hour": 11,
        "trip_end_hour": 15,
        "description": "Mid-day solar PV inverter trip (generation zeroed during peak hours 11:00-15:00)",
    },
    "bess_derate": {
        "derate_factor": 0.50,
        "description": "BESS power derate (battery maximum power halved to 50%)",
    },
    "grid_cap_reduction": {
        "cap_factor": 0.80,
        "description": "Grid import capacity reduction (contract cap tightened to 80%)",
    },
}


def get_ood_contingency_config(
    contingency: str, overrides: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Retrieve contingency configuration with optional overrides from centralized dict."""
    c_key = str(contingency).strip().lower()
    if c_key not in DEFAULT_OOD_CONTINGENCIES:
        raise ValueError(
            f"Unknown OOD contingency '{contingency}'. Supported: {list(DEFAULT_OOD_CONTINGENCIES.keys())}"
        )
    cfg = dict(DEFAULT_OOD_CONTINGENCIES[c_key])
    if overrides:
        cfg.update(overrides)
    return cfg


# =========================================================================
# === Tail-Risk & Safety Breach Helper ====================================
# =========================================================================

def compute_tail_and_safety_metrics(
    hourly_costs: Sequence[float],
    hourly_exceedances: Sequence[float],
    violation_flags: Optional[Sequence[bool]] = None,
    violation_tolerance_kw: float = 1e-4,
    alpha: float = 0.95,
) -> Dict[str, Union[float, int]]:
    r"""Compute tail-risk (CVaR_0.95) and safety breach metrics across hourly trajectories.

    Metrics:
        - cvar95_cost_eur: Expected value (mean) of the worst $(1 - \alpha)$ tail (top 5%)
          of hourly operating costs (EUR).
        - cvar95_exceedance_kw: Expected value (mean) of the worst $(1 - \alpha)$ tail (top 5%)
          of hourly grid import cap exceedances (kW) across all evaluated hours.
        - max_hourly_exceedance_kw: Maximum single-hour grid import cap exceedance (kW).
        - max_consecutive_violation_hours: Longest consecutive run of hours with a grid cap violation.
    """
    costs = np.asarray(hourly_costs, dtype=np.float64)
    excs = np.asarray(hourly_exceedances, dtype=np.float64)

    # 1. CVaR_0.95 on hourly operating costs (EUR)
    if len(costs) > 0:
        var95_cost = float(np.percentile(costs, 100.0 * alpha))
        tail_costs = costs[costs >= var95_cost]
        cvar95_cost = float(np.mean(tail_costs)) if len(tail_costs) > 0 else var95_cost
    else:
        cvar95_cost = 0.0

    # 2. CVaR_0.95 on hourly grid exceedance (kW) across all evaluated hours
    if len(excs) > 0:
        var95_exc = float(np.percentile(excs, 100.0 * alpha))
        tail_excs = excs[excs >= var95_exc]
        cvar95_exc = float(np.mean(tail_excs)) if len(tail_excs) > 0 else var95_exc
        max_exc = float(np.max(excs))
    else:
        cvar95_exc = 0.0
        max_exc = 0.0

    # 3. Max consecutive breach hours
    if violation_flags is not None:
        flags = [bool(f) for f in violation_flags]
    else:
        flags = [bool(e > violation_tolerance_kw) for e in excs]

    max_consecutive = 0
    current_run = 0
    for v in flags:
        if v:
            current_run += 1
            if current_run > max_consecutive:
                max_consecutive = current_run
        else:
            current_run = 0

    return {
        "cvar95_cost_eur": round(float(cvar95_cost), 2),
        "cvar95_exceedance_kw": round(float(cvar95_exc), 4),
        "max_hourly_exceedance_kw": round(float(max_exc), 4),
        "max_consecutive_violation_hours": int(max_consecutive),
    }


# =========================================================================
# === Non-Destructive OOD Perturbation Wrapper ============================
# =========================================================================

class OODPerturbationWrapper:
    r"""Non-destructive environment wrapper for out-of-distribution (OOD) stress testing.

    Supported contingencies:
        1. 'forecast_noise': Injects Gaussian noise into lookahead forecast features or MPC forecast.
        2. 'pv_inverter_trip': Zeroes solar PV generation during peak mid-day hours.
        3. 'bess_derate': Halves or derates maximum battery charge/discharge power (bat_max_power_kw).
        4. 'grid_cap_reduction': Reduces grid import capacity limit (grid_import_cap_kw).

    Guarantees non-destructive execution:
        - Deep/copy backups of any altered state/series/attributes.
        - Supports Python context manager protocol (`with OODPerturbationWrapper(...) as env:`)
          and explicit `restore()` / `close()` methods.
        - Never mutates dataset files or persistent lock artifacts.
    """

    def __init__(
        self,
        env: Any,
        contingency: str,
        contingency_config: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = 42,
    ) -> None:
        self.env = env
        self.contingency = str(contingency).strip().lower()
        self.config = get_ood_contingency_config(self.contingency, contingency_config)
        self._rng = np.random.RandomState(seed)
        self._restored = False

        # Find unwrapped SmartGridEnv
        self.base_env = getattr(env, "unwrapped", env)

        # Backups for non-destructive restore
        self._orig_pv_kw_series: Optional[np.ndarray] = None
        self._orig_net_load_kw_series: Optional[np.ndarray] = None
        self._orig_bat_max_power_kw: Optional[float] = None
        self._orig_grid_import_cap_kw: Optional[float] = None
        self._orig_shield_cap_kw: Optional[float] = None

        self._apply_contingency()

    def _apply_contingency(self) -> None:
        if self.contingency == "forecast_noise":
            # Handled dynamically on obs during reset/step and via configure_policy()
            pass
        elif self.contingency == "pv_inverter_trip":
            if hasattr(self.base_env, "_pv_kw_series") and hasattr(self.base_env, "_timestamps"):
                import pandas as pd

                self._orig_pv_kw_series = self.base_env._pv_kw_series.copy()
                hours = pd.DatetimeIndex(self.base_env._timestamps).hour.values
                trip_start = int(self.config.get("trip_start_hour", 11))
                trip_end = int(self.config.get("trip_end_hour", 15))
                trip_mask = (hours >= trip_start) & (hours <= trip_end)
                perturbed_pv = self.base_env._pv_kw_series.copy()
                perturbed_pv[trip_mask] = 0.0
                self.base_env._pv_kw_series = perturbed_pv

                if hasattr(self.base_env, "_net_load_kw_series") and hasattr(self.base_env, "_load_kw"):
                    self._orig_net_load_kw_series = self.base_env._net_load_kw_series.copy()
                    self.base_env._net_load_kw_series = np.maximum(
                        0.0, self.base_env._load_kw - self.base_env._pv_kw_series
                    ).astype(np.float32)
        elif self.contingency == "bess_derate":
            factor = float(self.config.get("derate_factor", 0.50))
            if hasattr(self.base_env, "bat_max_power_kw"):
                self._orig_bat_max_power_kw = float(self.base_env.bat_max_power_kw)
                self.base_env.bat_max_power_kw = float(self._orig_bat_max_power_kw * factor)
            if hasattr(self.env, "bat_max_power_kw") and self.env is not self.base_env:
                self.env.bat_max_power_kw = self.base_env.bat_max_power_kw
        elif self.contingency == "grid_cap_reduction":
            factor = float(self.config.get("cap_factor", 0.80))
            if hasattr(self.base_env, "grid_import_cap_kw"):
                self._orig_grid_import_cap_kw = float(self.base_env.grid_import_cap_kw)
                self.base_env.grid_import_cap_kw = float(self._orig_grid_import_cap_kw * factor)
            if hasattr(self.env, "cap_kw"):
                self._orig_shield_cap_kw = float(self.env.cap_kw)
                self.env.cap_kw = float(self._orig_shield_cap_kw * factor)

    def configure_policy(self, policy: Any) -> None:
        """Propagate contingency parameters to policies that support them (e.g. MPCPolicy)."""
        if self.contingency == "forecast_noise":
            noise_std = float(self.config.get("noise_std", 0.20))
            relative = bool(self.config.get("relative", True))
            if hasattr(policy, "forecast_noise_std"):
                policy.forecast_noise_std = noise_std
            if hasattr(policy, "forecast_noise_relative"):
                policy.forecast_noise_relative = relative
        if hasattr(policy, "_sync_parameters"):
            policy._sync_parameters(self)

    def _perturb_obs(self, obs: np.ndarray) -> np.ndarray:
        if self.contingency != "forecast_noise":
            return obs
        noisy_obs = obs.copy()
        noise_std = float(self.config.get("noise_std", 0.20))
        relative = bool(self.config.get("relative", True))
        if len(noisy_obs) > 17:
            future_slice = noisy_obs[17:]
            if relative:
                noise = self._rng.normal(0.0, noise_std * (np.abs(future_slice) + 1e-3))
            else:
                noise = self._rng.normal(0.0, noise_std, size=future_slice.shape)
            noisy_obs[17:] = np.maximum(0.0, future_slice + noise).astype(np.float32)
        else:
            if relative:
                noise_pv = self._rng.normal(0.0, noise_std * (abs(noisy_obs[1]) + 1e-3))
                noise_pr = self._rng.normal(0.0, noise_std * (abs(noisy_obs[7]) + 1e-3))
            else:
                noise_pv = self._rng.normal(0.0, noise_std)
                noise_pr = self._rng.normal(0.0, noise_std)
            noisy_obs[1] = max(0.0, float(noisy_obs[1] + noise_pv))
            noisy_obs[7] = max(0.0, float(noisy_obs[7] + noise_pr))
        return noisy_obs

    def reset(self, **kwargs: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        return self._perturb_obs(obs), info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._perturb_obs(obs), reward, terminated, truncated, info

    def restore(self) -> None:
        """Idempotently restore base environment to unperturbed state."""
        if self._restored:
            return
        if self._orig_pv_kw_series is not None and hasattr(self.base_env, "_pv_kw_series"):
            self.base_env._pv_kw_series = self._orig_pv_kw_series
            self._orig_pv_kw_series = None
        if self._orig_net_load_kw_series is not None and hasattr(self.base_env, "_net_load_kw_series"):
            self.base_env._net_load_kw_series = self._orig_net_load_kw_series
            self._orig_net_load_kw_series = None
        if self._orig_bat_max_power_kw is not None and hasattr(self.base_env, "bat_max_power_kw"):
            self.base_env.bat_max_power_kw = self._orig_bat_max_power_kw
            if hasattr(self.env, "bat_max_power_kw") and self.env is not self.base_env:
                self.env.bat_max_power_kw = self._orig_bat_max_power_kw
            self._orig_bat_max_power_kw = None
        if self._orig_grid_import_cap_kw is not None and hasattr(self.base_env, "grid_import_cap_kw"):
            self.base_env.grid_import_cap_kw = self._orig_grid_import_cap_kw
            self._orig_grid_import_cap_kw = None
        if self._orig_shield_cap_kw is not None and hasattr(self.env, "cap_kw"):
            self.env.cap_kw = self._orig_shield_cap_kw
            self._orig_shield_cap_kw = None
        self._restored = True

    def close(self) -> None:
        self.restore()
        if hasattr(self.env, "close") and callable(self.env.close):
            self.env.close()

    def __enter__(self) -> "OODPerturbationWrapper":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.restore()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)


# =========================================================================
# === Reference Baseline Policies =========================================
# =========================================================================

class ShieldIdlePolicy:
    """Zero-training baseline: idle battery command ([0.0]) inside CapShield."""

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        return np.array([0.0], dtype=np.float32)


class ShieldPSMPolicy:
    """Zero-training baseline: PSM' 1-D battery command inside CapShield."""

    def __init__(self, psm: Optional[PeakShaveMinPolicy] = None) -> None:
        self.psm = psm or PeakShaveMinPolicy()

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        return np.array([self.psm.get_action(obs)[0]], dtype=np.float32)


def get_git_commit() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT)
        return out.decode("utf-8").strip()
    except Exception:
        return "unknown"


# =========================================================================
# === Core Rollout & Measurement Logic ====================================
# =========================================================================

def _step_policy_and_measure(
    env: Any,
    obs: np.ndarray,
    policy: Any,
    agent_wrapper: Any = None,
    action_tol: float = 1e-4,
) -> Tuple[np.ndarray, np.ndarray, float, float, bool, Dict[str, Any]]:
    """Select action, measure step latency, check shield intervention, and step environment.

    Returns:
        (next_obs, raw_action, step_latency_ms, shield_delta_l2, done, info)
    """
    # 1. Action selection with latency measurement
    t_start = time.perf_counter()
    if agent_wrapper is not None:
        raw_action = agent_wrapper.select_action(obs, evaluate=True)[0]
    elif hasattr(policy, "get_action"):
        raw_action = policy.get_action(obs)
    elif hasattr(policy, "act"):
        raw_action = policy.act(obs)
    elif callable(policy):
        raw_action = policy(obs)
    else:
        raise ValueError(f"Unsupported policy type: {type(policy)}")
    t_elapsed_ms = (time.perf_counter() - t_start) * 1000.0

    # Solve latency compatibility (e.g. MPCPolicy with last_solve_ms)
    if hasattr(policy, "last_solve_ms") and policy.last_solve_ms > 0:
        step_latency_ms = float(policy.last_solve_ms)
    elif hasattr(agent_wrapper, "last_solve_ms") and agent_wrapper.last_solve_ms > 0:
        step_latency_ms = float(agent_wrapper.last_solve_ms)
    else:
        step_latency_ms = t_elapsed_ms

    # 2. Shield intervention calculation
    raw_arr = np.asarray(raw_action, dtype=np.float32).ravel()
    if hasattr(env, "compose_action"):
        if len(raw_arr) == 1:
            raw_full = np.array([raw_arr[0], -1.0, -1.0, -1.0, -1.0, -1.0], dtype=np.float32)
        else:
            raw_full = raw_arr
        executed_action = env.compose_action(raw_action, obs)
        shield_delta_l2 = float(np.linalg.norm(executed_action - raw_full))
    else:
        shield_delta_l2 = 0.0

    # 3. Environment transition
    next_obs, _, terminated, truncated, info = env.step(raw_action)
    done = bool(terminated or truncated)

    return next_obs, raw_action, step_latency_ms, shield_delta_l2, done, info


def _evaluate_rollout(
    env: SmartGridEnv,
    start_indices: Sequence[int],
    policy: Any,
    agent_wrapper: Any = None,
    action_tol: float = 1e-4,
) -> Dict[str, Any]:
    """Execute unified deterministic rollout loop across specified week start indices."""
    ep_len = getattr(env, "episode_length", 168)
    dt_hours = float(getattr(env, "dt_hours", 1.0))
    violation_tol = float(getattr(env, "violation_tolerance_kw", 1e-4))

    total_cost = 0.0
    unserved_kwh = 0.0
    demand_kwh = 0.0
    violation_hours = 0
    exceedances: List[float] = []
    hourly_costs: List[float] = []
    violation_flags: List[bool] = []
    latencies_ms: List[float] = []
    shield_deltas: List[float] = []
    cap_clipped_kw: List[float] = []

    if hasattr(policy, "reset") and callable(policy.reset):
        try:
            policy.reset(env)
        except TypeError:
            policy.reset()

    for start_idx in start_indices:
        obs, _ = env.reset(options={"start_idx": start_idx})
        for _ in range(ep_len):
            next_obs, _, lat_ms, s_delta, done, info = _step_policy_and_measure(
                env=env, obs=obs, policy=policy, agent_wrapper=agent_wrapper, action_tol=action_tol
            )
            obs = next_obs

            step_cost = float(info["energy_cost"] + info["unserved_cost"] + info["battery_wear_cost"])
            hourly_costs.append(step_cost)
            total_cost += step_cost

            uns_kw = float(info.get("total_unserved_kw", 0.0))
            serv_kw = float(info.get("total_served_kw", 0.0))
            unserved_kwh += uns_kw * dt_hours
            demand_kwh += (serv_kw + uns_kw) * dt_hours

            exc = float(info["grid_exceedance_kw"])
            exceedances.append(exc)
            is_viol = bool(info.get("is_violation", exc > violation_tol))
            violation_flags.append(is_viol)
            if is_viol:
                violation_hours += 1

            cap_clipped_kw.append(float(info.get("charge_clipped_by_cap_kw", 0.0)))
            latencies_ms.append(lat_ms)
            shield_deltas.append(s_delta)

            if done:
                break

    total_hours = len(exceedances)
    tail_metrics = compute_tail_and_safety_metrics(
        hourly_costs=hourly_costs,
        hourly_exceedances=exceedances,
        violation_flags=violation_flags,
        violation_tolerance_kw=violation_tol,
    )

    shield_interventions = [1.0 if d > action_tol else 0.0 for d in shield_deltas]
    shield_rate = float(np.mean(shield_interventions)) if shield_interventions else 0.0
    shield_mean_l2 = float(np.mean(shield_deltas)) if shield_deltas else 0.0

    lat_p50 = float(np.percentile(latencies_ms, 50)) if latencies_ms else 0.0
    lat_p99 = float(np.percentile(latencies_ms, 99)) if latencies_ms else 0.0
    lat_mean = float(np.mean(latencies_ms)) if latencies_ms else 0.0

    violation_rate = float(violation_hours / max(1, total_hours) * 100.0)
    unserved_rate = float(unserved_kwh / max(1e-6, demand_kwh) * 100.0)
    mean_exc = float(np.mean(exceedances)) if exceedances else 0.0
    mean_cap_clip = float(np.mean(cap_clipped_kw)) if cap_clipped_kw else 0.0

    return {
        "cost_eur": round(float(total_cost), 2),
        "mean_exceedance_kw": round(mean_exc, 4),
        "violation_hours": int(violation_hours),
        "violation_rate_pct": round(violation_rate, 2),
        "unserved_pct": round(unserved_rate, 4),
        "mean_cap_clipped_kw": round(mean_cap_clip, 4),
        "cvar95_cost_eur": tail_metrics["cvar95_cost_eur"],
        "cvar95_exceedance_kw": tail_metrics["cvar95_exceedance_kw"],
        "max_hourly_exceedance_kw": tail_metrics["max_hourly_exceedance_kw"],
        "max_consecutive_violation_hours": tail_metrics["max_consecutive_violation_hours"],
        "shield_intervention_rate": round(float(shield_rate), 4),
        "shield_mean_delta_l2": round(float(shield_mean_l2), 4),
        "latency_p50_ms": round(float(lat_p50), 4),
        "latency_p99_ms": round(float(lat_p99), 4),
        "latency_mean_ms": round(float(lat_mean), 4),
    }


def evaluate_single_week(
    env: SmartGridEnv,
    start_idx: int,
    policy: Any,
    agent_wrapper: Any = None,
    action_tol: float = 1e-4,
) -> Dict[str, Any]:
    """Evaluate one 168-hour week starting at start_idx with full safety & tail metrics."""
    return _evaluate_rollout(
        env=env,
        start_indices=[start_idx],
        policy=policy,
        agent_wrapper=agent_wrapper,
        action_tol=action_tol,
    )


def evaluate_policy_on_split(
    env: SmartGridEnv,
    policy: Any,
    agent_wrapper: Any = None,
    action_tol: float = 1e-4,
) -> Dict[str, Any]:
    """Deterministic rollout over all weeks of an environment split with full safety & tail metrics."""
    ep_len = getattr(env, "episode_length", 168)
    n_timesteps = getattr(env, "_n_timesteps", ep_len)
    n_weeks = max(1, n_timesteps // ep_len)
    start_indices = [w * ep_len for w in range(n_weeks)]
    return _evaluate_rollout(
        env=env,
        start_indices=start_indices,
        policy=policy,
        agent_wrapper=agent_wrapper,
        action_tol=action_tol,
    )


# =========================================================================
# === OOD Stress Suite & MPC Baselines Evaluators =========================
# =========================================================================

def evaluate_ood_stress_suite(
    env: SmartGridEnv,
    policy: Any,
    agent_wrapper: Any = None,
    start_idx: Optional[int] = None,
    contingencies: Optional[List[str]] = None,
    contingency_configs: Optional[Dict[str, Dict[str, Any]]] = None,
    action_tol: float = 1e-4,
) -> Dict[str, Dict[str, Any]]:
    """Evaluate policy under nominal conditions and 4 OOD stress contingencies.

    Args:
        env: Target SmartGridEnv or wrapped environment.
        policy: Policy instance with get_action / act method.
        agent_wrapper: Optional trained RL agent wrapper.
        start_idx: Optional single-week start index (e.g. W49). If None, evaluates across full split.
        contingencies: List of contingency keys to evaluate. Defaults to all 4 contingencies.
        contingency_configs: Optional dictionary of parameter overrides per contingency.
        action_tol: Tolerance for shield intervention detection.

    Returns:
        Dictionary mapping contingency names ('nominal', 'forecast_noise', 'pv_inverter_trip',
        'bess_derate', 'grid_cap_reduction') to evaluation metric dictionaries.
    """
    suite_contingencies = contingencies or list(DEFAULT_OOD_CONTINGENCIES.keys())
    results: Dict[str, Dict[str, Any]] = {}

    # 1. Nominal (unperturbed) baseline
    if start_idx is not None:
        results["nominal"] = evaluate_single_week(
            env, start_idx=start_idx, policy=policy, agent_wrapper=agent_wrapper, action_tol=action_tol
        )
    else:
        results["nominal"] = evaluate_policy_on_split(
            env, policy=policy, agent_wrapper=agent_wrapper, action_tol=action_tol
        )

    # 2. Perturbed OOD evaluations
    for cont in suite_contingencies:
        cfg = get_ood_contingency_config(cont, (contingency_configs or {}).get(cont))
        with OODPerturbationWrapper(env, contingency=cont, contingency_config=cfg) as ood_env:
            ood_env.configure_policy(policy)
            if start_idx is not None:
                results[cont] = evaluate_single_week(
                    ood_env, start_idx=start_idx, policy=policy, agent_wrapper=agent_wrapper, action_tol=action_tol
                )
            else:
                results[cont] = evaluate_policy_on_split(
                    ood_env, policy=policy, agent_wrapper=agent_wrapper, action_tol=action_tol
                )

        if hasattr(policy, "_sync_parameters"):
            policy._sync_parameters(env)

    return results


def evaluate_mpc_baselines(
    env: SmartGridEnv,
    start_idx: Optional[int] = None,
    forecast_modes: Optional[List[str]] = None,
    horizon: int = 24,
    use_milp: bool = False,
    forecast_noise_std: float = 0.0,
    forecast_noise_relative: bool = False,
    action_tol: float = 1e-4,
) -> Dict[str, Dict[str, Any]]:
    """Evaluate receding-horizon causal MPC baselines across forecast regimes.

    Args:
        env: Target SmartGridEnv or wrapped environment.
        start_idx: Optional single-week start index. If None, evaluates across full split.
        forecast_modes: List of forecast modes (default: ['perfect', 'persistence']).
        horizon: MPC lookahead horizon in hours (default: 24).
        use_milp: Whether to solve mixed-integer linear programming.
        forecast_noise_std: Optional forecast noise std.
        forecast_noise_relative: Whether forecast noise is relative.
        action_tol: Tolerance for shield intervention detection.

    Returns:
        Dictionary mapping MPC baseline names (e.g. 'MPC_perfect', 'MPC_persistence')
        to evaluation metric dictionaries.
    """
    modes = forecast_modes or ["perfect", "persistence"]
    results: Dict[str, Dict[str, Any]] = {}

    for mode in modes:
        mpc_pol = MPCPolicy(
            env=env,
            horizon=horizon,
            forecast_mode=mode,
            use_milp=use_milp,
            forecast_noise_std=forecast_noise_std,
            forecast_noise_relative=forecast_noise_relative,
        )
        model_name = f"MPC_{mode}"
        if start_idx is not None:
            results[model_name] = evaluate_single_week(
                env, start_idx=start_idx, policy=mpc_pol, action_tol=action_tol
            )
        else:
            results[model_name] = evaluate_policy_on_split(
                env, policy=mpc_pol, action_tol=action_tol
            )
    return results


# =========================================================================
# === Locked Test Split CLI Entrypoint ====================================
# =========================================================================

def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate on closed Test split.")
    parser.add_argument("--unlock-test-split", action="store_true",
                        help="Safety gate: explicitly unlock evaluation on the Test split.")
    parser.add_argument("--family", default="clamped",
                        choices=["clamped", "clamped_mb", "clamped_ppo_diff_shield", "clamped_ppo_shield_aux", "unclamped", "all"],
                        help="Which checkpoint family to evaluate.")
    parser.add_argument("--include-mpc", action="store_true",
                        help="Include receding-horizon MPC baselines in evaluation.")
    parser.add_argument("--include-ood", action="store_true",
                        help="Include OOD contingency stress suite in evaluation.")
    args = parser.parse_args()

    if not args.unlock_test_split:
        print("[LOCKED] The Test split (2019 H2) is closed.")
        print("Pass `--unlock-test-split` to confirm this is the final one-shot evaluation.")
        return 1

    split_name = "test"
    report_path = os.path.join(REPO_ROOT, "data", "processed", "split_report.json")
    if not os.path.exists(report_path):
        print(f"Error: {report_path} not found.")
        return 1

    with open(report_path, encoding="utf-8") as f:
        split_report = json.load(f)

    test_stress = split_report.get("stress_weeks", {}).get("test", {})

    # Build reference environment
    env = SmartGridEnv({"split": split_name, "charge_clamp": True})
    ep_len = env.episode_length
    n_weeks = env._n_timesteps // ep_len

    # Map stress week labels to start_idx in the test environment
    import pandas as pd
    test_parquet = os.path.join(REPO_ROOT, "data", "processed", f"{split_name}.parquet")
    df_test = pd.read_parquet(test_parquet)
    iso_labels = (
        df_test.index.isocalendar().year.astype(str)
        + "-W"
        + df_test.index.isocalendar().week.astype(str).str.zfill(2)
    ).values

    stress_indices = {}
    for crit, info in test_stress.items():
        week_target = info["week"]
        matches = np.where(iso_labels == week_target)[0]
        if len(matches) >= ep_len:
            stress_indices[crit] = int(matches[0])

    print("=" * 80)
    print(f"LOCKED TEST SPLIT EVALUATION (Split: {split_name}, {n_weeks} weeks, {env._n_timesteps} hours)")
    print("=" * 80)

    # 1. Evaluate reference heuristics and shield baselines
    eval_records = []
    stress_records = []

    shield_env = CapShield(SmartGridEnv({"split": split_name, "charge_clamp": True}))
    psm_pol = PeakShaveMinPolicy()

    baselines = [
        ("Grid-Only", env, GridOnlyPolicy()),
        ("PSM'", env, psm_pol),
        ("Shield + Idle", shield_env, ShieldIdlePolicy()),
        ("Shield + PSM'", shield_env, ShieldPSMPolicy(psm_pol)),
    ]
    for name, eval_e, pol in baselines:
        print(f"Evaluating {name}...", end=" ", flush=True)
        res = evaluate_policy_on_split(eval_e, pol)
        print(f"Cost={res['cost_eur']:,.2f} EUR | Exc={res['mean_exceedance_kw']:.4f} kW | Viol={res['violation_rate_pct']:.2f}% | CVaR95={res['cvar95_cost_eur']:,.2f}")
        eval_records.append({
            "model": name, "family": "baseline", "seed": "N/A", **res
        })
        for crit, s_idx in stress_indices.items():
            s_res = evaluate_single_week(eval_e, s_idx, pol)
            stress_records.append({
                "model": name, "family": "baseline", "seed": "N/A",
                "stress_criterion": crit, "week": test_stress[crit]["week"], **s_res
            })

    # Optional MPC baselines
    if args.include_mpc:
        print("Evaluating MPC baselines...", end=" ", flush=True)
        mpc_results = evaluate_mpc_baselines(env, forecast_modes=["perfect", "persistence"])
        for mpc_name, m_res in mpc_results.items():
            print(f"\n  {mpc_name}: Cost={m_res['cost_eur']:,.2f} EUR | Exc={m_res['mean_exceedance_kw']:.4f} kW", end="")
            eval_records.append({
                "model": mpc_name, "family": "baseline_mpc", "seed": "N/A", **m_res
            })
            for crit, s_idx in stress_indices.items():
                s_res = evaluate_single_week(env, s_idx, MPCPolicy(env=env, forecast_mode=mpc_name.replace("MPC_", "")))
                stress_records.append({
                    "model": mpc_name, "family": "baseline_mpc", "seed": "N/A",
                    "stress_criterion": crit, "week": test_stress[crit]["week"], **s_res
                })
        print()

    # 2. Find and evaluate matching checkpoints
    ckpt_pattern = os.path.join(REPO_ROOT, "checkpoints", "*", "*_final.pt")
    all_ckpts = sorted(glob.glob(ckpt_pattern))

    for ckpt_path in all_ckpts:
        fname = os.path.basename(ckpt_path)
        if fname.startswith("clamped_mb_"):
            ckpt_fam = "clamped_mb"
        elif fname.startswith("clamped_ppo_diff_shield_"):
            ckpt_fam = "clamped_ppo_diff_shield"
        elif fname.startswith("clamped_ppo_shield_aux_"):
            ckpt_fam = "clamped_ppo_shield_aux"
        elif fname.startswith("clamped_") or fname.startswith("s3v2_"):
            ckpt_fam = "clamped"
        elif fname.startswith("unclamped_") or fname.startswith("s3_"):
            ckpt_fam = "unclamped"
        else:
            continue

        if args.family != "all" and ckpt_fam != args.family:
            continue

        stem = fname.replace("_final.pt", "")
        parts = stem.split("_seed")
        config_name = parts[0]
        seed = int(parts[1]) if len(parts) > 1 else 42

        try:
            cfg = load_config_with_base(config_name)
            cfg["seed"] = seed
            method = cfg["method"]
            agent = load_agent_for_method(method, cfg)
            agent.load_models(ckpt_path)
        except Exception as e:
            print(f"Skipping {fname}: {e}")
            continue

        env_cfg = {"split": split_name, "charge_clamp": cfg.get("charge_clamp", True)}
        env_eval = SmartGridEnv(env_cfg)

        if cfg.get("shielded"):
            env_eval = CapShield(env_eval)

        print(f"Evaluating {stem}...", end=" ", flush=True)
        res = evaluate_policy_on_split(env_eval, None, agent_wrapper=agent)
        print(f"Cost={res['cost_eur']:,.2f} EUR | Exc={res['mean_exceedance_kw']:.4f} kW | Viol={res['violation_rate_pct']:.2f}% | CVaR95={res['cvar95_cost_eur']:,.2f}")
        eval_records.append({
            "model": stem, "family": ckpt_fam, "seed": seed, **res
        })
        for crit, s_idx in stress_indices.items():
            s_res = evaluate_single_week(env_eval, s_idx, None, agent_wrapper=agent)
            stress_records.append({
                "model": stem, "family": ckpt_fam, "seed": seed,
                "stress_criterion": crit, "week": test_stress[crit]["week"], **s_res
            })

    # Save outputs
    tables_dir = os.path.join(REPO_ROOT, "results", "tables")
    runs_dir = os.path.join(REPO_ROOT, "results", "runs")
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(runs_dir, exist_ok=True)

    out_csv = os.path.join(tables_dir, "test_evaluation.csv")
    if eval_records:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(eval_records[0]))
            w.writeheader()
            w.writerows(eval_records)
        print(f"Wrote {out_csv} ({len(eval_records)} rows)")

    stress_csv = os.path.join(tables_dir, "test_stress_weeks.csv")
    if stress_records:
        with open(stress_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(stress_records[0]))
            w.writeheader()
            w.writerows(stress_records)
        print(f"Wrote {stress_csv} ({len(stress_records)} rows)")

    lock_record = {
        "timestamp_utc": str(pd.Timestamp.now("UTC")),
        "git_commit": get_git_commit(),
        "split": split_name,
        "n_hours": env._n_timesteps,
        "n_weeks": n_weeks,
        "records_count": len(eval_records),
    }
    lock_file = os.path.join(runs_dir, "test_lock.json")
    with open(lock_file, "w", encoding="utf-8") as f:
        json.dump(lock_record, f, indent=2)
    print(f"Wrote lock provenance to {lock_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
