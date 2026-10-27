import sys
import os
import time
import json
from typing import Optional
import yaml
import numpy as np
import torch as th

# Ensure repo root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from env.smartgrid_env import SmartGridEnv


def set_seed(seed: int = 42):
    np.random.seed(seed)
    th.manual_seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(seed)


def evaluate_deterministic_val(env_val: SmartGridEnv, agent, extra_log=None) -> dict:
    """Evaluate an agent deterministically on all validation weeks."""
    base_env = getattr(env_val, "env", env_val)
    episode_hours = base_env.episode_length
    num_val_weeks = base_env._n_timesteps // episode_hours
    total_val_hours = num_val_weeks * episode_hours

    sat_thresh = float(getattr(base_env, "action_saturation_threshold", 0.99))
    soc_pin_tol = float(getattr(base_env, "soc_pinned_tolerance", 0.005))
    soc_lower_pin = base_env.bat_soc_min + soc_pin_tol
    soc_upper_pin = base_env.bat_soc_max - soc_pin_tol

    total_purchase = 0.0
    total_unserved = 0.0
    total_wear = 0.0
    violation_hours = 0
    unserved_kwh = 0.0
    demand_kwh = 0.0
    exceedances = []
    sat_fracs = []
    soc_pins = []
    cap_clipped_kw_list = []
    soc_clipped_kw_list = []
    is_cap_clipped_list = []

    # The observation normalizer stays frozen during evaluation.
    for week in range(num_val_weeks):
        obs, _ = env_val.reset(options={"start_idx": week * episode_hours})
        for _ in range(episode_hours):
            # PPOAgent applies the residual internally; evaluate=True uses the mean action.
            action = agent.select_action(obs, evaluate=True)[0]
            sat_fracs.append(float(np.mean(np.abs(np.asarray(action)) >= sat_thresh)))
            soc_pins.append(float(obs[0] <= soc_lower_pin or obs[0] >= soc_upper_pin))

            obs, _, _, _, info = env_val.step(action)
            total_purchase += info["energy_cost"]
            total_unserved += info["unserved_cost"]
            total_wear += info["battery_wear_cost"]
            unserved_kwh += info["total_unserved_kw"] * base_env.dt_hours
            demand_kwh += (info["total_served_kw"] + info["total_unserved_kw"]) * base_env.dt_hours

            grid_import = info["grid_import_kw"]
            excess = max(0.0, grid_import - base_env.grid_import_cap_kw)
            exceedances.append(excess)
            if info.get("is_violation", excess > base_env.violation_tolerance_kw):
                violation_hours += 1

            if "charge_clipped_by_cap_kw" in info:
                cap_clipped_kw_list.append(info["charge_clipped_by_cap_kw"])
                soc_clipped_kw_list.append(info.get("charge_clipped_by_soc_kw", 0.0))
                is_cap_clipped_list.append(bool(info.get("is_charge_clipped_by_cap", False)))
            elif "charge_clipped_kw" in info:
                cap_clipped_kw_list.append(info["charge_clipped_kw"])
                soc_clipped_kw_list.append(0.0)
                is_cap_clipped_list.append(bool(info.get("charge_clipped_by_cap", False)))

    total_cost = total_purchase + total_unserved + total_wear
    violation_rate = (violation_hours / total_val_hours) * 100.0
    unserved_pct = (unserved_kwh / demand_kwh) * 100.0
    mean_exceedance = float(np.mean(exceedances))
    out = {
        "energy_purchase_eur": float(total_purchase),
        "unserved_penalty_eur": float(total_unserved),
        "battery_wear_eur": float(total_wear),
        "total_cost_eur": float(total_cost),
        "unserved_energy_pct": float(unserved_pct),
        "violation_rate_pct": float(violation_rate),
        "violation_hours": int(violation_hours),
        "mean_exceedance_kw": float(mean_exceedance),
        "frac_saturated": float(np.mean(sat_fracs)),
        "soc_pinned_pct": float(np.mean(soc_pins)) * 100.0,
    }
    if cap_clipped_kw_list:
        out["mean_charge_clipped_by_cap_kw"] = float(np.mean(cap_clipped_kw_list))
        out["mean_charge_clipped_by_soc_kw"] = float(np.mean(soc_clipped_kw_list))
        out["cap_clipped_steps_pct"] = float(np.mean(is_cap_clipped_list)) * 100.0
        # Backward compatibility aliases
        out["charge_clipped_kw_mean"] = out["mean_charge_clipped_by_cap_kw"]
        out["clipped_pct"] = out["cap_clipped_steps_pct"]
    if extra_log:
        out.update(extra_log)
    return out


RESULTS_DIR = os.path.join("results", "runs")   # finished run JSONs, one folder per agent
CHECKPOINT_DIR = "checkpoints"                 # policy weights, one folder per agent

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_FILE = os.path.join(REPO_ROOT, "configs", "config.yaml")

# Keys that only make sense for a Lagrangian method.
LAG_ONLY = {"lr_dual", "lambda_init", "lambda_lr_mult", "lag_normalize_by_1p_lam"}

# Every method the trainer accepts; the CLIs read this list.
PPO_METHODS = (
    "ppo",
    "ppo_lagrangian",
    "ppo_potential_shaping",
    "ppo_lagrangian_potential_shaping",
)

ENV_KEYS = (
    "data_path", "battery_capacity_kwh", "battery_max_power_kw",
    "battery_eta_charge", "battery_eta_discharge", "battery_soc_min",
    "battery_soc_max", "battery_soc_init", "cost_unserved_eur_kwh",
    "cost_wear_eur_kwh", "grid_import_cap_kw", "soc_reserve_floor",
    "pv_area_m2", "pv_efficiency", "pv_performance_ratio",
    "motor_min_pct", "episode_length_hours", "shed_mapping",
    "charge_clamp", "dt_hours", "industrial_shutdown_threshold",
    "ind_off_threshold", "cap_clamp_tolerance_kw", "cap_clamp_tol_kw",
    "violation_tolerance_kw", "violation_tol_kw",
    "action_saturation_threshold", "action_sat_threshold",
    "soc_pinned_tolerance", "soc_pin_tol", "forecast_horizon_hours",
)


def build_env(cfg: dict, split: str = "train", seed: Optional[int] = None) -> SmartGridEnv:
    """Build a SmartGridEnv forwarding whitelisted env configuration keys from cfg."""
    env_cfg = {"split": split}
    if seed is not None:
        env_cfg["seed"] = seed
    elif "seed" in cfg:
        env_cfg["seed"] = cfg["seed"]

    for k in ENV_KEYS:
        if k in cfg:
            env_cfg[k] = cfg[k]

    return SmartGridEnv(env_cfg)


def agent_name(method_name: str, cfg: dict) -> str:
    """Folder name for a run: ppo, ppo_lagrangian, ppo_residual, or ppo_shielded."""
    if cfg.get("shielded"):
        return "ppo_shielded"
    if cfg.get("residual_on_heuristic"):
        return "ppo_residual"
    return method_name


def run_paths(method_name: str, cfg: dict, run_id: str):
    """Result JSON path and checkpoint directory for one (method, config, seed) run."""
    agent = agent_name(method_name, cfg)
    stem = "%s_seed%s" % (run_id, cfg["seed"])
    return (os.path.join(RESULTS_DIR, agent, stem + ".json"),
            os.path.join(CHECKPOINT_DIR, agent))


def grid_only_reward_scale(cfg: dict, shed_mapping: str, method_name: str = "agent") -> float:
    """Constant reward scale c = 1/mean|r| on Grid-Only Train.

    Recomputed from the full Train split; an explicit `reward_scale` always wins.
    """
    reward_scale = cfg.get("reward_scale", 1.0)
    if not (cfg.get("reward_scale_mode", "fixed") == "grid_only"
            and not cfg.get("reward_scale_explicit")):
        return reward_scale
    from agents.heuristics import GridOnlyPolicy

    env = build_env(dict(cfg, shed_mapping=shed_mapping), split="train", seed=cfg["seed"])
    env.action_space.seed(cfg["seed"])
    env.reset()
    policy = GridOnlyPolicy()
    rewards = []
    ep_hours = env.episode_length
    n_episodes = env._n_timesteps // ep_hours
    for episode in range(n_episodes):
        obs, _ = env.reset(options={"start_idx": episode * ep_hours})
        for _ in range(ep_hours):
            next_obs, reward, _, truncated, _ = env.step(policy.get_action(obs))
            rewards.append(abs(reward))
            obs = next_obs
            # Reset only while another episode remains: a trailing reset would index
            # past the end of the split.
            if truncated and episode + 1 < n_episodes:
                obs, _ = env.reset(options={"start_idx": (episode + 1) * ep_hours})
    scale = 1.0 / float(sum(rewards) / len(rewards))
    print(f"[{method_name}] reward_scale c = 1/mean|r| = {scale:.6f} "
          f"(Grid-Only Train, {len(rewards)} steps, {n_episodes} episodes)", flush=True)
    return scale


def _require_ppo(method_name: str) -> None:
    """Reject method names outside the PPO family this release ships."""
    if method_name not in PPO_METHODS:
        raise ValueError(
            f"unsupported method {method_name!r}: this release ships the PPO family "
            f"only ({', '.join(PPO_METHODS)})"
        )


def load_agent_for_method(method_name: str, cfg: dict):
    """Build an agent without training, using the same kwargs as the trainer."""
    _require_ppo(method_name)

    from agents.ppo import PPOAgent, PPOPotentialShapingAgent

    kwargs = _ppo_kwargs(cfg, method_name)
    if cfg.get("residual_on_heuristic", False):
        kwargs["residual_policy"] = _ppo_residual_policy()
    cls = PPOPotentialShapingAgent if kwargs["pbrs"] else PPOAgent
    return cls(**kwargs)


def _ppo_kwargs(cfg: dict, method_name: str) -> dict:
    """PPOAgent constructor kwargs shared by training and agent restoration."""
    state_dim = cfg.get("state_dim")
    if state_dim is None:
        state_dim = 17 + (6 if int(cfg.get("forecast_horizon_hours", 0)) > 0 else 0)

    return dict(
        state_dim=state_dim, action_dim=cfg["action_dim"],
        shed_mapping=cfg.get("shed_mapping", "linear"),
        lr_policy=cfg.get("lr_policy", 3e-4), lr_value=cfg.get("lr_value", 1e-3),
        lr_dual=cfg.get("lr_dual", 0.01),
        gamma=cfg["gamma"], gae_lambda=cfg.get("gae_lambda", 0.95),
        clip_ratio=cfg.get("ppo_clip_ratio", 0.2),
        entropy_coef=cfg.get("ppo_entropy_coef", 0.0),
        target_kl=cfg.get("ppo_target_kl", 0.05),
        n_epochs=cfg.get("ppo_n_epochs", 10), batch_size=cfg.get("ppo_batch_size", 64),
        reward_scale=cfg.get("reward_scale", 1.0),
        lagrangian=("lagrangian" in method_name),
        pbrs=("potential_shaping" in method_name),
        cost_limit=cfg["cost_limit"], lambda_init=cfg.get("lambda_init", 0.0),
        lambda_lr_mult=cfg.get("lambda_lr_mult", 0.1),
        potential_shaping_kappa=cfg.get("potential_shaping_kappa", 105.0),
        lag_normalize_by_1p_lam=cfg.get("lag_normalize_by_1p_lam", True),
        residual_coef=cfg.get("residual_coef", 0.25),
        use_minibatches=cfg.get("ppo_use_minibatches", False),
        value_epochs=cfg.get("ppo_value_epochs", 1),
        store_raw_action=cfg.get("ppo_store_raw_action", False),
        init_log_std=cfg.get("ppo_init_log_std", -1.0),
        init_log_std_shed=cfg.get("ppo_init_log_std_shed", None),
        device=cfg.get("device", "cpu"),
    )


def _ppo_residual_policy():
    """The PSM' heuristic the residual agent corrects (read-only base policy)."""
    from agents.heuristics import psm_prime_policy

    return psm_prime_policy()


def train_single_method(method_name: str, cfg: dict, ckpt_dir: str = None,
                        run_name: str = "", allow_overwrite: bool = False):
    """Train one run with the PPO-family trainer."""
    return train_single_method_ppo(method_name, cfg, ckpt_dir=ckpt_dir,
                                   run_name=run_name,
                                   allow_overwrite=allow_overwrite)


def train_single_method_ppo(method_name: str, cfg: dict, ckpt_dir: str = None,
                            run_name: str = "", allow_overwrite: bool = False):
    """Train one PPO run: rollouts, GAE, clipped updates, deterministic validation."""
    _require_ppo(method_name)
    run_id = clean_run_id(run_name)
    if ckpt_dir is None:
        _result_path, ckpt_dir = run_paths(method_name, cfg, run_id)
    print(f"\n{'='*75}")
    print(f"PPO RUN: {method_name.upper()} (Seed {cfg['seed']}, {cfg['total_steps']} Steps)")
    print(f"{'='*75}")

    from agents.ppo import PPOAgent, PPOPotentialShapingAgent

    set_seed(cfg["seed"])
    shed_mapping = cfg.get("shed_mapping", "linear")
    env_train = build_env(dict(cfg, shed_mapping=shed_mapping), split="train", seed=cfg["seed"])
    env_train.action_space.seed(cfg["seed"])
    env_val = build_env(dict(cfg, shed_mapping=shed_mapping), split="val", seed=cfg["seed"] + 100)

    # CapShield wraps the env and only sheds the excess over the cap.
    shielded = bool(cfg.get("shielded", False))
    if shielded:
        from env.cap_shield import CapShield
        env_train = CapShield(env_train)
        env_val = CapShield(env_val)

    base_env_val = getattr(env_val, "env", env_val)
    base_env_train = getattr(env_train, "env", env_train)

    total_steps = cfg["total_steps"]
    eval_interval = cfg["eval_interval_steps"]
    rollout_steps = cfg.get("rollout_steps", 2048)
    cost_limit = cfg["cost_limit"]
    is_lagrangian = "lagrangian" in method_name

    kwargs = _ppo_kwargs(cfg, method_name)
    if cfg.get("residual_on_heuristic", False):
        kwargs["residual_policy"] = _ppo_residual_policy()
    cls = PPOPotentialShapingAgent if kwargs["pbrs"] else PPOAgent
    agent = cls(**kwargs)
    agent._rng = np.random.RandomState(cfg["seed"])
    agent._shed_mapping = shed_mapping

    ckpt_stem = "%s_seed%s" % (run_id, cfg["seed"])
    os.makedirs(ckpt_dir, exist_ok=True)
    final_ckpt_path = os.path.join(ckpt_dir, ckpt_stem + "_final.pt")
    if os.path.exists(final_ckpt_path) and not allow_overwrite:
        raise FileExistsError(
            f"refusing to overwrite {final_ckpt_path} — pass allow_overwrite=True "
            f"(or --overwrite) if the old run is really superseded")

    eval_checkpoints, wall_clock_history, lambda_trajectory = {}, {}, []
    all_train_exceedances, all_train_costs_eur = [], []
    actions_taken, soc_history = [], []

    obs, _ = env_train.reset()
    start_time = time.time()
    last_chunk_time = start_time
    last_term = False
    step = 0
    # Eval fires when step crosses a boundary, not on exact multiples: rollout
    # segments never land exactly on 50k.
    next_eval = eval_interval
    while step < total_steps:
        # ---- collect one rollout ----
        for _ in range(min(rollout_steps, total_steps - step)):
            step += 1
            action, logp, value, cost_value = agent.select_action(obs, evaluate=False)
            next_obs, reward, terminated, truncated, info = env_train.step(action)
            cost = float(info["grid_exceedance_kw"])
            grid_import = float(info.get("grid_import_kw", 0.0))
            excess = max(0.0, grid_import - env_train.grid_import_cap_kw)
            all_train_exceedances.append(excess)
            all_train_costs_eur.append(info["energy_cost"] + info["unserved_cost"]
                                       + info["battery_wear_cost"])
            actions_taken.append(np.asarray(action, dtype=np.float32))
            soc_history.append(float(next_obs[0]))
            shaped = agent.shape(reward, obs, next_obs, terminated, truncated)
            agent.store_transition(obs, action, shaped, cost, value, cost_value,
                                   logp, terminated, truncated)
            obs = next_obs
            last_term = bool(terminated)
            if truncated or terminated:
                agent.end_episode(bool(terminated))
                if is_lagrangian:
                    lambda_value = agent.update_lambda_from_source()
                    lambda_trajectory.append({"step": step, "lambda": lambda_value,
                                              "J_c": float(np.mean(agent.cost_history[-1:])) if agent.cost_history else 0.0})
                obs, _ = env_train.reset()

        # ---- update on the rollout (bootstrap unless the segment ended terminal) ----
        with th.no_grad():
            last_state = agent._t(agent.obs_normalizer.normalize(obs[None, :]))
            last_value = float(agent.value_net(last_state))
            last_cost_value = float(agent.cost_net(last_state)) if agent.cost_net is not None else 0.0
        batch = agent.finish_path(0.0 if last_term else last_value,
                                  0.0 if (last_term and is_lagrangian) else last_cost_value)
        if batch is not None:
            agent.update(batch)
        agent._reset_buffer()

        # ---- eval ----
        while next_eval <= step or step >= total_steps:
            if f"{next_eval}_steps" not in eval_checkpoints and next_eval <= step:
                step_tag = next_eval
            elif step >= total_steps and f"{total_steps}_steps" not in eval_checkpoints:
                step_tag = total_steps
            else:
                break
            now = time.time()
            chunk_time = now - last_chunk_time
            last_chunk_time = now
            wall_clock_history[f"{step_tag}_steps"] = {
                "chunk_time_sec": chunk_time,
                "cumulative_time_sec": now - start_time,
                "steps_per_sec": step_tag / max(now - start_time, 1e-9),
            }
            extra = None
            if is_lagrangian:
                mean_cost = float(np.mean(agent.cost_history[-10:])) if agent.cost_history else 0.0
                extra = {"lambda": agent.lambda_, "mean_ep_cost": mean_cost}
            val_metrics = evaluate_deterministic_val(env_val, agent, extra_log=extra)
            eval_checkpoints[f"{step_tag}_steps"] = val_metrics
            # Save every evaluation so a cancelled run keeps its weights.
            ckpt_path = os.path.join(ckpt_dir, f"{ckpt_stem}_step{step_tag}.pt")
            agent.save_models(ckpt_path)
            if step_tag >= next_eval:
                next_eval += eval_interval

    total_wall_clock = time.time() - start_time
    avg_steps_per_sec = total_steps / total_wall_clock

    if not eval_checkpoints and eval_interval <= total_steps:
        raise RuntimeError(f"No evaluation was recorded for {method_name}")
    feasible_checkpoints = [
        (k, v) for k, v in eval_checkpoints.items()
        if v["mean_exceedance_kw"] <= cost_limit + base_env_val.violation_tolerance_kw
    ]
    if feasible_checkpoints:
        best_ckpt_key, best_ckpt_val = min(feasible_checkpoints, key=lambda x: x[1]["total_cost_eur"])
    else:
        best_ckpt_key, best_ckpt_val = min(eval_checkpoints.items(), key=lambda x: (x[1]["mean_exceedance_kw"], x[1]["total_cost_eur"]))
    final_ckpt_val = eval_checkpoints[f"{total_steps}_steps"]

    train_mean_exceedance = float(np.mean(all_train_exceedances))
    train_total_cost_eur = float(np.sum(all_train_costs_eur))
    train_violation_rate_pct = (
        float(np.mean(np.array(all_train_exceedances) > base_env_train.violation_tolerance_kw)) * 100.0
    )

    if is_lagrangian:
        lambda_start = lambda_trajectory[0]["lambda"] if lambda_trajectory else 0.0
        lambda_mid = lambda_trajectory[len(lambda_trajectory) // 2]["lambda"] if lambda_trajectory else 0.0
        lambda_end = agent.lambda_
        recent_lambdas = [x["lambda"] for x in lambda_trajectory[-20:]] if len(lambda_trajectory) >= 20 else [lambda_end]
        lambda_std = float(np.std(recent_lambdas))
        lambda_status = "settled" if lambda_std < 0.02 else ("converging" if lambda_std < 0.10 else "oscillating")
    else:
        lambda_start, lambda_mid, lambda_end, lambda_status = None, None, None, "N/A (unconstrained/penalty)"

    action_array = np.array(actions_taken)
    soc_array = np.array(soc_history)
    mean_battery_action = float(np.mean(action_array[:, 0]))
    # 1-D shielded actions have no shed dims.
    mean_shed_action = float(np.mean(action_array[:, 1:])) if action_array.shape[1] > 1 else 0.0
    soc_pinned_lo = base_env_train.soc_reserve_floor + base_env_train.soc_pinned_tolerance
    soc_pinned_hi = base_env_train.bat_soc_max - base_env_train.soc_pinned_tolerance
    soc_pinned_pct = float(np.mean((soc_array <= soc_pinned_lo) | (soc_array >= soc_pinned_hi))) * 100.0
    degenerate_flags = []
    if np.all(action_array[:, 0] <= 0.01):
        degenerate_flags.append("Never charges battery")
    if action_array.shape[1] > 1 and np.all(action_array[:, 1:] < -0.95):
        degenerate_flags.append("Never sheds load")
    elif action_array.shape[1] > 1 and np.all(action_array[:, 1:] > 0.90):
        degenerate_flags.append("Sheds everything continuously")
    if soc_pinned_pct > 90.0:
        degenerate_flags.append("Runs SoC constantly pinned at bounds")
    is_degenerate = len(degenerate_flags) > 0
    degeneracy_summary = "; ".join(degenerate_flags) if is_degenerate else "None (healthy dynamic dispatch)"

    agent.save_models(final_ckpt_path)

    return {
        "method": method_name,
        "agent": agent_name(method_name, cfg),
        "run_id": run_id,
        "seed": cfg["seed"],
        "total_steps": total_steps,
        "resolved_config": dict(
            cfg,
            grid_import_cap_kw=env_train.grid_import_cap_kw,
            charge_clamp=env_train.charge_clamp,
            reward_scale_effective=cfg.get("reward_scale", 1.0),
        ),
        "wall_clock_sec": total_wall_clock,
        "steps_per_sec": avg_steps_per_sec,
        "wall_clock_history": wall_clock_history,
        "eval_checkpoints": eval_checkpoints,
        "final_checkpoint": {
            "checkpoint": f"{total_steps}_steps",
            "metrics": final_ckpt_val,
            "weights": final_ckpt_path,
        },
        "best_checkpoint": {
            "checkpoint": best_ckpt_key,
            "metrics": best_ckpt_val,
            "weights": os.path.join(ckpt_dir, f"{ckpt_stem}_step{best_ckpt_key.split('_')[0]}.pt"),
            "role": "SECONDARY (reported only to quantify checkpoint-selection optimism)",
        },
        "final_metrics": final_ckpt_val,
        "training_time_metrics": {
            "total_cost_eur": train_total_cost_eur,
            "mean_exceedance_kw": train_mean_exceedance,
            "violation_rate_pct": train_violation_rate_pct,
            "cost_limit_d_main": cost_limit,
            "satisfies_d_main_on_train": train_mean_exceedance <= cost_limit,
        },
        "lambda_summary": {
            "lambda_start": lambda_start,
            "lambda_mid": lambda_mid,
            "lambda_end": lambda_end,
            "status": lambda_status,
            "trajectory": lambda_trajectory,
        },
        "degeneracy_check": {
            "is_degenerate": is_degenerate,
            "summary": degeneracy_summary,
            "mean_battery_cmd": mean_battery_action,
            "mean_shed_cmd": mean_shed_action,
            "soc_pinned_bounds_pct": soc_pinned_pct,
        }
    }


def clean_run_id(run_name: str) -> str:
    """A run id with no dangling separators: "s3_ppo_" -> "s3_ppo"."""
    return (run_name or "").strip().strip("_").strip()


def resolve_run_name(config_arg: str, run_name_arg: str) -> str:
    """Run id: the experiment id plus an optional extra name.

    For id-based configs the experiment id comes first, so two runs of the same
    method and seed cannot overwrite each other.
    """
    name = clean_run_id(run_name_arg)
    stem = os.path.splitext(os.path.basename(str(config_arg)))[0]
    if resolve_config_path(config_arg):
        # A config file: the explicit run name wins, else the file stem.
        return name or stem
    exp_id = stem
    if not name or name == exp_id or exp_id.endswith(name) or name.endswith(exp_id):
        return exp_id
    return f"{exp_id}_{name}"


def resolve_config_path(path):
    """Absolute path of a config file, or None when `path` is an experiment id."""
    for candidate in (path, os.path.abspath(str(path)), os.path.join(REPO_ROOT, str(path))):
        if candidate and os.path.exists(candidate):
            return os.path.abspath(candidate)
    return None


def load_config_with_base(path):
    """Load the defaults plus an optional experiment or YAML override."""

    def load_yaml(file_path):
        with open(file_path) as f:
            return yaml.safe_load(f) or {}

    store = load_yaml(CONFIG_FILE)
    base = store.get("defaults") or {}
    experiments = store.get("experiments") or {}

    found = resolve_config_path(path)

    if found:
        loaded = load_yaml(found)
        label = str(found)
        # The merged config file itself: its defaults block is the base.
        if "defaults" in loaded and "experiments" in loaded:
            return dict(loaded["defaults"])
        override = loaded
    else:
        exp_id = os.path.splitext(os.path.basename(str(path)))[0]
        if exp_id not in experiments:
            raise FileNotFoundError(f"config {path!r}: no such id in configs/config.yaml")
        override, label = experiments[exp_id] or {}, f"config.yaml:{exp_id}"

    unknown = sorted(k for k in override if k not in base)
    if unknown:
        raise KeyError(f"{label}: unknown config key(s) {unknown}")
    method = override.get("method", base.get("method"))
    if "lagrangian" not in str(method):
        bad_keys = sorted(k for k in override if k in LAG_ONLY)
        if bad_keys:
            raise ValueError(f"{label}: Lagrangian-only key(s) {bad_keys} under method {method!r}")
    cfg = dict(base)
    cfg.update(override)
    return cfg


def main():
    # Kept so old invocations still run; the entry point is experiments/train.py.
    config_path = CONFIG_FILE
    cfg = load_config_with_base(config_path)

    # CLI: python experiments/train_core.py [--methods a,b] [--steps N] [--seed S] [--run-name NAME]
    methods = list(PPO_METHODS)
    steps_override = None
    seed_override = None
    run_name = ""
    argv = sys.argv[1:]
    if "--methods" in argv:
        methods = argv[argv.index("--methods") + 1].split(",")
        assert all(m in PPO_METHODS for m in methods), f"unknown method in {methods}"
    if "--steps" in argv:
        steps_override = int(argv[argv.index("--steps") + 1])
    if "--seed" in argv:
        seed_override = int(argv[argv.index("--seed") + 1])
    if "--run-name" in argv:
        run_name = argv[argv.index("--run-name") + 1]
    if steps_override:
        cfg["total_steps"] = steps_override
    if seed_override is not None:
        cfg["seed"] = seed_override

    run_id = resolve_run_name(config_path, run_name) or "run"
    results = {}
    for method in methods:
        started_at = time.time()
        result = train_single_method(method, cfg, run_name=run_id)
        results[method] = result
        result_path, _ = run_paths(method, cfg, run_id)
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Finished {method} in {time.time() - started_at:.1f}s -> {result_path}", flush=True)

    print("\n" + "=" * 75)
    print(f"COMPLETE: {len(results)} runs -> {RESULTS_DIR}")
    print("=" * 75)


if __name__ == "__main__":
    main()
