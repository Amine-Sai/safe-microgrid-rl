import sys
import os
import time
import json
import yaml
import numpy as np
import torch as th

# Ensure repo root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from env.smartgrid_env import SmartGridEnv
from agents.shaping import PotentialBasedShaping, FixedCostPenalty

# The SAC agents were part of the earlier prototype but are not included in this
# release (the published results are all PPO). They are imported lazily inside the
# SAC branches so the PPO path stays importable; requesting a SAC method raises a
# clear error. See NOTES.md (D1).


def set_seed(seed: int = 42):
    np.random.seed(seed)
    th.manual_seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(seed)


def evaluate_deterministic_val(env_val: SmartGridEnv, agent, normalizer, extra_log=None) -> dict:
    """Evaluate agent deterministically on all 26 weeks of validation with normalizer frozen."""
    tot_pur = 0.0
    tot_uns = 0.0
    tot_wear = 0.0
    tot_viol = 0
    tot_unserved_kwh = 0.0
    tot_demand_kwh = 0.0
    exceedances = []
    tot_hours = 26 * 168
    sat_fracs = []
    soc_pins = []

    is_ppo = hasattr(agent, "policy") and hasattr(agent, "select_action")

    # We do NOT call normalizer.update() during evaluation -> frozen
    for ep in range(26):
        obs, _ = env_val.reset(options={"start_idx": ep * 168})
        for _ in range(168):
            if is_ppo:
                # PPOAgent composes the residual internally; evaluate=True = mean action.
                act = agent.select_action(obs, evaluate=True)[0]
            else:
                norm_obs = normalizer.normalize(obs)
                obs_t = th.as_tensor(norm_obs, dtype=th.float32, device=agent.device).unsqueeze(0)
                with th.no_grad():
                    act = agent.actor.deterministic(obs_t).squeeze(0).cpu().numpy()
            sat_fracs.append(float(np.mean(np.abs(np.asarray(act)) >= 0.99)))
            soc_pins.append(float(obs[0] <= 0.105 or obs[0] >= 0.895))

            obs, r, term, trunc, info = env_val.step(act)
            tot_pur += info["energy_cost"]
            tot_uns += info["unserved_cost"]
            tot_wear += info["battery_wear_cost"]
            tot_unserved_kwh += info["total_unserved_kw"] * env_val.dt_hours
            tot_demand_kwh += (info["total_served_kw"] + info["total_unserved_kw"]) * env_val.dt_hours

            imp = info["grid_import_kw"]
            exc = max(0.0, imp - env_val.grid_import_cap_kw)
            exceedances.append(exc)
            if imp > env_val.grid_import_cap_kw:
                tot_viol += 1

    tot_cost = tot_pur + tot_uns + tot_wear
    viol_rate = (tot_viol / tot_hours) * 100.0
    unserved_pct = (tot_unserved_kwh / tot_demand_kwh) * 100.0
    mean_exc = float(np.mean(exceedances))
    out = {
        "energy_purchase_eur": float(tot_pur),
        "unserved_penalty_eur": float(tot_uns),
        "battery_wear_eur": float(tot_wear),
        "total_cost_eur": float(tot_cost),
        "unserved_energy_pct": float(unserved_pct),
        "violation_rate_pct": float(viol_rate),
        "violation_hours": int(tot_viol),
        "mean_exceedance_kw": float(mean_exc),
        # saturation + SoC-pin fractions at every eval (cheap; works for SAC too).
        "frac_saturated": float(np.mean(sat_fracs)),
        "soc_pinned_pct": float(np.mean(soc_pins)) * 100.0,
    }
    if extra_log:
        out.update(extra_log)
    return out


RESULTS_DIR = os.path.join("results", "runs")   # finished run JSONs, one folder per agent
CHECKPOINT_DIR = "checkpoints"                 # policy weights, one folder per agent

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_FILE = os.path.join(REPO_ROOT, "configs", "config.yaml")

# Lagrangian-only config keys. Under a non-Lagrangian method they are silent
# no-ops, so the loader refuses them in a row that declares such a
# method. LEGACY_EXEMPT holds labels of historical rows that must stay loadable; the scan
# of configs/config.yaml found NONE that trip the guard, so it is empty.
LAG_ONLY = {"lr_dual", "dual_cost_horizon", "lambda_init", "lambda_source",
            "lambda_max", "lambda_lr_mult", "lag_normalize_by_1p_lam"}
LEGACY_EXEMPT = set()


def variant_label(method_name: str, cfg: dict) -> str:
    """The agent a run belongs to — the folder name under results/runs/ and checkpoints/.

    The base agent alone is ambiguous: the residual rung runs the Lagrangian agent
    and the shielded rung runs plain PPO. The two extra flags decide the folder, so
    every artefact of one agent lands in one place.

        ppo | ppo_lagrangian | ppo_residual | ppo_shielded | <SAC method name>
    """
    if cfg.get("shielded"):
        return "ppo_shielded"
    if cfg.get("residual_on_heuristic"):
        return "ppo_residual"
    return method_name


def run_paths(method_name: str, cfg: dict, run_id: str):
    """(result JSON path, checkpoint dir) for one (method, config, seed) run.

    The result file is `results/runs/<agent>/<run id>_seed<S>.json`; the weights go
    to `checkpoints/<agent>/<run id>_seed<S>[_step<N>].pt`. One folder per agent,
    one file per (experiment, seed) — nothing is shared between two experiments.
    """
    agent = variant_label(method_name, cfg)
    stem = "%s_seed%s" % (run_id, cfg["seed"])
    return (os.path.join(RESULTS_DIR, agent, stem + ".json"),
            os.path.join(CHECKPOINT_DIR, agent))


def grid_only_reward_scale(cfg: dict, shed_mapping: str, method_name: str = "agent") -> float:
    """Constant reward scale c = 1/mean|r| on Grid-Only Train.

    Applied to learning targets only (metrics stay unscaled; d and lambda semantics
    unchanged). This recomputes c from the FULL Train split (104 episodes),
    never from 40, and never overwrites an explicitly set `reward_scale`
    (`reward_scale_explicit: true`) — the defect that made the documented
    c=0.096879 differ from the c=0.101380 that actually ran.
    """
    reward_scale = cfg.get("reward_scale", 1.0)
    if not (cfg.get("reward_scale_mode", "fixed") == "grid_only"
            and not cfg.get("reward_scale_explicit")):
        return reward_scale
    from agents.heuristics import GridOnlyPolicy

    _env = SmartGridEnv({"split": "train", "seed": cfg["seed"], "shed_mapping": shed_mapping})
    _env.action_space.seed(cfg["seed"])
    _obs, _ = _env.reset()
    _pol = GridOnlyPolicy()
    _rs = []
    n_train_eps = _env._n_timesteps // 168
    for _ep in range(n_train_eps):
        _obs, _ = _env.reset(options={"start_idx": _ep * 168})
        for _ in range(168):
            _o, _r, _t, _tr, _i = _env.step(_pol.get_action(_obs))
            _rs.append(abs(_r))
            _obs = _o
            # only follow the truncation reset while another episode remains.
            # The trailing reset used to run at the end of the LAST episode and index
            # past the end of the split (IndexError at idx == n_timesteps).
            if _tr and _ep + 1 < n_train_eps:
                _obs, _ = _env.reset(options={"start_idx": (_ep + 1) * 168})
    c = 1.0 / float(sum(_rs) / len(_rs))
    print(f"[{method_name}] reward_scale c = 1/mean|r| = {c:.6f} "
          f"(Grid-Only Train, {len(_rs)} steps, {n_train_eps} episodes)", flush=True)
    return c


def _sac_agents():
    """Import the SAC agents, or explain why they are unavailable.

    The published study is PPO-only; the SAC implementation was part of the
    earlier prototype and is not shipped here. Historical SAC rows remain in
    configs/config.yaml, so this guard replaces a bare ModuleNotFoundError with
    an actionable message.
    """
    try:
        from agents.sac_agent import SACAgent
        from agents.sac_lagrangian import SACLagrangianAgent
    except ModuleNotFoundError as exc:
        if exc.name not in {"agents.sac_agent", "agents.sac_lagrangian"}:
            raise
        raise ModuleNotFoundError(
            f"{exc.name!r} is not included in this PPO-focused release. "
            "Runnable methods are the ones containing 'ppo'; see NOTES.md (D1)."
        ) from exc
    return SACAgent, SACLagrangianAgent


def load_agent_for_method(method_name: str, cfg: dict):
    """Build the agent object for a (method, cfg) pair WITHOUT training.

    Use this to restore the FINAL policy of a finished run and evaluate it. The kwargs
    are identical to `train_single_method` by construction, so a restored checkpoint
    always sees the object it trained in;
    `tests/test_config.py::test_agent_helper_matches_training_kwargs` pins this.
    """
    state_dim, action_dim = cfg["state_dim"], cfg["action_dim"]
    d_main = cfg["cost_limit"]
    shed_mapping = cfg.get("shed_mapping", "linear")
    reward_scale = grid_only_reward_scale(cfg, shed_mapping)
    target_entropy = cfg.get("target_entropy", -3.0)
    entropy_clamp = cfg.get("entropy_clamp", None)

    # PPO methods branch FIRST - "ppo_lagrangian" contains "lagrangian" and would
    # otherwise fall into the SAC constructor below.
    if "ppo" in method_name:
        from agents.ppo import PPOAgent, PPOPotentialShapingAgent
        kwargs = _ppo_kwargs(cfg, method_name)
        if cfg.get("residual_on_heuristic", False):
            kwargs["residual_policy"] = _ppo_residual_policy()
        cls = PPOPotentialShapingAgent if kwargs["pbrs"] else PPOAgent
        return cls(**kwargs)

    SACAgent, SACLagrangianAgent = _sac_agents()

    if "lagrangian" in method_name:
        return SACLagrangianAgent(
            state_dim=state_dim, action_dim=action_dim, cost_limit=d_main,
            lr_actor=cfg["lr_actor"], lr_critic=cfg["lr_critic"], lr_dual=cfg["lr_dual"],
            gamma=cfg["gamma"], tau=cfg["tau"], alpha_init=cfg["alpha_init"],
            lambda_init=cfg["lambda_init"], auto_entropy=cfg["auto_entropy"],
            target_entropy=target_entropy, batch_size=cfg["batch_size"],
            reward_scale=reward_scale, entropy_clamp=entropy_clamp,
            lambda_source=cfg.get("lambda_source", "episode_mean"),
            lambda_max=cfg.get("lambda_max", float("inf")))
    # NOTE: `SACAgent` (non-Lagrangian) deliberately does NOT take `reward_scale` here —
    # identical to the training path, so a restored checkpoint sees the object it trained in.
    # (The PPO branch above already returned, so the check below is SAC-only.)
    return SACAgent(
        state_dim=state_dim, action_dim=action_dim,
        lr_actor=cfg["lr_actor"], lr_critic=cfg["lr_critic"], gamma=cfg["gamma"],
        tau=cfg["tau"], alpha_init=cfg["alpha_init"], auto_entropy=cfg["auto_entropy"],
        target_entropy=target_entropy, batch_size=cfg["batch_size"],
        entropy_clamp=entropy_clamp)


def train_single_method(method_name: str, cfg: dict, ckpt_dir: str = None,
                        run_name: str = "", allow_overwrite: bool = False):
    run_id = clean_run_id(run_name)
    if ckpt_dir is None:
        _result_path, ckpt_dir = run_paths(method_name, cfg, run_id)
    # PPO methods take the on-policy path; SAC methods keep the off-policy path below.
    if "ppo" in method_name:
        return train_single_method_ppo(method_name, cfg, ckpt_dir=ckpt_dir,
                                       run_name=run_id,
                                       allow_overwrite=allow_overwrite)
    SACAgent, SACLagrangianAgent = _sac_agents()
    print(f"\n{'='*75}")
    print(f"RUN: {method_name.upper()} (Seed {cfg['seed']}, {cfg['total_steps']} Steps)")
    print(f"{'='*75}")

    set_seed(cfg["seed"])
    # shed-action mapping. "linear" = s=(a+1)/2 (no dead zone);
    # "deadzone" = s=max(0,a) so a~0 means no shed.
    shed_mapping = cfg.get("shed_mapping", "linear")
    env_train = SmartGridEnv({"split": "train", "seed": cfg["seed"], "shed_mapping": shed_mapping})
    # seed the gymnasium action_space RNG so warmup actions are reproducible.
    # set_seed() only sets the global numpy/torch seeds; the action_space has its own RNG.
    env_train.action_space.seed(cfg["seed"])
    env_val = SmartGridEnv({"split": "val", "seed": cfg["seed"] + 100})

    state_dim = cfg["state_dim"]
    action_dim = cfg["action_dim"]
    batch_size = cfg["batch_size"]
    total_steps = cfg["total_steps"]
    warmup_steps = cfg["warmup_steps"]
    eval_interval = cfg["eval_interval_steps"]
    d_main = cfg["cost_limit"]
    lr_dual = cfg["lr_dual"]

    # reward scale, entropy target/clamp, and the agent object itself are
    # resolved by the shared helpers below so evaluation cannot drift from the
    # training path. `reward_scale_mode: grid_only` recomputes c from the FULL
    # Train split (104 episodes) and never overwrites an explicit value.
    reward_scale = grid_only_reward_scale(cfg, shed_mapping, method_name=method_name)
    target_entropy = cfg.get("target_entropy", -3.0)
    entropy_clamp = cfg.get("entropy_clamp", None)

    is_lagrangian = "lagrangian" in method_name
    use_pot_shaping = "potential_shaping" in method_name
    use_fixed_penalty = "fixed_cost_penalty" in method_name

    # Probe 0 (P1b): warmup action mode. "uniform" = action_space.sample() (current);
    # "informed" = battery ~ U(-1,1), shed dims ~ small noise around the no-shed action (a=-1).
    warmup_mode = cfg.get("warmup_mode", "uniform")

    pot_shaping = PotentialBasedShaping(gamma=cfg["gamma"], kappa=cfg["potential_shaping_kappa"]) if use_pot_shaping else None
    cost_penalty = FixedCostPenalty(beta=cfg["fixed_cost_penalty_beta"]) if use_fixed_penalty else None

    if is_lagrangian:
        agent = SACLagrangianAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            cost_limit=d_main,
            lr_actor=cfg["lr_actor"],
            lr_critic=cfg["lr_critic"],
            lr_dual=lr_dual,
            gamma=cfg["gamma"],
            tau=cfg["tau"],
            alpha_init=cfg["alpha_init"],
            lambda_init=cfg["lambda_init"],
            auto_entropy=cfg["auto_entropy"],
            target_entropy=target_entropy,
            batch_size=batch_size,
            reward_scale=reward_scale,
            entropy_clamp=entropy_clamp,
            lambda_source=cfg.get("lambda_source", "episode_mean"),
            lambda_max=cfg.get("lambda_max", float("inf")),
        )
    else:
        agent = SACAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            lr_actor=cfg["lr_actor"],
            lr_critic=cfg["lr_critic"],
            gamma=cfg["gamma"],
            tau=cfg["tau"],
            alpha_init=cfg["alpha_init"],
            auto_entropy=cfg["auto_entropy"],
            target_entropy=target_entropy,
            batch_size=batch_size,
            entropy_clamp=entropy_clamp,
        )
    obs, _ = env_train.reset()
    start_time = time.time()
    t_chunk_prev = start_time

    wall_clock_history = {}
    eval_checkpoints = {}
    lambda_trajectory = []
    ep_costs = []
    ep_rewards = []
    ep_viols = []
    current_ep_costs = []
    current_ep_rewards = []
    current_ep_viols = 0

    # Running training metrics
    all_train_exceedances = []
    all_train_costs_eur = []

    # Degeneracy diagnostics
    actions_taken = []
    soc_history = []

    # One folder per agent, one file name per (experiment, seed): two experiments can
    # never write to the same artefact.
    ckpt_stem = "%s_seed%s" % (run_id, cfg["seed"])
    os.makedirs(ckpt_dir, exist_ok=True)
    final_ckpt_path = os.path.join(ckpt_dir, ckpt_stem + "_final.pt")
    if os.path.exists(final_ckpt_path) and not allow_overwrite:
        raise FileExistsError(
            f"refusing to overwrite {final_ckpt_path} — pass allow_overwrite=True "
            f"(or --overwrite) if the old run is really superseded")

    for step in range(1, total_steps + 1):
        if step <= warmup_steps:
            if warmup_mode == "informed":
                # P1b: battery ~ U(-1,1); shed dims near no-shed (a=-1) + small noise.
                action = np.full(6, -1.0, dtype=np.float32)
                action[0] = float(np.random.uniform(-1.0, 1.0))
                action[1:] = np.clip(-1.0 + np.random.normal(0.0, 0.1, 5), -1.0, 1.0)
            else:
                action = env_train.action_space.sample()
        else:
            action = agent.select_action(obs, evaluate=False)

        next_obs, reward, terminated, truncated, info = env_train.step(action)
        # the Lagrangian cost is the GRID EXCEEDANCE IN kW only.
        cost = float(info["grid_exceedance_kw"])
        imp = float(info.get("grid_import_kw", 0.0))
        exc = max(0.0, imp - env_train.grid_import_cap_kw)

        all_train_exceedances.append(exc)
        step_cost_eur = info["energy_cost"] + info["unserved_cost"] + info["battery_wear_cost"]
        all_train_costs_eur.append(step_cost_eur)

        # Track degeneracy signals
        actions_taken.append(action)
        soc_history.append(float(next_obs[0]))

        current_ep_costs.append(cost)
        current_ep_rewards.append(reward)
        if imp > env_train.grid_import_cap_kw:
            current_ep_viols += 1

        # Reward modification
        step_reward = reward
        if use_pot_shaping:
            # F = gamma * Phi(s') - Phi(s). Phi is NOT zeroed on truncation!
            f_t = pot_shaping.shape(obs, next_obs)
            step_reward += f_t
        elif use_fixed_penalty:
            pen_t = cost_penalty.penalty(cost)
            step_reward += pen_t

        agent.store_transition(obs, action, step_reward, next_obs, terminated, truncated, cost=cost)

        obs = next_obs

        # Episode boundary handling
        if truncated:
            ep_mean_cost = float(np.mean(current_ep_costs))
            ep_costs.append(ep_mean_cost)
            ep_rewards.append(float(np.sum(current_ep_rewards)))
            ep_viols.append(current_ep_viols)

            if is_lagrangian:
                agent.record_episode_cost(ep_mean_cost)

            current_ep_costs = []
            current_ep_rewards = []
            current_ep_viols = 0
            obs, _ = env_train.reset()

        # Update gradient
        if step > warmup_steps:
            metrics = agent.update()
            if is_lagrangian and step % 1000 == 0:
                lambda_trajectory.append({"step": step, "lambda": agent.lam, "J_c": metrics.get("J_c", 0.0)})

        # Wall clock checkpoint & validation evaluation
        if step % eval_interval == 0:
            now = time.time()
            elapsed_chunk = now - t_chunk_prev
            t_chunk_prev = now
            rate = eval_interval / elapsed_chunk
            wall_clock_history[f"{step}_steps"] = {
                "chunk_time_sec": elapsed_chunk,
                "cumulative_time_sec": now - start_time,
                "steps_per_sec": rate,
            }

            # Evaluate deterministically on Validation (26 weeks)
            val_metrics = evaluate_deterministic_val(env_val, agent, agent.obs_normalizer)
            eval_checkpoints[f"{step}_steps"] = val_metrics

            # A checkpoint per evaluation interval: a cancelled run keeps its weights.
            ckpt_path = os.path.join(ckpt_dir, f"{ckpt_stem}_step{step}.pt")
            agent.save_models(ckpt_path)

            lam_info = f" | lambda={agent.lam:.4f}" if is_lagrangian else ""
            print(f"[{method_name}] Step {step:6d}/{total_steps} | Rate: {rate:5.1f} steps/s{lam_info} | Val Cost: {val_metrics['total_cost_eur']:8.2f} EUR | Val Viol: {val_metrics['violation_rate_pct']:4.2f}% | Val Exc: {val_metrics['mean_exceedance_kw']:6.4f} kW", flush=True)

    total_wall_clock = time.time() - start_time
    avg_steps_per_sec = total_steps / total_wall_clock

    # Identify Best Checkpoint on Validation.
    # Best checkpoint criteria: minimum total cost among candidates with
    # mean_exceedance <= d_main (or lowest exceedance if none).
    if not eval_checkpoints and eval_interval <= total_steps:
        raise RuntimeError(
            f"No evaluation was recorded for {method_name}: total_steps={total_steps} "
            f"< eval_interval={eval_interval}. Lower eval_interval_steps or raise total_steps."
        )
    feasible_ckpts = [
        (k, v) for k, v in eval_checkpoints.items()
        if v["mean_exceedance_kw"] <= d_main + 1e-4
    ]
    if feasible_ckpts:
        best_ckpt_key, best_ckpt_val = min(feasible_ckpts, key=lambda x: x[1]["total_cost_eur"])
    else:
        best_ckpt_key, best_ckpt_val = min(eval_checkpoints.items(), key=lambda x: (x[1]["mean_exceedance_kw"], x[1]["total_cost_eur"]))

    final_ckpt_val = eval_checkpoints[f"{total_steps}_steps"]

    # Training-time metrics
    train_mean_exceedance = float(np.mean(all_train_exceedances))
    train_total_cost_eur = float(np.sum(all_train_costs_eur))
    train_violation_rate_pct = float(np.mean(np.array(all_train_exceedances) > 0.0)) * 100.0

    # Lambda summary
    if is_lagrangian:
        lam_start = lambda_trajectory[0]["lambda"] if lambda_trajectory else 0.0
        mid_idx = len(lambda_trajectory) // 2
        lam_mid = lambda_trajectory[mid_idx]["lambda"] if lambda_trajectory else 0.0
        lam_end = agent.lam
        # Check oscillation vs settling: std of lambda over last 20 measurements
        recent_lams = [x["lambda"] for x in lambda_trajectory[-20:]] if len(lambda_trajectory) >= 20 else [lam_end]
        lam_std = float(np.std(recent_lams))
        lam_status = "settled" if lam_std < 0.02 else ("converging" if lam_std < 0.10 else "oscillating")
    else:
        lam_start, lam_mid, lam_end, lam_status = None, None, None, "N/A (unconstrained/penalty)"

    # Degeneracy diagnostics
    actions_arr = np.array(actions_taken[warmup_steps:])
    soc_arr = np.array(soc_history[warmup_steps:])
    mean_bat_cmd = float(np.mean(actions_arr[:, 0]))
    mean_shed_cmd = float(np.mean(actions_arr[:, 1:]))
    soc_at_bounds_pct = float(np.mean((soc_arr <= 0.205) | (soc_arr >= 0.895))) * 100.0

    degenerate_flags = []
    if np.all(actions_arr[:, 0] <= 0.01):
        degenerate_flags.append("Never charges battery")
    if np.all(actions_arr[:, 1:] < -0.95):
        degenerate_flags.append("Never sheds load")
    elif np.all(actions_arr[:, 1:] > 0.90):
        degenerate_flags.append("Sheds everything continuously")
    if soc_at_bounds_pct > 90.0:
        degenerate_flags.append("Runs SoC constantly pinned at bounds")

    is_degenerate = len(degenerate_flags) > 0
    degeneracy_summary = "; ".join(degenerate_flags) if is_degenerate else "None (healthy dynamic dispatch)"

    # Decision 10: the FINAL policy is the primary result; best-by-validation is secondary
    # and is reported only to expose how much checkpoint selection hides.
    agent.save_models(final_ckpt_path)

    return {
        "method": method_name,
        "agent": variant_label(method_name, cfg),
        "run_id": run_id,
        "seed": cfg["seed"],
        "total_steps": total_steps,
        # the RESOLVED config actually used, including the reward scale c
        # that survived the grid_only recompute, alpha0, target entropy,
        # clamp, mapping, warmup mode, thread count and seed. Read this, not the YAML.
        "resolved_config": dict(cfg, reward_scale_effective=reward_scale),
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
        # NOTE: the dict key `final_checkpoint` above already carries the metrics;
        # this repeat was a silent Python dict-overwrite (F601). Kept under the name
        # `final_metrics` so both survive; consumers reading `final_checkpoint`
        # (the metrics dict itself) are unaffected.
        "final_metrics": final_ckpt_val,
        "training_time_metrics": {
            "total_cost_eur": train_total_cost_eur,
            "mean_exceedance_kw": train_mean_exceedance,
            "violation_rate_pct": train_violation_rate_pct,
            "cost_limit_d_main": d_main,
            "satisfies_d_main_on_train": train_mean_exceedance <= d_main,
        },
        "lambda_summary": {
            "lambda_start": lam_start,
            "lambda_mid": lam_mid,
            "lambda_end": lam_end,
            "status": lam_status,
            # The FULL trajectory (one point per dual update). It used to live in a
            # separate partial-results file; it is part of the run result now, so
            # one file holds everything a run produced.
            "trajectory": lambda_trajectory,
        },
        "degeneracy_check": {
            "is_degenerate": is_degenerate,
            "summary": degeneracy_summary,
            "mean_battery_cmd": mean_bat_cmd,
            "mean_shed_cmd": mean_shed_cmd,
            "soc_pinned_bounds_pct": soc_at_bounds_pct,
        }
    }


def _ppo_kwargs(cfg: dict, method_name: str) -> dict:
    """Single source of truth for PPOAgent constructor kwargs.

    Used by BOTH `train_single_method_ppo` and `load_agent_for_method`, so the S3 Test
    evaluation cannot replay a differently-configured agent. `residual_policy` (the
    live PSM' object, r3 only) is attached by the caller, never stored here.
    """
    return dict(
        state_dim=cfg["state_dim"], action_dim=cfg["action_dim"],
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
    )


def _ppo_residual_policy():
    """The PSM' heuristic the residual agent corrects (read-only base policy)."""
    from agents.heuristics import psm_prime_policy

    return psm_prime_policy()


def train_single_method_ppo(method_name: str, cfg: dict, ckpt_dir: str = None,
                            run_name: str = "", allow_overwrite: bool = False):
    """PPO trainer: on-policy rollouts, GAE, clipped updates, deterministic Val.

    Same result schema as the SAC path (final/best checkpoints, lambda_summary,
    degeneracy_check, resolved_config) so every reader works unchanged. PPO needs no
    warmup: the policy mean starts exactly at the Grid-Only action (test_ppo.py §5).
    """
    run_id = clean_run_id(run_name)
    if ckpt_dir is None:
        _result_path, ckpt_dir = run_paths(method_name, cfg, run_id)
    print(f"\n{'='*75}")
    print(f"PPO RUN: {method_name.upper()} (Seed {cfg['seed']}, {cfg['total_steps']} Steps)")
    print(f"{'='*75}")

    from agents.ppo import PPOAgent, PPOPotentialShapingAgent

    set_seed(cfg["seed"])
    shed_mapping = cfg.get("shed_mapping", "linear")
    env_train = SmartGridEnv({"split": "train", "seed": cfg["seed"], "shed_mapping": shed_mapping})
    env_train.action_space.seed(cfg["seed"])
    env_val = SmartGridEnv({"split": "val", "seed": cfg["seed"] + 100})

    # shielded runs wrap the env OUTSIDE its physics (CapShield only sheds the
    # minimum excess over C; the SoC floor stays unshielded). The learned action is
    # then 1-D (battery); CapShield.step is a drop-in replacement.
    shielded = bool(cfg.get("shielded", False))
    if shielded:
        from env.cap_shield import CapShield
        env_train = CapShield(env_train)
        env_val = CapShield(env_val)

    total_steps = cfg["total_steps"]
    eval_interval = cfg["eval_interval_steps"]
    rollout_len = cfg.get("rollout_steps", 2048)
    d_main = cfg["cost_limit"]
    is_lagrangian = "lagrangian" in method_name

    kwargs = _ppo_kwargs(cfg, method_name)
    residual_on = bool(cfg.get("residual_on_heuristic", False))
    if residual_on:
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
    t_chunk_prev = start_time
    last_term = False
    step = 0
    # eval fires when the step counter CROSSES a boundary, not on exact multiples -
    # rollout segments (2048 steps) never land exactly on 50k.
    next_eval = eval_interval
    while step < total_steps:
        # ---- collect one rollout ----
        for _ in range(min(rollout_len, total_steps - step)):
            step += 1
            action, logp, value, cost_value = agent.select_action(obs, evaluate=False)
            next_obs, reward, terminated, truncated, info = env_train.step(action)
            cost = float(info["grid_exceedance_kw"])
            imp = float(info.get("grid_import_kw", 0.0))
            exc = max(0.0, imp - env_train.grid_import_cap_kw)
            all_train_exceedances.append(exc)
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
                    lam = agent.update_lambda_from_source()
                    lambda_trajectory.append({"step": step, "lambda": lam,
                                              "J_c": float(np.mean(agent.cost_history[-1:])) if agent.cost_history else 0.0})
                obs, _ = env_train.reset()

        # ---- update on the rollout (bootstrap unless the segment ended terminal) ----
        with th.no_grad():
            s_last = agent._t(agent.obs_normalizer.normalize(obs[None, :]))
            last_v = float(agent.value_net(s_last))
            last_cv = float(agent.cost_net(s_last)) if agent.cost_net is not None else 0.0
        batch = agent.finish_path(0.0 if last_term else last_v,
                                  0.0 if (last_term and is_lagrangian) else last_cv)
        update_diag = agent.update(batch) if batch is not None else {}
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
            elapsed_chunk = now - t_chunk_prev
            t_chunk_prev = now
            wall_clock_history[f"{step_tag}_steps"] = {
                "chunk_time_sec": elapsed_chunk,
                "cumulative_time_sec": now - start_time,
                "steps_per_sec": step_tag / max(now - start_time, 1e-9),
            }
            extra = None
            if is_lagrangian:
                mean_c = float(np.mean(agent.cost_history[-10:])) if agent.cost_history else 0.0
                extra = {"lambda": agent.lambda_, "mean_ep_cost": mean_c}
            val_metrics = evaluate_deterministic_val(env_val, agent, agent.obs_normalizer,
                                                     extra_log=extra)
            eval_checkpoints[f"{step_tag}_steps"] = val_metrics
            # A checkpoint per evaluation interval: a cancelled run keeps its weights.
            ckpt_path = os.path.join(ckpt_dir, f"{ckpt_stem}_step{step_tag}.pt")
            agent.save_models(ckpt_path)
            _ = update_diag
            if step_tag >= next_eval:
                next_eval += eval_interval

    total_wall_clock = time.time() - start_time
    avg_steps_per_sec = total_steps / total_wall_clock

    if not eval_checkpoints and eval_interval <= total_steps:
        raise RuntimeError(f"No evaluation was recorded for {method_name}")
    feasible_ckpts = [(k, v) for k, v in eval_checkpoints.items()
                      if v["mean_exceedance_kw"] <= d_main + 1e-4]
    if feasible_ckpts:
        best_ckpt_key, best_ckpt_val = min(feasible_ckpts, key=lambda x: x[1]["total_cost_eur"])
    else:
        best_ckpt_key, best_ckpt_val = min(eval_checkpoints.items(), key=lambda x: (x[1]["mean_exceedance_kw"], x[1]["total_cost_eur"]))
    final_ckpt_val = eval_checkpoints[f"{total_steps}_steps"]

    train_mean_exceedance = float(np.mean(all_train_exceedances))
    train_total_cost_eur = float(np.sum(all_train_costs_eur))
    train_violation_rate_pct = float(np.mean(np.array(all_train_exceedances) > 0.0)) * 100.0

    if is_lagrangian:
        lam_start = lambda_trajectory[0]["lambda"] if lambda_trajectory else 0.0
        lam_mid = lambda_trajectory[len(lambda_trajectory) // 2]["lambda"] if lambda_trajectory else 0.0
        lam_end = agent.lambda_
        recent_lams = [x["lambda"] for x in lambda_trajectory[-20:]] if len(lambda_trajectory) >= 20 else [lam_end]
        lam_std = float(np.std(recent_lams))
        lam_status = "settled" if lam_std < 0.02 else ("converging" if lam_std < 0.10 else "oscillating")
    else:
        lam_start, lam_mid, lam_end, lam_status = None, None, None, "N/A (unconstrained/penalty)"

    actions_arr = np.array(actions_taken)
    soc_arr = np.array(soc_history)
    mean_bat_cmd = float(np.mean(actions_arr[:, 0]))
    # 1-D shielded actions have no shed dims.
    mean_shed_cmd = float(np.mean(actions_arr[:, 1:])) if actions_arr.shape[1] > 1 else 0.0
    soc_at_bounds_pct = float(np.mean((soc_arr <= 0.205) | (soc_arr >= 0.895))) * 100.0
    degenerate_flags = []
    if np.all(actions_arr[:, 0] <= 0.01):
        degenerate_flags.append("Never charges battery")
    if actions_arr.shape[1] > 1 and np.all(actions_arr[:, 1:] < -0.95):
        degenerate_flags.append("Never sheds load")
    elif actions_arr.shape[1] > 1 and np.all(actions_arr[:, 1:] > 0.90):
        degenerate_flags.append("Sheds everything continuously")
    if soc_at_bounds_pct > 90.0:
        degenerate_flags.append("Runs SoC constantly pinned at bounds")
    is_degenerate = len(degenerate_flags) > 0
    degeneracy_summary = "; ".join(degenerate_flags) if is_degenerate else "None (healthy dynamic dispatch)"

    agent.save_models(final_ckpt_path)

    return {
        "method": method_name,
        "agent": variant_label(method_name, cfg),
        "run_id": run_id,
        "seed": cfg["seed"],
        "total_steps": total_steps,
        "resolved_config": dict(cfg, reward_scale_effective=cfg.get("reward_scale", 1.0)),
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
            "cost_limit_d_main": d_main,
            "satisfies_d_main_on_train": train_mean_exceedance <= d_main,
        },
        "lambda_summary": {
            "lambda_start": lam_start,
            "lambda_mid": lam_mid,
            "lambda_end": lam_end,
            "status": lam_status,
            # The FULL trajectory (one point per dual update). It used to live in a
            # separate partial-results file; it is part of the run result now, so
            # one file holds everything a run produced.
            "trajectory": lambda_trajectory,
        },
        "degeneracy_check": {
            "is_degenerate": is_degenerate,
            "summary": degeneracy_summary,
            "mean_battery_cmd": mean_bat_cmd,
            "mean_shed_cmd": mean_shed_cmd,
            "soc_pinned_bounds_pct": soc_at_bounds_pct,
        }
    }


def clean_run_id(run_name: str) -> str:
    """A run id with no dangling separators: "s3_ppo_" -> "s3_ppo"."""
    return (run_name or "").strip().strip("_").strip()


def resolve_run_name(config_arg: str, run_name_arg: str) -> str:
    """The run id: the experiment id from the config, plus an optional extra name.

    The run name always carries the experiment id. r2 and r3
    shared method (ppo_lagrangian) + seed (42) + run-name ("s1c_"), so r3
    overwrote r2's result JSON and checkpoints. When --config names an experiment id
    (rather than a file path), the id comes first: ("r2", "s1c") -> "r2_s1c".
    A run name that already IS the experiment id is not repeated, and no dangling
    separator is left behind. `test_run_names_carry_experiment_id` pins this.
    """
    name = clean_run_id(run_name_arg)
    stem = os.path.splitext(os.path.basename(str(config_arg)))[0]
    if resolve_config_path(config_arg):
        # A config FILE: the explicit run name wins (old behaviour); without one the
        # file stem identifies the run ("base").
        return name or stem
    exp_id = stem
    if not name or name == exp_id or exp_id.endswith(name) or name.endswith(exp_id):
        return exp_id
    return f"{exp_id}_{name}"


def resolve_config_path(path):
    """Absolute path of a config file, or None when `path` names an experiment id.

    A relative path is accepted both against the cwd and against the repo root, so the
    CLIs work from any working directory.
    """
    for cand in (path, os.path.abspath(str(path)), os.path.join(REPO_ROOT, str(path))):
        if cand and os.path.exists(cand):
            return os.path.abspath(cand)
    return None


def load_config_with_base(path):
    """Load config from configs/config.yaml (defaults + experiments).

    `path` may be an experiment id ("s3_ppo"), the path of a YAML file, or the path of
    configs/config.yaml itself (which yields the defaults). The config store is
    resolved relative to the repo root, so this works from any working directory.
    """
    def load_yaml(p2):
        with open(p2) as f:
            return yaml.safe_load(f) or {}

    store = load_yaml(CONFIG_FILE)
    base = store.get("defaults") or {}
    experiments = store.get("experiments") or {}

    # A path may be given relative to the cwd or to the repo root; try both.
    found = resolve_config_path(path)

    if found:
        with open(found) as f:
            loaded = yaml.safe_load(f) or {}
        label = str(found)
        # the merged file itself: its `defaults` block is the base
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
    _m = override.get("method")
    if _m and "lagrangian" not in str(_m):
        _bad = sorted(k for k in override if k in LAG_ONLY)
        if _bad and label not in LEGACY_EXEMPT:
            raise ValueError(f"{label}: Lagrangian-only key(s) {_bad} under method {_m!r}")
    cfg = dict(base)
    cfg.update(override)
    return cfg


def main():
    # `main()` is retained only so old invocations of this module still run.
    # The canonical entry point is `experiments/train.py --config configs/config.yaml`.
    config_path = CONFIG_FILE
    cfg = load_config_with_base(config_path)

    all_methods = [
        "sac_lagrangian",
        "sac_fixed_cost_penalty",
        "sac_lagrangian_potential_shaping",
        "sac_potential_shaping",
        "plain_sac",
    ]
    # CLI: python experiments/train_core.py [--methods a,b] [--steps N] [--seed S] [--run-name NAME]
    methods = all_methods
    steps_override = None
    seed_override = None
    run_name = ""
    argv = sys.argv[1:]
    if "--methods" in argv:
        methods = argv[argv.index("--methods") + 1].split(",")
        assert all(m in all_methods for m in methods), f"unknown method in {methods}"
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
    for m in methods:
        t0 = time.time()
        res = train_single_method(m, cfg, run_name=run_id)
        results[m] = res
        # One file per method, written as soon as that method finishes.
        per_method_path, _ckpt_dir = run_paths(m, cfg, run_id)
        os.makedirs(os.path.dirname(per_method_path), exist_ok=True)
        with open(per_method_path, "w") as f:
            json.dump(res, f, indent=2)
        print(f"Finished {m} in {time.time() - t0:.1f}s -> {per_method_path}", flush=True)

    print("\n" + "=" * 75)
    print(f"COMPLETE: {len(results)} runs -> {RESULTS_DIR}")
    print("=" * 75)


if __name__ == "__main__":
    main()
