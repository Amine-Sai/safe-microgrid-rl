import os
from typing import Optional

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from agents.running_stat import RunningMeanStd


def grid_only_action(shed_mapping: str, action_dim: int = 6) -> np.ndarray:
    """The action that reproduces Grid-Only, derived from the env's mapping rule.

    env/smartgrid_env.py::step parses the action as
        relu/deadzone : shed = max(0, a)          -> a <= 0 sheds nothing
        linear        : shed = (a + 1) / 2        -> a = -1   sheds nothing
    and battery_cmd = action[0], so 0.0 means the battery is idle. Grid-Only is the
    battery idle plus no shedding, i.e. [0.0, a_noshed, ...]. This function mirrors
    that rule so a change in the env cannot silently leave the init stale.
    """
    if shed_mapping in ("relu", "deadzone"):
        no_shed = 0.0
    else:
        no_shed = -1.0
    a = np.full(action_dim, no_shed, dtype=np.float32)
    a[0] = 0.0
    return a


def compute_gae(rewards, values, dones, truncated, last_value, gamma=0.99, lam=0.95):
    """GAE(lambda) with truncation-aware bootstrapping.

    `dones` must mark terminal transitions only. A truncation (time limit) still
    bootstraps from `last_value`, which is why it is passed in separately and never
    zeroed.

    Returns (advantages, returns), both plain numpy arrays of length T.
    """
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float64)
    gae = 0.0
    for t in reversed(range(T)):
        not_terminal = 0.0 if dones[t] else 1.0
        next_value = last_value if t == T - 1 else values[t + 1]
        # a truncation does not discard the bootstrap
        if truncated[t] and not (t == T - 1):
            next_value = values[t + 1] if not dones[t] else 0.0
        delta = rewards[t] + gamma * next_value * not_terminal - values[t]
        gae = delta + gamma * lam * not_terminal * gae
        adv[t] = gae
    returns = adv + np.asarray(values, dtype=np.float64)
    return adv, returns


class _PolicyNet(nn.Module):
    """Gaussian policy: state-independent log-std, clipped (not squashed) actions."""

    def __init__(self, state_dim, action_dim, hidden_dims=(64, 64), init_mean=None,
                 init_log_std=None):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.mu = nn.Linear(hidden_dims[1], action_dim)
        ls = th.full((action_dim,), -1.0, dtype=th.float32)
        if init_log_std is not None:
            ls = th.as_tensor(init_log_std, dtype=th.float32).clone()
        self.log_std = nn.Parameter(ls)
        if init_mean is not None:
            # zero-init the head so the policy mean starts exactly at init_mean
            self.mu.weight.data.zero_()
            self.mu.bias.data.copy_(th.as_tensor(init_mean, dtype=th.float32))

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.mu(x)

    def dist(self, state):
        mu = self.forward(state)
        std = th.exp(self.log_std)
        return th.distributions.Normal(mu, std)


class _ValueNet(nn.Module):
    def __init__(self, state_dim, hidden_dims=(64, 64)):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.v = nn.Linear(hidden_dims[1], 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.v(x)


class PPOAgent:
    """PPO, with optional Lagrangian dual ascent and potential-based reward shaping."""

    def __init__(
        self,
        state_dim: int = 17,
        action_dim: int = 6,
        shed_mapping: str = "linear",
        lr_policy: float = 3e-4,
        lr_value: float = 1e-3,
        lr_dual: float = 0.01,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        entropy_coef: float = 0.0,
        target_kl: float = 0.05,
        n_epochs: int = 10,
        batch_size: int = 64,
        reward_scale: float = 1.0,
        lagrangian: bool = False,
        pbrs: bool = False,
        cost_limit: float = 0.05,          # d_main, kW/step
        lambda_init: float = 0.0,
        lambda_lr_mult: float = 0.1,       # the dual runs slower than the policy
        potential_shaping_kappa: float = 105.0,
        normalize_advantage: bool = True,
        lag_normalize_by_1p_lam: bool = True,
        max_grad_norm: float = 0.5,
        residual_policy=None,      # residual agents only: fixed base policy with get_action(obs)
        residual_coef: float = 0.25,
        use_minibatches: bool = False,     # default = legacy full-batch
        value_epochs: int = 1,             # default = legacy 1 GD step
        store_raw_action: bool = False,    # default = legacy clip(u) storage
        init_log_std: float = -1.0,        # default = legacy
        init_log_std_shed=None,            # override for shed dims (indices 1:)
        device: Optional[str] = None,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.target_kl = target_kl
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.reward_scale = reward_scale
        self.lagrangian = lagrangian
        self.pbrs = pbrs
        self.cost_limit = cost_limit
        self.normalize_advantage = normalize_advantage
        self.lag_normalize_by_1p_lam = lag_normalize_by_1p_lam
        self.max_grad_norm = max_grad_norm
        self.residual_policy = residual_policy
        self.residual_coef = residual_coef
        self.use_minibatches = use_minibatches
        self.value_epochs = int(value_epochs)
        self.store_raw_action = store_raw_action
        self._last_u = None
        self._rng = np.random.RandomState(0)   # trainer overwrites with the run seed
        self.device = th.device(device or "cpu")

        # log-std init (default -1.0). The shed dims (indices 1:) can take a
        # separate init, which limits how much load random exploration sheds early
        # in training.
        ls = np.full(action_dim, init_log_std, dtype=np.float32)
        if init_log_std_shed is not None and action_dim > 1:
            ls[1:] = init_log_std_shed

        # Zero-init the policy head so the mean starts at init_mean (Grid-Only), or
        # at a zero correction when a residual base policy is present.
        init_mean = (np.zeros(action_dim, dtype=np.float32)
                     if residual_policy is not None
                     else grid_only_action(shed_mapping, action_dim))
        self.policy = _PolicyNet(state_dim, action_dim, init_mean=init_mean,
                                 init_log_std=ls).to(self.device)
        self.value_net = _ValueNet(state_dim).to(self.device)
        self.cost_net = _ValueNet(state_dim).to(self.device) if lagrangian else None

        self.policy_optimizer = th.optim.Adam(self.policy.parameters(), lr=lr_policy)
        self.value_optimizer = th.optim.Adam(self.value_net.parameters(), lr=lr_value)
        if self.cost_net is not None:
            self.cost_optimizer = th.optim.Adam(self.cost_net.parameters(), lr=lr_value)

        # dual runs on its own, slower timescale
        self.lambda_ = float(max(0.0, lambda_init))
        self.lambda_lr = lr_dual * lambda_lr_mult
        self.lambda_history = []
        self.cost_history = []

        # potential-based reward shaping state
        self.kappa = potential_shaping_kappa if pbrs else 0.0

        self.obs_normalizer = RunningMeanStd(shape=(state_dim,))
        # Value-target normalisation. The critic was fitting raw return magnitudes
        # with one gradient step per rollout and learned only the mean (EV -0.02 on
        # same-scale Monte-Carlo returns). Targets are normalised with a running
        # mean/std and the net output is denormalised at inference, so every stored
        # value and bootstrap stays in true reward units and GAE stays consistent.
        self.ret_normalizer = RunningMeanStd(shape=())
        self.cost_normalizer = RunningMeanStd(shape=())
        self._reset_buffer()

    # ---------------------------------------------------------------- buffer
    def _reset_buffer(self):
        self.buf = {k: [] for k in ("s", "a", "r", "c", "v", "cv", "term", "trunc", "logp")}
        self.rollout_costs = []

    def store_transition(self, state, action, reward, cost, value, cost_value,
                         log_prob, terminated, truncated):
        b = self.buf
        b["s"].append(np.asarray(state, dtype=np.float32))
        raw = self._last_u if (self.store_raw_action and self._last_u is not None) else action
        b["a"].append(np.asarray(raw, dtype=np.float32))
        b["r"].append(float(reward))
        b["c"].append(float(cost))
        b["v"].append(float(value))
        b["cv"].append(float(cost_value))
        b["logp"].append(float(log_prob))
        b["term"].append(bool(terminated))
        b["trunc"].append(bool(truncated))
        self.rollout_costs.append(float(cost))

    # ---------------------------------------------------------------- acting
    def _t(self, x):
        return th.as_tensor(x, dtype=th.float32, device=self.device)

    def _denorm_v(self, v):
        """Denormalise a value-net output into true reward units."""
        std = float(np.sqrt(self.ret_normalizer.var) + self.ret_normalizer.epsilon)
        return v * std + float(self.ret_normalizer.mean)

    def _denorm_c(self, v):
        std = float(np.sqrt(self.cost_normalizer.var) + self.cost_normalizer.epsilon)
        return v * std + float(self.cost_normalizer.mean)

    def select_action(self, state: np.ndarray, evaluate: bool = False):
        """Returns (action, log_prob, value, cost_value).

        With `residual_policy` set (r3 only): the policy output u is a CORRECTION and the
        executed action is a = clip(a_heuristic(obs) + residual_coef * u). The policy is
        zero-init, so training starts exactly at the heuristic.

        The observation normalizer is updated ONLY when acting for
        training (evaluate=False). Updating it during evaluation drifts the restored
        statistics and broke Val reproduction to the cent (45,258.10 vs 45,258.17).
        """
        if not evaluate:
            self.obs_normalizer.update(state[None, :])
        s = self._t(self.obs_normalizer.normalize(state[None, :]))
        with th.no_grad():
            if evaluate:
                u = self.policy.forward(s)
            else:
                d = self.policy.dist(s)
                u = d.sample()
            if self.residual_policy is not None:
                base = self.residual_policy.get_action(state)
                a = th.clamp(self._t(base) + self.residual_coef * u, -1.0, 1.0)
                # log-prob is under the residual distribution (base is deterministic
                # given obs, so this is exact)
                logp = self.policy.dist(s).log_prob(u).sum(dim=-1)
            else:
                a = th.clamp(u, -1.0, 1.0)
                if evaluate:
                    logp = th.zeros((s.shape[0],), device=self.device)
                else:
                    logp = self.policy.dist(s).log_prob(u).sum(dim=-1)
            v = self._denorm_v(self.value_net(s))
            cv_raw = self.cost_net(s) if self.cost_net is not None else th.zeros_like(v)
            cv = self._denorm_c(cv_raw) if self.cost_net is not None else cv_raw
        if not evaluate:
            # Keep the pre-clip sample u. With store_raw_action the buffer stores
            # this u, so the PPO ratio log_pi(u)-log_pi_old(u) matches the log-prob
            # recorded at acting time (the legacy path stored clip(u) instead).
            self._last_u = u.squeeze(0).cpu().numpy().astype(np.float32)
        return (a.squeeze(0).cpu().numpy().astype(np.float32),
                float(logp), float(v), float(cv))

    # ---------------------------------------------------------------- shapes
    def shape(self, reward, state, next_state, terminated, truncated):
        """Apply the learning-time reward transform.

        The explicit reward_scale c multiplies EVERY learning target (the directive
        uses c = 0.096879). With pbrs the potential shaping is added on top; Phi is
        state-only and is NOT zeroed on truncation (only on true termination).
        """
        scaled = float(reward) * self.reward_scale
        if not self.pbrs:
            return scaled
        phi_s = self._phi(state)
        phi_ns = 0.0 if terminated else self._phi(next_state)
        return scaled + self.kappa * (self.gamma * phi_ns - phi_s)

    def _phi(self, state):
        """Potential function for PBRS. Overridden by PPOPotentialShapingAgent."""
        return 0.0

    # ---------------------------------------------------------------- learn
    def finish_path(self, last_value, last_cost_value=0.0):
        """Compute GAE for the buffered rollout and return the training batch.

        `last_value` bootstraps from the value of the final observation. It is ALWAYS
        used when the episode ended by truncation, and only zeroed on true termination.
        """
        b = self.buf
        T = len(b["r"])
        if T == 0:
            return None
        adv_r, ret_r = compute_gae(b["r"], b["v"], b["term"], b["trunc"], last_value,
                                   self.gamma, self.gae_lambda)
        out = {
            "states": np.asarray(b["s"], dtype=np.float32),
            "actions": np.asarray(b["a"], dtype=np.float32),
            "logp": np.asarray(b["logp"], dtype=np.float32),
            "adv_r": adv_r, "ret_r": ret_r,
        }
        if self.lagrangian:
            adv_c, ret_c = compute_gae(b["c"], b["cv"], b["term"], b["trunc"],
                                        last_cost_value, self.gamma, self.gae_lambda)
            out.update(adv_c=adv_c, ret_c=ret_c)
        return out

    def update_lambda_from_source(self):
        """Projected ascent on lambda, on a slower timescale than the policy.

        J_c = mean UNDISCOUNTED per-step cost over recent episodes.
        lambda <- max(0, lambda + lr_dual * (J_c - d)).
        """
        if not self.lagrangian:
            return self.lambda_
        # end_episode() owns cost_history (one entry per episode); read it, don't append.
        recent = self.cost_history[-10:] if self.cost_history else self.rollout_costs
        j_c = float(np.mean(recent)) if recent else 0.0
        self.lambda_ = max(0.0, self.lambda_ + self.lambda_lr * (j_c - self.cost_limit))
        self.lambda_history.append(self.lambda_)
        return self.lambda_

    def _minibatches(self, batch):
        n = len(batch["states"])
        idx = np.arange(n)
        rng = np.random.RandomState(0) if not hasattr(self, "_rng") else self._rng
        rng.shuffle(idx)
        for start in range(0, n, self.batch_size):
            sel = idx[start:start + self.batch_size]
            yield {k: v[sel] for k, v in batch.items()}

    # ---------------------------------------------------- minibatched fits
    def _fit_critics_mb(self, s_t, ret_norm, retc_norm):
        """Minibatched critic epochs. Returns (last value loss, gradient steps)."""
        n, steps, v_loss = s_t.shape[0], 0, None
        for _ in range(self.value_epochs):
            perm = th.as_tensor(self._rng.permutation(n), device=self.device)
            for st in range(0, n, self.batch_size):
                i = perm[st:st + self.batch_size]
                v_loss = F.mse_loss(self.value_net(s_t[i]), ret_norm[i])
                self.value_optimizer.zero_grad()
                v_loss.backward()
                nn.utils.clip_grad_norm_(self.value_net.parameters(), self.max_grad_norm)
                self.value_optimizer.step()
                if retc_norm is not None:
                    c_loss = F.mse_loss(self.cost_net(s_t[i]), retc_norm[i])
                    self.cost_optimizer.zero_grad()
                    c_loss.backward()
                    nn.utils.clip_grad_norm_(self.cost_net.parameters(), self.max_grad_norm)
                    self.cost_optimizer.step()
                steps += 1
        return float(v_loss.detach()), steps

    def _policy_epochs_mb(self, s_t, a_t, old_logp, adv_t):
        """Minibatched policy epochs with an epoch-end full-batch KL early stop.

        Returns (loss, entropy, KL, clip-binding fraction, gradient steps).
        """
        n, steps, loss_v = s_t.shape[0], 0, 0.0
        ent = kl = clipf = 0.0
        for _ in range(self.n_epochs):
            perm = th.as_tensor(self._rng.permutation(n), device=self.device)
            for st in range(0, n, self.batch_size):
                i = perm[st:st + self.batch_size]
                d = self.policy.dist(s_t[i])
                logp = d.log_prob(a_t[i]).sum(dim=-1, keepdim=True)
                ratio = th.exp(logp - old_logp[i])
                ai = adv_t[i]
                surr = th.min(ratio * ai,
                              th.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * ai)
                loss = -surr.mean() - self.entropy_coef * d.entropy().mean()
                self.policy_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy_optimizer.step()
                steps += 1
                loss_v = float(loss.detach())
            with th.no_grad():                      # epoch-end full-batch KL (SB3 style)
                d = self.policy.dist(s_t)
                lp = d.log_prob(a_t).sum(dim=-1, keepdim=True)
                kl = float((old_logp - lp).mean())
                clipf = float((th.abs(th.exp(lp - old_logp) - 1.0)
                               > self.clip_ratio).float().mean())
                ent = float(d.entropy().mean())
            if self.target_kl and kl > 1.5 * self.target_kl:
                break
        return loss_v, ent, kl, clipf, steps

    def update(self, batch):
        """One PPO update over the rollout. Returns diagnostics."""
        states = self.obs_normalizer.normalize(batch["states"])
        s_t = self._t(states)
        a_t = self._t(batch["actions"])
        old_logp = self._t(batch["logp"]).view(-1, 1)
        # Diagnostic only: no behaviour change, no RNG use. This is the share of
        # samples whose stored action is already off the recorded log-prob before any
        # gradient step, i.e. the fraction where the PPO ratio starts above
        # |ratio - 1| = clip_ratio.
        with th.no_grad():
            lp0 = self.policy.dist(s_t).log_prob(a_t).sum(dim=-1, keepdim=True)
            ratio_off_at_start = float(
                (th.abs(th.exp(lp0 - old_logp) - 1.0) > self.clip_ratio).float().mean())
        adv = batch["adv_r"].copy()

        if self.lagrangian:
            adv = adv - self.lambda_ * batch["adv_c"]
            if self.lag_normalize_by_1p_lam and (1.0 + self.lambda_) > 1e-8:
                adv = adv / (1.0 + self.lambda_)
        # Advantage-normalisation audit: the norm ratio is measured here, on the raw
        # advantages at the point where they are combined and before the
        # normalisation below. Normalisation applies to the combined advantage (not
        # separately to A_r and A_c), so a large lambda cannot be normalised away: it
        # changes the direction of adv, not just its scale. Logged per update.
        norm_ratio = None
        if self.lagrangian:
            denom = float(np.abs(batch["adv_r"]).mean()) + 1e-12
            norm_ratio = float(np.abs(self.lambda_ * batch["adv_c"]).mean()) / denom
        if self.normalize_advantage:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        adv_t = self._t(adv).view(-1, 1)
        ret_t = self._t(batch["ret_r"]).view(-1, 1)
        retc_t = self._t(batch["ret_c"]).view(-1, 1) if self.lagrangian else None

        # Fit the critic first, on normalised targets (the net output is in norm units).
        self.ret_normalizer.update(np.asarray(batch["ret_r"], dtype=np.float64))
        v_std = float(np.sqrt(self.ret_normalizer.var) + self.ret_normalizer.epsilon)
        v_mean = float(self.ret_normalizer.mean)
        ret_norm = (ret_t - v_mean) / v_std
        if self.lagrangian:
            self.cost_normalizer.update(np.asarray(batch["ret_c"], dtype=np.float64))
            c_std = float(np.sqrt(self.cost_normalizer.var) + self.cost_normalizer.epsilon)
            c_mean = float(self.cost_normalizer.mean)
            retc_norm = (retc_t - c_mean) / c_std
        else:
            retc_norm = None
        if self.use_minibatches:
            v_loss_val, n_val_steps = self._fit_critics_mb(s_t, ret_norm, retc_norm)
        else:
            n_val_steps = self.value_epochs
            for _ in range(self.value_epochs):      # legacy: value_epochs == 1
                v = self.value_net(s_t)
                v_loss = F.mse_loss(v, ret_norm)
                self.value_optimizer.zero_grad()
                v_loss.backward()
                nn.utils.clip_grad_norm_(self.value_net.parameters(), self.max_grad_norm)
                self.value_optimizer.step()
                if retc_norm is not None:
                    cv = self.cost_net(s_t)
                    c_loss = F.mse_loss(cv, retc_norm)
                    self.cost_optimizer.zero_grad()
                    c_loss.backward()
                    nn.utils.clip_grad_norm_(self.cost_net.parameters(), self.max_grad_norm)
                    self.cost_optimizer.step()
            v_loss_val = float(v_loss.detach())

        # clipped policy epochs with early stop on KL
        pg_loss_val, ent_val, kl_val, frac_sat = 0.0, 0.0, 0.0, 0.0
        clip_frac_val, ev_val = 0.0, 0.0
        n_pol_steps = 0
        mean_abs_adv = float(np.abs(batch["adv_r"]).mean())
        if self.use_minibatches:
            pg_loss_val, ent_val, kl_val, clip_frac_val, n_pol_steps = \
                self._policy_epochs_mb(s_t, a_t, old_logp, adv_t)
            frac_sat = float(np.mean(np.abs(batch["actions"]) >= 0.99))
            with th.no_grad():
                ev_val = 1.0 - float(F.mse_loss(self._denorm_v(self.value_net(s_t)), ret_t)
                                     / (ret_t.var() + 1e-8))
        else:
            for _ in range(self.n_epochs):
                d = self.policy.dist(s_t)
                logp = d.log_prob(a_t).sum(dim=-1, keepdim=True)
                ratio = th.exp(logp - old_logp)
                surr1 = ratio * adv_t
                surr2 = th.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * adv_t
                entropy = d.entropy().mean()
                pg_loss = -th.min(surr1, surr2).mean() - self.entropy_coef * entropy

                self.policy_optimizer.zero_grad()
                pg_loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy_optimizer.step()
                n_pol_steps += 1

                with th.no_grad():
                    kl = float((old_logp - logp).mean())
                    frac_sat = float(np.mean(np.abs(batch["actions"]) >= 0.99))
                    # Diagnostics only: clip-binding fraction and value explained variance.
                    clip_frac_val = float((th.abs(ratio - 1.0) > self.clip_ratio).float().mean())
                    v_pred = self.value_net(s_t)
                    ev_val = 1.0 - float(F.mse_loss(v_pred, ret_t) / (ret_t.var() + 1e-8))
                pg_loss_val, ent_val, kl_val = (float(pg_loss.detach()),
                                                float(entropy.detach()), kl)
                if self.target_kl and kl > 1.5 * self.target_kl:
                    break

        return {
            "pg_loss": pg_loss_val,
            "value_loss": v_loss_val,
            "entropy": ent_val,
            "approx_kl": kl_val,
            "frac_saturated": frac_sat,
            "lag_adv_norm_ratio": norm_ratio,  # ||lam*A_c|| / ||A_r|| at combination
            "clip_frac": clip_frac_val,
            "explained_variance": ev_val,
            "mean_abs_adv": mean_abs_adv,
            "log_std": self.policy.log_std.detach().cpu().numpy().tolist(),
            # additive diagnostics (no behaviour change)
            "n_policy_grad_steps": n_pol_steps,
            "n_value_grad_steps": n_val_steps,
            "ratio_off_at_start": ratio_off_at_start,
        }

    def end_episode(self, terminated):
        """Record the episode's mean cost for the dual; clear the rollout."""
        if self.rollout_costs:
            self.cost_history.append(float(np.mean(self.rollout_costs)))
        self.rollout_costs = []

    # ---------------------------------------------------------------- io
    def save_models(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        payload = {
            "policy": self.policy.state_dict(),
            "value_net": self.value_net.state_dict(),
            "obs_mean": self.obs_normalizer.mean.tolist(),
            "obs_var": self.obs_normalizer.var.tolist(),
            "obs_count": float(self.obs_normalizer.count),
            "ret_mean": float(self.ret_normalizer.mean),
            "ret_var": float(self.ret_normalizer.var),
            "ret_count": float(self.ret_normalizer.count),
            "cost_mean": float(self.cost_normalizer.mean),
            "cost_var": float(self.cost_normalizer.var),
            "cost_count": float(self.cost_normalizer.count),
            "lambda": self.lambda_,
            "lagrangian": self.lagrangian,
            "pbrs": self.pbrs,
            "residual": self.residual_policy is not None,
            "residual_coef": self.residual_coef,
            "shed_mapping": getattr(self, "_shed_mapping", "linear"),
        }
        if self.cost_net is not None:
            payload["cost_net"] = self.cost_net.state_dict()
        th.save(payload, path)

    def load_models(self, path: str):
        payload = th.load(path, map_location=self.device)
        self.policy.load_state_dict(payload["policy"])
        self.value_net.load_state_dict(payload["value_net"])
        self.obs_normalizer.mean = np.array(payload["obs_mean"], dtype=np.float32)
        self.obs_normalizer.var = np.array(payload["obs_var"], dtype=np.float32)
        self.obs_normalizer.count = float(payload["obs_count"])
        for prefix, norm in (("ret_", self.ret_normalizer), ("cost_", self.cost_normalizer)):
            if prefix + "mean" in payload:
                norm.mean = np.array(payload[prefix + "mean"], dtype=np.float64)
                norm.var = np.array(payload[prefix + "var"], dtype=np.float64)
                norm.count = float(payload[prefix + "count"])
        self.lambda_ = float(payload.get("lambda", 0.0))
        if self.cost_net is not None and "cost_net" in payload:
            self.cost_net.load_state_dict(payload["cost_net"])


class PPOPotentialShapingAgent(PPOAgent):
    """PPO + potential-based reward shaping.

    Phi(s) = kappa * SoC (state[0]), via the shared PotentialBasedShaping class so
    the potential stays consistent with the reward-shaping definition.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # kappa=1.0 here: PotentialBasedShaping.phi returns kappa*SoC, and shape()
        # above applies self.kappa - passing kappa twice would square it.
        from agents.shaping import PotentialBasedShaping
        self._shaping = PotentialBasedShaping(gamma=self.gamma, kappa=1.0)

    def _phi(self, state):
        return float(self._shaping.phi(np.asarray(state, dtype=np.float32)))
