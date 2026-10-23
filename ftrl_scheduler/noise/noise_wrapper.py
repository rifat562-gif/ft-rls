"""
noise_wrapper.py
----------------
Gymnasium wrapper that applies the three IIoT noise modes from Section 3.4
to the observation vector before it reaches the agent.

Noise modes
-----------
1. Bernoulli packet loss  : each feature dropped with prob p_loss
2. Gaussian sensor drift  : additive N(0, sigma^2) noise
3. Bursty delay (MMPP)    : agent receives observation from tau steps ago
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque


class IIoTNoiseWrapper(gym.ObservationWrapper):
    """
    Wraps an FJSPEnv and corrupts observations according to the
    parametric IIoT noise model.

    Parameters
    ----------
    env        : FJSPEnv instance
    p_loss     : packet-loss probability in [0,1]
    sigma      : std of Gaussian drift (as fraction of obs range, e.g. 0.05)
    delay_mode : 'none' | 'markov'  -- bursty delay model
    q01        : P(normal -> congested)   (used when delay_mode='markov')
    q10        : P(congested -> normal)
    lam0       : mean delay in normal state (steps)
    lam1       : mean delay in congested state (steps)
    seed       : random seed
    """

    def __init__(self, env,
                 p_loss=0.0, sigma=0.0,
                 delay_mode="none", q01=0.02, q10=0.3,
                 lam0=1, lam1=5, seed=None):
        super().__init__(env)
        self.p_loss     = p_loss
        self.sigma      = sigma
        self.delay_mode = delay_mode
        self.q01        = q01
        self.q10        = q10
        self.lam0       = lam0
        self.lam1       = lam1
        self._rng       = np.random.default_rng(seed)

        obs_dim  = env.observation_space.shape[0]
        # extend obs with missingness indicators
        self._obs_dim = obs_dim
        total_dim     = obs_dim * 2          # [imputed_obs || missing_mask]
        self.observation_space = spaces.Box(
            low=-6.0, high=6.0, shape=(total_dim,), dtype=np.float32
        )

        # delay buffer: stores past true obs
        self._max_delay   = max(lam1 * 4, 20)
        self._obs_buffer  = deque(maxlen=self._max_delay + 1)
        # MMPP state
        self._net_state   = 0        # 0 = normal, 1 = congested
        # last known values for LOCF imputation
        self._last_known  = None

    # ── reset ────────────────────────────────────────────────────────────────

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._obs_buffer.clear()
        self._obs_buffer.append(obs.copy())
        self._last_known = obs.copy()
        self._net_state  = 0
        corrupted = self._corrupt(obs)
        info["action_mask"] = self.env.action_mask()
        return corrupted, info

    # ── observation wrapping ─────────────────────────────────────────────────

    def observation(self, obs):
        self._obs_buffer.append(obs.copy())
        return self._corrupt(obs)

    def _corrupt(self, true_obs):
        """Apply delay → packet loss → Gaussian drift, then LOCF + mask."""
        # 1. bursty delay
        if self.delay_mode == "markov":
            obs = self._apply_delay()
        else:
            obs = true_obs.copy()

        # 2. Gaussian drift
        if self.sigma > 0.0:
            obs = obs + self._rng.normal(0.0, self.sigma, size=obs.shape).astype(np.float32)

        # 3. packet loss + LOCF imputation
        missing = np.zeros(self._obs_dim, dtype=np.float32)
        if self.p_loss > 0.0:
            drop = self._rng.random(self._obs_dim) < self.p_loss
            missing[drop] = 1.0
            if self._last_known is None:
                obs[drop] = 0.0
            else:
                obs[drop] = self._last_known[drop]   # LOCF

        # update last-known with non-missing values
        if self._last_known is None:
            self._last_known = obs.copy()
        else:
            self._last_known[missing == 0] = obs[missing == 0]

        # concatenate [obs || missingness_mask]
        return np.concatenate([obs, missing], axis=0).astype(np.float32)

    def _apply_delay(self):
        """MMPP delay: advance network state, sample delay, return old obs."""
        # state transition
        if self._net_state == 0:
            if self._rng.random() < self.q01:
                self._net_state = 1
        else:
            if self._rng.random() < self.q10:
                self._net_state = 0

        lam  = self.lam0 if self._net_state == 0 else self.lam1
        tau  = min(int(self._rng.poisson(lam)), len(self._obs_buffer) - 1)
        idx  = max(0, len(self._obs_buffer) - 1 - tau)
        return list(self._obs_buffer)[idx].copy()

    # ── action mask passthrough ──────────────────────────────────────────────

    def action_mask(self):
        return self.env.action_mask()

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        corrupted = self.observation(obs)
        info["action_mask"] = self.env.action_mask()
        info["true_obs"]    = obs          # available for reward computation
        return corrupted, reward, terminated, truncated, info


# ── convenience factory ───────────────────────────────────────────────────────

NOISE_PRESETS = {
    # (p_loss, sigma, delay_mode, lam1)
    "clean"       : dict(p_loss=0.00, sigma=0.00, delay_mode="none"),
    "loss_low"    : dict(p_loss=0.05, sigma=0.00, delay_mode="none"),
    "loss_med"    : dict(p_loss=0.20, sigma=0.00, delay_mode="none"),
    "loss_high"   : dict(p_loss=0.40, sigma=0.00, delay_mode="none"),
    "drift_low"   : dict(p_loss=0.00, sigma=0.01, delay_mode="none"),
    "drift_med"   : dict(p_loss=0.00, sigma=0.05, delay_mode="none"),
    "drift_high"  : dict(p_loss=0.00, sigma=0.15, delay_mode="none"),
    "delay_low"   : dict(p_loss=0.00, sigma=0.00, delay_mode="markov", lam1=1),
    "delay_med"   : dict(p_loss=0.00, sigma=0.00, delay_mode="markov", lam1=3),
    "delay_high"  : dict(p_loss=0.00, sigma=0.00, delay_mode="markov", lam1=5),
    "severe"      : dict(p_loss=0.20, sigma=0.05, delay_mode="markov", lam1=5),
}


def make_noisy_env(n_jobs=10, n_machines=5, alpha=0.5,
                   noise_preset="clean", seed=None):
    """Create an FJSPEnv wrapped with a noise preset."""
    from env.fjsp_env import FJSPEnv
    env    = FJSPEnv(n_jobs=n_jobs, n_machines=n_machines, alpha=alpha, seed=seed)
    kwargs = NOISE_PRESETS[noise_preset].copy()
    return IIoTNoiseWrapper(env, seed=seed, **kwargs)
