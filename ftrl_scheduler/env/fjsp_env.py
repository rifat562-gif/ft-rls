"""
fjsp_env.py
-----------
Gymnasium environment for the Flexible Job-Shop Scheduling Problem (FJSP).

State  : flat vector [machine_features | job_features]
Action : integer index -> (job_idx, machine_idx) pair
Reward : negative delta-cost (makespan + weighted tardiness)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class FJSPEnv(gym.Env):
    """
    Flexible Job-Shop Scheduling Problem environment.

    Parameters
    ----------
    n_jobs     : number of jobs
    n_machines : number of machines
    alpha      : weight on makespan vs tardiness (0=only tardiness, 1=only makespan)
    seed       : random seed for instance generation
    instance   : optional pre-built instance dict (overrides random generation)
    """

    metadata = {"render_modes": []}

    def __init__(self, n_jobs=10, n_machines=5, alpha=0.5, seed=None, instance=None):
        super().__init__()
        self.n_jobs      = n_jobs
        self.n_machines  = n_machines
        self.alpha       = alpha
        self.base_seed   = seed
        self._instance   = instance

        # observation dim: per-machine (3 features) + per-job (4 features) + global (2)
        self.machine_feat = 3   # [idle_time, utilisation, queue_len]
        self.job_feat     = 4   # [remaining_ops, time_to_due, weight, is_started]
        self.global_feat  = 2   # [normalised_t, n_scheduled_ops]
        self.obs_dim = (n_machines * self.machine_feat +
                        n_jobs     * self.job_feat +
                        self.global_feat)

        self.observation_space = spaces.Box(
            low=-5.0, high=5.0, shape=(self.obs_dim,), dtype=np.float32
        )
        # action = (job, machine) pair flattened
        self.action_space = spaces.Discrete(n_jobs * n_machines)

        self._rng = np.random.default_rng(seed)
        self._build_instance()

    # ── instance generation ──────────────────────────────────────────────────

    def _build_instance(self):
        """Generate or load an FJSP instance."""
        if self._instance is not None:
            inst = self._instance
        else:
            inst = self._random_instance()
        self._proc_times  = inst["proc_times"]   # (n_jobs, max_ops, n_machines), NaN = ineligible
        self._n_ops       = inst["n_ops"]         # (n_jobs,) int
        self._due_dates   = inst["due_dates"]     # (n_jobs,) float
        self._weights     = inst["weights"]       # (n_jobs,) float
        self._max_ops     = self._proc_times.shape[1]
        self._horizon     = float(np.nansum(self._proc_times))  # loose upper bound

    def _random_instance(self):
        rng = self._rng
        n_ops = rng.integers(2, 6, size=self.n_jobs)          # 2-5 ops per job
        max_o = int(n_ops.max())
        proc  = np.full((self.n_jobs, max_o, self.n_machines), np.nan)
        for i in range(self.n_jobs):
            for j in range(n_ops[i]):
                eligible = rng.choice(self.n_machines,
                                      size=rng.integers(1, self.n_machines + 1),
                                      replace=False)
                proc[i, j, eligible] = rng.uniform(1.0, 10.0, size=len(eligible))

        total_work = np.nansum(proc) / self.n_machines
        due_dates  = rng.uniform(total_work * 0.8, total_work * 1.5, size=self.n_jobs)
        weights    = rng.uniform(0.5, 2.0, size=self.n_jobs)
        return dict(proc_times=proc, n_ops=n_ops, due_dates=due_dates, weights=weights)

    # ── reset / step ─────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self._build_instance()

        self._t            = 0.0                          # current time
        self._machine_free = np.zeros(self.n_machines)   # time when machine becomes free
        self._job_op_idx   = np.zeros(self.n_jobs, dtype=int)  # next op index per job
        self._job_end_time = np.zeros(self.n_jobs)        # completion time per job
        self._scheduled    = 0                            # total ops scheduled
        self._total_ops    = int(self._n_ops.sum())
        self._done         = False
        self._prev_cost    = 0.0

        obs  = self._get_obs()
        info = {"action_mask": self._get_mask()}
        return obs, info

    def step(self, action):
        assert not self._done, "Call reset() before stepping."
        job_idx = action // self.n_machines
        mac_idx = action  % self.n_machines

        # --- validate ---
        op_idx = self._job_op_idx[job_idx]
        pt     = self._proc_times[job_idx, op_idx, mac_idx]
        if np.isnan(pt) or op_idx >= self._n_ops[job_idx]:
            # infeasible action
            obs  = self._get_obs()
            info = {"action_mask": self._get_mask(), "infeasible": True}
            return obs, -10.0, False, False, info

        # --- schedule ---
        start  = max(self._t, self._machine_free[mac_idx], self._job_end_time[job_idx])
        finish = start + pt
        self._machine_free[mac_idx]  = finish
        self._job_end_time[job_idx]  = finish
        self._job_op_idx[job_idx]   += 1
        self._scheduled             += 1

        # advance global clock to min free time among jobs with pending ops
        pending_jobs = [j for j in range(self.n_jobs) if self._job_op_idx[j] < self._n_ops[j]]
        if pending_jobs:
            self._t = min(self._job_end_time[j] for j in pending_jobs)
        else:
            self._t = float(np.max(self._job_end_time))

        # --- reward (delta cost) ---
        cost     = self._compute_cost()
        reward   = -(cost - self._prev_cost)
        self._prev_cost = cost

        # --- done ---
        self._done = (self._scheduled == self._total_ops)
        obs  = self._get_obs()
        info = {"action_mask": self._get_mask(),
                "makespan": float(np.max(self._job_end_time)),
                "tardiness": float(self._weighted_tardiness())}
        return obs, float(reward), self._done, False, info

    # ── helpers ──────────────────────────────────────────────────────────────

    def _compute_cost(self):
        cmax = float(np.max(self._job_end_time))
        tard = float(self._weighted_tardiness())
        return self.alpha * cmax + (1 - self.alpha) * tard

    def _weighted_tardiness(self):
        return float(np.sum(self._weights *
                            np.maximum(0.0, self._job_end_time - self._due_dates)))

    def _get_mask(self):
        """Boolean mask of shape (n_jobs * n_machines,). True = valid action."""
        mask = np.zeros(self.n_jobs * self.n_machines, dtype=bool)
        for j in range(self.n_jobs):
            op = self._job_op_idx[j]
            if op >= self._n_ops[j]:
                continue
            for k in range(self.n_machines):
                if not np.isnan(self._proc_times[j, op, k]):
                    mask[j * self.n_machines + k] = True
        return mask

    def _get_obs(self):
        """Return normalised observation vector (true state, no noise)."""
        h   = max(self._horizon, 1.0)
        obs = []

        # machine features
        for k in range(self.n_machines):
            idle  = max(0.0, self._machine_free[k] - self._t) / h
            util  = min(self._machine_free[k], self._t) / max(self._t, 1.0)
            qlen  = sum(1 for j in range(self.n_jobs)
                        if self._job_op_idx[j] < self._n_ops[j]
                        and not np.isnan(self._proc_times[j, self._job_op_idx[j], k])) / self.n_jobs
            obs.extend([idle, util, qlen])

        # job features
        for j in range(self.n_jobs):
            rem   = (self._n_ops[j] - self._job_op_idx[j]) / max(self._n_ops[j], 1)
            slack = (self._due_dates[j] - self._t) / h
            w     = self._weights[j] / 2.0          # normalise (weights in [0.5, 2])
            started = float(self._job_op_idx[j] > 0)
            obs.extend([rem, np.clip(slack, -2, 2), w, started])

        # global features
        obs.extend([self._t / h,
                    self._scheduled / max(self._total_ops, 1)])

        return np.array(obs, dtype=np.float32)

    def get_true_state(self):
        """Return the raw true state dict (used by noise wrapper)."""
        return {
            "obs_vector"    : self._get_obs(),
            "machine_free"  : self._machine_free.copy(),
            "job_end_time"  : self._job_end_time.copy(),
            "job_op_idx"    : self._job_op_idx.copy(),
            "t"             : self._t,
        }

    def action_mask(self):
        return self._get_mask()
