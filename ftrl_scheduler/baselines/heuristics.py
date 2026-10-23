"""
heuristics.py  —  Classic dispatching rule baselines
"""
import numpy as np


def _get_base(env):
    inner = env
    while hasattr(inner, "env"):
        inner = inner.env
    return inner


def _feasible(env):
    base  = _get_base(env)
    mask  = base.action_mask()
    n_mac = base.n_machines
    return [(a // n_mac, a % n_mac) for a, ok in enumerate(mask) if ok], base


def spt_action(env, rng=None):
    pairs, base = _feasible(env)
    if not pairs:
        return 0
    best, best_pt = None, float("inf")
    for j, k in pairs:
        op = base._job_op_idx[j]
        pt = base._proc_times[j, op, k]
        if pt < best_pt:
            best_pt = pt; best = (j, k)
    return best[0] * base.n_machines + best[1]


def edd_action(env, rng=None):
    pairs, base = _feasible(env)
    if not pairs:
        return 0
    pairs_sorted = sorted(pairs, key=lambda jk: base._due_dates[jk[0]])
    j, _ = pairs_sorted[0]
    op   = base._job_op_idx[j]
    best_k, best_pt = None, float("inf")
    for jj, k in pairs_sorted:
        if jj != j:
            continue
        pt = base._proc_times[j, op, k]
        if pt < best_pt:
            best_pt = pt; best_k = k
    return j * base.n_machines + best_k


def fifo_action(env, rng=None):
    pairs, base = _feasible(env)
    if not pairs:
        return 0
    j, k = min(pairs, key=lambda jk: jk[0])
    return j * base.n_machines + k


def random_action(env, rng=None):
    pairs, base = _feasible(env)
    if not pairs:
        return 0
    if rng is None:
        rng = np.random.default_rng()
    j, k = pairs[rng.integers(len(pairs))]
    return j * base.n_machines + k


def run_heuristic(env, policy_fn, seed=None, rng=None):
    obs, info = env.reset(seed=seed)
    total_reward = 0.0
    done = False
    while not done:
        action = policy_fn(env, rng=rng)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
    base = _get_base(env)
    return {
        "makespan"    : float(np.max(base._job_end_time)),
        "tardiness"   : float(base._weighted_tardiness()),
        "total_reward": total_reward,
    }
