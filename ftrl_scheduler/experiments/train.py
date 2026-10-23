"""
train.py
--------
Train FT-RLS (LSTM belief + PPO) and all baselines across noise presets.

Usage
-----
python train.py --preset clean       # single preset
python train.py --preset all         # full experiment grid
python train.py --preset severe --n_jobs 10 --n_machines 5 --total_steps 300000
"""

import argparse
import json
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from env.fjsp_env       import FJSPEnv
from noise.noise_wrapper import IIoTNoiseWrapper, NOISE_PRESETS
from models.belief_policy import BeliefActorCriticPolicy, MLPPolicy
from baselines.heuristics  import spt_action, edd_action, fifo_action, run_heuristic


# ── helpers ───────────────────────────────────────────────────────────────────

def make_env(n_jobs, n_machines, alpha, noise_preset, seed):
    """Factory: create one env instance (noisy, with action masker)."""
    noise_kwargs = NOISE_PRESETS[noise_preset].copy()
    base = FJSPEnv(n_jobs=n_jobs, n_machines=n_machines, alpha=alpha, seed=seed)
    noisy = IIoTNoiseWrapper(base, seed=seed, **noise_kwargs)

    def mask_fn(e):
        # navigate wrapper chain to reach base env
        inner = e
        while hasattr(inner, "env"):
            inner = inner.env
        return inner.action_mask()

    masked = ActionMasker(noisy, mask_fn)
    return Monitor(masked)


def evaluate_agent(model, env_factory, n_eval=20, seed_offset=1000):
    """Roll out trained model for n_eval episodes; return mean metrics."""
    makespans, tardinesses, rewards = [], [], []
    for i in range(n_eval):
        env  = env_factory(seed=seed_offset + i)
        obs, info = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            mask   = info.get("action_mask", None)
            action, _ = model.predict(obs, action_masks=mask, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated

        # get true metrics from base env
        base = env
        while hasattr(base, "env"):
            base = base.env

        makespans.append(float(np.max(base._job_end_time)))
        tardinesses.append(float(base._weighted_tardiness()))
        rewards.append(ep_reward)

    return {
        "makespan_mean" : float(np.mean(makespans)),
        "makespan_std"  : float(np.std(makespans)),
        "tardiness_mean": float(np.mean(tardinesses)),
        "tardiness_std" : float(np.std(tardinesses)),
        "reward_mean"   : float(np.mean(rewards)),
    }


def evaluate_heuristic(policy_fn, env_factory, n_eval=20, seed_offset=1000):
    """Evaluate a heuristic dispatching rule."""
    results = []
    rng     = np.random.default_rng(42)
    for i in range(n_eval):
        env = env_factory(seed=seed_offset + i)
        res = run_heuristic(env, policy_fn, seed=seed_offset + i, rng=rng)
        results.append(res)
    return {
        "makespan_mean" : float(np.mean([r["makespan"]     for r in results])),
        "makespan_std"  : float(np.std( [r["makespan"]     for r in results])),
        "tardiness_mean": float(np.mean([r["tardiness"]    for r in results])),
        "tardiness_std" : float(np.std( [r["tardiness"]    for r in results])),
        "reward_mean"   : float(np.mean([r["total_reward"] for r in results])),
    }


# ── training ──────────────────────────────────────────────────────────────────

def train_ftrl(args, noise_preset, run_seed):
    """Train FT-RLS (LSTM belief + MaskablePPO) for one (preset, seed) run."""
    print(f"\n[FT-RLS] preset={noise_preset}  seed={run_seed}")

    def env_factory(seed=None):
        return make_env(args.n_jobs, args.n_machines, args.alpha,
                        noise_preset, seed or run_seed)

    train_env = make_vec_env(lambda: env_factory(run_seed), n_envs=args.n_envs)

    model = MaskablePPO(
        policy          = BeliefActorCriticPolicy,
        env             = train_env,
        learning_rate   = 3e-4,
        n_steps         = 2048,
        batch_size      = 256,
        n_epochs        = 10,
        gamma           = 0.99,
        gae_lambda      = 0.95,
        clip_range      = 0.2,
        ent_coef        = 0.01,
        vf_coef         = 0.5,
        max_grad_norm   = 0.5,
        verbose         = 0,
        seed            = run_seed,
    )

    t0 = time.time()
    model.learn(total_timesteps=args.total_steps, progress_bar=False)
    train_time = time.time() - t0

    metrics = evaluate_agent(model, env_factory, n_eval=args.n_eval,
                             seed_offset=run_seed + 1000)
    metrics["train_time_s"] = round(train_time, 1)
    return model, metrics


def train_noisy_mlp(args, noise_preset, run_seed):
    """Train vanilla PPO with MLP (no belief) on noisy obs — ablation baseline."""
    print(f"\n[MLP-noisy] preset={noise_preset}  seed={run_seed}")

    def env_factory(seed=None):
        return make_env(args.n_jobs, args.n_machines, args.alpha,
                        noise_preset, seed or run_seed)

    train_env = make_vec_env(lambda: env_factory(run_seed), n_envs=args.n_envs)

    model = MaskablePPO(
        policy        = "MlpPolicy",
        env           = train_env,
        learning_rate = 3e-4,
        n_steps       = 2048,
        batch_size    = 256,
        n_epochs      = 10,
        gamma         = 0.99,
        verbose       = 0,
        seed          = run_seed,
    )
    model.learn(total_timesteps=args.total_steps, progress_bar=False)
    metrics = evaluate_agent(model, env_factory, n_eval=args.n_eval,
                             seed_offset=run_seed + 1000)
    return model, metrics


def train_clean_ppo(args, run_seed):
    """Train PPO on clean (no noise) observations — upper bound baseline."""
    print(f"\n[PPO-clean] seed={run_seed}")

    def env_factory(seed=None):
        return make_env(args.n_jobs, args.n_machines, args.alpha,
                        "clean", seed or run_seed)

    train_env = make_vec_env(lambda: env_factory(run_seed), n_envs=args.n_envs)

    model = MaskablePPO(
        policy        = "MlpPolicy",
        env           = train_env,
        learning_rate = 3e-4,
        n_steps       = 2048,
        batch_size    = 256,
        n_epochs      = 10,
        gamma         = 0.99,
        verbose       = 0,
        seed          = run_seed,
    )
    model.learn(total_timesteps=args.total_steps, progress_bar=False)
    metrics = evaluate_agent(model, env_factory, n_eval=args.n_eval,
                             seed_offset=run_seed + 1000)
    return model, metrics


# ── main experiment loop ──────────────────────────────────────────────────────

def run_experiment(args):
    presets = list(NOISE_PRESETS.keys()) if args.preset == "all" else [args.preset]
    seeds   = list(range(args.n_seeds))
    results = {}

    os.makedirs(args.out_dir, exist_ok=True)

    for preset in presets:
        results[preset] = {"FT-RLS": [], "MLP-noisy": [],
                           "SPT": [], "EDD": [], "FIFO": []}

        for seed in seeds:
            # ── FT-RLS ──
            _, m = train_ftrl(args, preset, seed)
            results[preset]["FT-RLS"].append(m)

            # ── MLP noisy baseline ──
            _, m = train_noisy_mlp(args, preset, seed)
            results[preset]["MLP-noisy"].append(m)

            # ── heuristics (fast, no training needed) ──
            def env_f(seed=None):
                return make_env(args.n_jobs, args.n_machines, args.alpha,
                                preset, seed or 0)

            for name, fn in [("SPT", spt_action), ("EDD", edd_action), ("FIFO", fifo_action)]:
                m = evaluate_heuristic(fn, env_f, n_eval=args.n_eval, seed_offset=seed + 1000)
                results[preset][name].append(m)

        # aggregate across seeds
        for agent in list(results[preset].keys()):
            runs = results[preset][agent]
            agg  = {}
            for k in runs[0]:
                vals = [r[k] for r in runs]
                agg[k + "_avg"] = round(float(np.mean(vals)), 4)
                agg[k + "_sem"] = round(float(np.std(vals) / max(len(vals)**0.5, 1)), 4)
            results[preset][agent + "_agg"] = agg

        # save incrementally
        out_path = os.path.join(args.out_dir, "results.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[saved] {out_path}")

    return results


def print_summary(results):
    print("\n" + "=" * 70)
    print(f"{'PRESET':<14} {'AGENT':<12} {'MAKESPAN':>10} {'TARDINESS':>11} {'REWARD':>10}")
    print("=" * 70)
    for preset, agents in results.items():
        for agent in ["FT-RLS", "MLP-noisy", "SPT", "EDD", "FIFO"]:
            agg_key = agent + "_agg"
            if agg_key not in agents:
                continue
            a = agents[agg_key]
            print(f"{preset:<14} {agent:<12} "
                  f"{a.get('makespan_mean_avg', 0):>10.2f} "
                  f"{a.get('tardiness_mean_avg', 0):>11.2f} "
                  f"{a.get('reward_mean_avg', 0):>10.3f}")
    print("=" * 70)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FT-RLS training and evaluation")
    parser.add_argument("--preset",       default="clean",
                        help="Noise preset name or 'all'")
    parser.add_argument("--n_jobs",       type=int, default=10)
    parser.add_argument("--n_machines",   type=int, default=5)
    parser.add_argument("--alpha",        type=float, default=0.5)
    parser.add_argument("--total_steps",  type=int, default=200_000,
                        help="PPO training steps per run")
    parser.add_argument("--n_seeds",      type=int, default=3,
                        help="Number of random seeds per preset")
    parser.add_argument("--n_eval",       type=int, default=10,
                        help="Episodes for evaluation")
    parser.add_argument("--n_envs",       type=int, default=4,
                        help="Parallel training environments")
    parser.add_argument("--out_dir",      default="results/")

    args   = parser.parse_args()
    res    = run_experiment(args)
    print_summary(res)
