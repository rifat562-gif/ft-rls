"""
test_pipeline.py
----------------
Smoke test: verifies the full FT-RLS pipeline runs without errors.
Runs a tiny training loop (500 steps) and evaluation on all noise presets.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from env.fjsp_env        import FJSPEnv
from noise.noise_wrapper  import IIoTNoiseWrapper, NOISE_PRESETS
from models.belief_policy import BeliefActorCriticPolicy
from baselines.heuristics  import spt_action, edd_action, run_heuristic


def make_test_env(noise_preset="clean", seed=0):
    noise_kw = NOISE_PRESETS[noise_preset].copy()
    base  = FJSPEnv(n_jobs=6, n_machines=3, alpha=0.5, seed=seed)
    noisy = IIoTNoiseWrapper(base, seed=seed, **noise_kw)

    def mask_fn(e):
        inner = e
        while hasattr(inner, "env"):
            inner = inner.env
        return inner.action_mask()

    return Monitor(ActionMasker(noisy, mask_fn))


def test_env():
    print("── Test 1: FJSPEnv + NoiseWrapper ──")
    for preset in ["clean", "loss_med", "drift_med", "delay_med", "severe"]:
        env = make_test_env(preset)
        obs, info = env.reset()
        assert "action_mask" in info, "Missing action_mask in info"
        for _ in range(10):
            mask   = info["action_mask"]
            valid  = np.where(mask)[0]
            action = int(np.random.choice(valid))
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        print(f"  {preset:<14}: obs shape={obs.shape}  reward={reward:.3f}  OK")
    print("  PASSED\n")


def test_belief_estimator():
    print("── Test 2: LSTM Belief Estimator ──")
    from models.belief_policy import LSTMBeliefEstimator
    estimator = LSTMBeliefEstimator(input_dim=40, hidden_size=64, belief_dim=32)
    x = torch.randn(4, 10, 40)          # batch=4, seq=10, input=40
    b, (h, c) = estimator(x)
    assert b.shape == (4, 32), f"Bad belief shape: {b.shape}"
    print(f"  Belief shape: {b.shape}  |  Hidden: {h.shape}  PASSED\n")


def test_ppo_training():
    print("── Test 3: MaskablePPO + BeliefPolicy (500 steps) ──")
    train_env = make_vec_env(lambda: make_test_env("loss_med"), n_envs=2)
    model = MaskablePPO(
        policy       = BeliefActorCriticPolicy,
        env          = train_env,
        learning_rate= 3e-4,
        n_steps      = 64,
        batch_size   = 32,
        n_epochs     = 2,
        verbose      = 0,
    )
    model.learn(total_timesteps=500, progress_bar=False)
    print("  Training completed  PASSED\n")


def test_heuristics():
    print("── Test 4: Heuristic baselines ──")
    for name, fn in [("SPT", spt_action), ("EDD", edd_action)]:
        env = make_test_env("severe")
        res = run_heuristic(env, fn, seed=42)
        print(f"  {name}: makespan={res['makespan']:.2f}  tardiness={res['tardiness']:.2f}  OK")
    print("  PASSED\n")


def test_full_pipeline():
    print("── Test 5: Mini experiment (clean + severe, 1 seed) ──")
    for preset in ["clean", "severe"]:
        train_env = make_vec_env(lambda: make_test_env(preset), n_envs=2)
        model = MaskablePPO(
            policy       = BeliefActorCriticPolicy,
            env          = train_env,
            n_steps      = 128,
            batch_size   = 64,
            n_epochs     = 2,
            verbose      = 0,
        )
        model.learn(total_timesteps=1000, progress_bar=False)

        # eval
        env  = make_test_env(preset, seed=999)
        obs, info = env.reset()
        done = False
        while not done:
            mask   = info.get("action_mask")
            a, _   = model.predict(obs, action_masks=mask, deterministic=True)
            obs, r, terminated, truncated, info = env.step(a)
            done   = terminated or truncated
        print(f"  {preset:<8}: eval complete  OK")
    print("  PASSED\n")


if __name__ == "__main__":
    print("=" * 50)
    print("FT-RLS Pipeline Smoke Test")
    print("=" * 50 + "\n")
    test_env()
    test_belief_estimator()
    test_ppo_training()
    test_heuristics()
    test_full_pipeline()
    print("=" * 50)
    print("ALL TESTS PASSED")
    print("=" * 50)
