# FT-RLS: Fault-Tolerant Reinforcement Learning Scheduler

## Project Structure

```
ftrl_scheduler/
├── env/
│   └── fjsp_env.py          # FJSP Gymnasium environment (Section 3.1-3.2)
├── noise/
│   └── noise_wrapper.py     # IIoT noise wrapper — 3 modes (Section 3.4)
├── models/
│   └── belief_policy.py     # LSTM belief estimator + PPO policy (Section 3.5)
├── baselines/
│   └── heuristics.py        # SPT, EDD, FIFO, Random baselines
├── experiments/
│   ├── train.py             # Full training + evaluation script
│   └── test_pipeline.py     # Smoke test (run this first)
└── results/                 # JSON output files
```

## Quick Start

### 1. Install dependencies
```bash
pip install torch gymnasium stable-baselines3 sb3-contrib
```

### 2. Smoke test (verify everything works)
```bash
cd ftrl_scheduler
python experiments/test_pipeline.py
```
Expected output: ALL TESTS PASSED

### 3. Run a single noise preset
```bash
python experiments/train.py \
  --preset loss_med \
  --n_jobs 10 --n_machines 5 \
  --total_steps 200000 \
  --n_seeds 3 --n_eval 20 --n_envs 4
```

### 4. Run full experiment grid (all noise presets)
```bash
python experiments/train.py --preset all --total_steps 200000 --n_seeds 5
```

## Noise Presets

| Preset     | p_loss | sigma | Delay mode | Severity |
|------------|--------|-------|------------|----------|
| clean      | 0.00   | 0.00  | none       | Baseline |
| loss_low   | 0.05   | 0.00  | none       | Low      |
| loss_med   | 0.20   | 0.00  | none       | Medium   |
| loss_high  | 0.40   | 0.00  | none       | High     |
| drift_low  | 0.00   | 0.01  | none       | Low      |
| drift_med  | 0.00   | 0.05  | none       | Medium   |
| drift_high | 0.00   | 0.15  | none       | High     |
| delay_low  | 0.00   | 0.00  | markov(1)  | Low      |
| delay_med  | 0.00   | 0.00  | markov(3)  | Medium   |
| delay_high | 0.00   | 0.00  | markov(5)  | High     |
| severe     | 0.20   | 0.05  | markov(5)  | Combined |

## Key Hyperparameters

| Parameter       | Value   | Notes                              |
|-----------------|---------|------------------------------------|
| LSTM layers     | 2       | hidden_size=256                    |
| Belief dim      | 128     | b_t ∈ R^128                        |
| PPO lr          | 3e-4    | Adam optimizer                     |
| n_steps         | 2048    | Rollout buffer size                |
| batch_size      | 256     | Mini-batch size                    |
| gamma           | 0.99    | Discount factor                    |
| gae_lambda      | 0.95    | GAE lambda                         |
| clip_range      | 0.2     | PPO epsilon                        |
| ent_coef        | 0.01    | Entropy bonus                      |
| Training steps  | 200,000 | Per (preset, seed) run             |

## Baselines

| Agent       | Description                                      |
|-------------|--------------------------------------------------|
| FT-RLS      | LSTM belief + MaskablePPO (proposed)             |
| MLP-noisy   | MLP PPO acting on raw noisy obs (ablation)       |
| PPO-clean   | MLP PPO on clean obs (upper bound)               |
| SPT         | Shortest Processing Time rule                    |
| EDD         | Earliest Due Date rule                           |
| FIFO        | First-In First-Out rule                          |

## Experiment Grid (for paper)

- Presets: 10 noise conditions + clean
- Seeds: 5 per condition
- Eval episodes: 20 per seed
- FJSP instances: Brandimarte MK01-MK05
- Total runs: 11 presets × 5 seeds × 2 RL agents = 110 training runs

## Paper Mapping

| Code module          | Paper section        |
|----------------------|----------------------|
| fjsp_env.py          | Section 3.1 – 3.2    |
| noise_wrapper.py     | Section 3.4 (C3)     |
| belief_policy.py     | Section 3.5 (C2)     |
| train.py             | Section 6            |
| results/             | Section 7            |
