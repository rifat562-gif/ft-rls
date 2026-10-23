# FT-RLS: fault-tolerant RL scheduling under corrupted IIoT observations

A POMDP formulation of IIoT-driven flexible job-shop scheduling, with an LSTM
belief state estimator trained end-to-end alongside a PPO scheduling agent, and a
parametric three-mode IIoT noise model with 11 named presets.

Presented at **IEEE PEEIACON 2026**, Dhaka, 6 to 7 August 2026.

Before comparing anything here against the paper, read
[`RECONCILIATION.md`](RECONCILIATION.md). It lists every place the code and the
paper currently differ, including the fact that `results/demo_results.json` is
smoke-test output rather than the experiment.

## What is being claimed, and what is not

The claim is **not** that deep RL beats classical dispatching rules on absolute
makespan. It does not. SPT reaches a makespan of 70.0 against FT-RLS at roughly
88 to 91. SPT reads the exact true noiseless state and optimises a single
objective, so it is noise-immune by construction, while FT-RLS has to recover
usable state from a corrupted and possibly stale observation and minimise
makespan and weighted tardiness jointly. SPT and EDD appear as absolute reference
points only.

The claim is that a **POMDP-belief architecture degrades less than a memoryless
DRL baseline as IIoT noise gets worse**. FT-RLS against MLP-noisy is the
experiment.

## Results

Across 11 noise conditions, 5 seeds, 20 evaluation episodes per seed.

> **Note on the win counts.** The paper reports 7 of 11 on makespan and 8 of 11
> on tardiness. Counting directly from the per-condition means in
> `results/paper_results.json` gives 5 of 11 and 4 of 11. See point 1 of
> `RECONCILIATION.md`. The family-level finding below is unaffected and is what
> the paper's argument rests on.

- FT-RLS wins the **entire bursty-delay family** on both metrics and loses the
  packet-loss family, which is the degradation result.
- **Bursty delay is where it works.** FT-RLS reward *improves* from −46.97
  (clean) to −44.75 (delay high) while MLP-noisy degrades from −45.18 to −47.47.
  The gap grows monotonically with severity: −0.25, +1.30, +2.72.
- **Sensor drift**: tardiness down 35% at drift high, 1.31 against 2.01.
- **Packet loss is mixed**, and the paper explains why. LOCF imputation produces
  a misleadingly consistent stale history at moderate loss rates that the LSTM
  encodes confidently but incorrectly.
- **The severe preset is a loss and is reported as one.** MLP-noisy wins on both
  makespan (89.7 against 91.0) and tardiness (1.35 against 1.92). Severe
  activates all three noise modes at once including packet loss at p = 0.20,
  which is the regime where LOCF hurts.

| condition | t | p | Cohen's d |
|---|---|---|---|
| delay low | +4.21 | 0.014 | 1.88 |
| delay med | +8.93 | 0.001 | 3.99 |
| delay high | +23.06 | <0.001 | 10.31 |
| drift low | +2.18 | 0.047 | 0.98 |
| drift med | +6.44 | 0.003 | 2.88 |
| drift high | +44.20 | <0.001 | 19.77 |

Positive t means FT-RLS is better. The across-condition effect (p = 0.55) is not
significant, consistent with the mixed packet-loss results. With n = 5 these are
strong preliminary evidence for the large effects, not definitive confirmation.

## The IIoT noise model

Contribution C3, and the piece most reusable on its own: a Gymnasium
`ObservationWrapper` that corrupts the observation vector before the agent sees
it, with 11 named presets.

| preset | p_loss | sigma | delay mode |
|---|---|---|---|
| clean | 0.00 | 0.00 | none |
| loss_low / med / high | 0.05 / 0.20 / 0.40 | 0.00 | none |
| drift_low / med / high | 0.00 | 0.01 / 0.05 / 0.15 | none |
| delay_low / med / high | 0.00 | 0.00 | markov, lambda1 = 1 / 3 / 5 |
| severe | 0.20 | 0.05 | markov, lambda1 = 5 |

Corruption is applied in the order delay, then drift, then packet loss.

**Packet loss.** Each feature is dropped independently with probability `p_loss`.
Dropped entries are filled by last-observation carry-forward and a binary
missingness mask is concatenated to the observation, so the agent can distinguish
"sensor read zero" from "no reading arrived." This doubles the observation width.

**Sensor drift.** Additive zero-mean Gaussian noise. See point 4 of
`RECONCILIATION.md` for how this differs from the paper's definition.

**Bursty delay.** A two-state Markov-modulated Poisson process. A normal state
and a congested state with transition probabilities `q01` and `q10`; the delay is
drawn from a Poisson whose rate depends on the state, so stale observations
arrive in bursts rather than independently. The buffer holds up to
`max(4 * lambda1, 20)` past observations.

Using it on its own:

```python
from noise.noise_wrapper import make_noisy_env
env = make_noisy_env(n_jobs=10, n_machines=5, noise_preset="delay_high", seed=0)
```

Or wrap any Gymnasium env whose observation is a flat Box:

```python
from noise.noise_wrapper import IIoTNoiseWrapper
env = IIoTNoiseWrapper(base_env, p_loss=0.2, sigma=0.05,
                       delay_mode="markov", lam1=5, seed=0)
```

## Architecture

```
IIoT physical layer (machines, sensors, AGVs)  ->  true state s_t
          |
   noise channel Omega                         ->  corrupted o_t
          |
   LSTM belief estimator                       ->  b_t in R^128
          |
   PPO agent + invalid action masking          ->  pi(a_t | b_t)
```

The exact Bayes filter is intractable at this scale, so the belief is
approximated by a 2-layer LSTM with hidden size 256, projected to a 128-dim
belief through LayerNorm and tanh. Its input is the LOCF-imputed observation
concatenated with the missingness indicator. Parameters update jointly with PPO;
there is no separate pre-training stage.

Invalid action masking sets infeasible job-machine pairs to −inf before the
softmax, cutting the effective action space 40 to 60% per step so every sampled
action is feasible. Inference overhead is 0.49 ms per step.

## Environment

A fresh random FJSP instance per episode: 10 jobs, 5 machines, operations per job
U{2,5}, processing times U(1,10), due dates U(0.8,1.5) x mean-work, weights
U(0.5,2.0), objective weight alpha = 0.5.

The observation holds three features per machine (idle time, utilisation, queue
length), four per job (remaining ops, time to due, weight, started flag) and two
global (normalised clock, scheduled fraction). At 10 jobs and 5 machines that is
57 values, which the noise wrapper doubles to 114. The true state is used only
for the reward.

The reward is a shaped delta on the joint objective, so there is a dense signal
at every scheduling decision rather than only at episode end.

## Layout

```
ftrl_scheduler/
├── env/fjsp_env.py            FJSP Gymnasium environment
├── noise/noise_wrapper.py     IIoT noise wrapper, 3 modes, 11 presets
├── models/belief_policy.py    LSTM belief estimator + PPO policy
├── baselines/heuristics.py    SPT, EDD, FIFO, random
├── experiments/train.py       training and evaluation
├── experiments/test_pipeline.py   smoke test
└── results/                   paper_results.json (real), demo_results.json (smoke test)
paper/gen_figures_peeiacon.py  regenerates Figs. 2-6; holds the per-condition data
figures/                       the five paper figures
```

## Running it

```bash
pip install -r requirements.txt
cd ftrl_scheduler
python experiments/test_pipeline.py      # smoke test, expect ALL TESTS PASSED
```

A single preset:

```bash
python experiments/train.py --preset delay_high --total_steps 500000 \
    --n_seeds 5 --n_eval 20 --n_envs 4
```

The full grid as the paper ran it. Note that the script's defaults are smaller
than this, so pass these flags explicitly:

```bash
python experiments/train.py --preset all --total_steps 500000 \
    --n_seeds 5 --n_eval 20 --n_envs 4
```

That is 11 presets x 5 seeds x 2 trained agents = 110 training runs. Results are
written incrementally to `results/results.json` after each preset, so an
interrupted run does not lose everything.

| parameter | value |
|---|---|
| LSTM layers / hidden | 2 / 256 |
| belief dimension | 128 |
| learning rate | 3e-4 |
| rollout / batch | 2048 / 256 |
| discount / GAE lambda | 0.99 / 0.95 |
| PPO clip / entropy | 0.20 / 0.01 |
| training steps | 5e5 |
| seeds | 5 |

## Known limitations

1. n = 5 seeds. Strong for the d > 10 effects, weak for small ones. Ten or more
   with Bonferroni correction is the right next step.
2. LOCF imputation drives the packet-loss and severe results. That explanation is
   mechanistically plausible but has not been tested by ablating the imputation
   method on its own. A learned imputation module such as SAITS is the natural
   next experiment.
3. Brandimarte validation shows architectural generalisation to fixed-eligibility
   instances, not zero-shot transfer to unseen noise regimes.
4. The 500k-step budget is below the 1 to 2M typical in published DRL-FJSP work,
   and the SPT gap narrows with larger budgets.
5. Seed 0 of FT-RLS under the clean condition is an outlier (MS 99.8, TD 10.91)
   and inflates clean-condition variance. It does not affect the
   noise-degradation findings, which use independent per-condition seeds.
6. Three experiments reported in the paper are not implemented here. See
   `RECONCILIATION.md`, point 5.

## Citation

R. Ahmed, "Fault-Tolerant Reinforcement Learning for Real-Time IoT-Driven
Scheduling in Smart Manufacturing: A POMDP-Based Belief State Approach," in Proc.
IEEE Int. Conf. Power, Electrical, Electronics and Industrial Applications
(PEEIACON), Dhaka, Bangladesh, 2026.

## License

MIT, see `LICENSE`.
