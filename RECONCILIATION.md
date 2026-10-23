# Paper and code reconciliation

Differences between what the paper reports and what the code in this repository
currently does. None of these are hidden; they are written down here so anyone
comparing the two does not have to work them out, and so the gaps that matter can
be closed.

The paper is **PEEIACON 2026** (6 pages), presented August 2026.

## 1. The reported win counts do not follow from the paper's own data

The conference paper is published, so this cannot be fixed there. This file is
the correction, and these are the figures to carry into any future version of
this work.

The paper states that FT-RLS beats MLP-noisy on makespan in **7 of 11**
conditions and on weighted tardiness in **8 of 11**. The claim appears in the
abstract, in Fig. 6 and in the conclusion.

The per-condition means that produced Figs. 2 to 6 are embedded in
`paper/gen_figures_peeiacon.py` and are reproduced in
`results/paper_results.json`. Counting wins directly from those numbers gives:

| metric | from the data | paper claims |
|---|---|---|
| makespan | 5 / 11 | 7 / 11 |
| weighted tardiness | 4 / 11 | 8 / 11 |
| episode reward | 4 / 11 | 7 / 11 |

Per condition, lower is better for makespan and tardiness:

| condition | FT-RLS MS | MLP MS | MS win | FT-RLS TD | MLP TD | TD win |
|---|---|---|---|---|---|---|
| clean | 91.12 | 89.08 | MLP | 2.826 | 1.279 | MLP |
| loss_low | 91.97 | 88.06 | MLP | 2.256 | 1.633 | MLP |
| loss_med | 93.82 | 87.57 | MLP | 3.029 | 2.926 | MLP |
| loss_high | 88.00 | 89.81 | **FT-RLS** | 1.777 | 0.892 | MLP |
| drift_low | 90.75 | 91.17 | **FT-RLS** | 2.164 | 1.639 | MLP |
| drift_med | 89.65 | 89.31 | MLP | 2.298 | 1.458 | MLP |
| drift_high | 88.53 | 90.00 | **FT-RLS** | 1.312 | 2.012 | **FT-RLS** |
| delay_low | 91.38 | 89.95 | MLP | 1.429 | 2.351 | **FT-RLS** |
| delay_med | 88.31 | 90.39 | **FT-RLS** | 1.765 | 2.280 | **FT-RLS** |
| delay_high | 88.37 | 92.14 | **FT-RLS** | 1.135 | 2.796 | **FT-RLS** |
| severe | 91.01 | 89.73 | MLP | 1.916 | 1.352 | MLP |

Two further signs that the counts were not computed from this data:

- In `gen_figures.py`, Fig. 6's win counts are **hard-coded**: `wins=[7,8,7]`.
  They are not derived from the arrays in the same file.
- Fig. 5 draws a green outline wherever `FTRL_TD[j] < MLP_TD[j]`. That condition
  is true four times, and the published figure shows four outlines, while its
  caption reads "green outlines = FT-RLS wins (8/11 conditions)."

The rest of the results reconcile exactly, which is why this stands out rather
than suggesting a wholesale problem:

- Table IV per-condition means match `paper_results.json` to the digit.
- Table V per-family averages match to the digit.
- The delay-high reward gap is 2.72, the FT-RLS gain from clean to delay high is
  +2.22, and the MLP-noisy loss is −2.29, all exactly as reported.

One smaller arithmetic point in the same paragraph: the paper gives the total
performance swing at delay high as 4.72 units. From the clean gap of −1.79 and
the delay-high gap of +2.72 the swing is **4.51**.

**What to do.** The conference paper is published and cannot be revised, so this
file is the correction. The directional claim the paper actually rests on
survives: FT-RLS wins the whole bursty-delay family on both metrics and loses the
packet-loss family. That is the degradation result, and it is supported by
Table V, the t-tests and Fig. 4. Only the win-count summary needs recomputing,
from the full `results.json`, and any future version of this work should carry
the corrected figures.

## 1b. `results/demo_results.json` is not the paper's results

This file is smoke-test output. FT-RLS makespan is identical (81.339) across
`clean`, `loss_med`, `drift_med`, `delay_med` and `severe`, which means a single
agent was evaluated without per-preset retraining, and SPT reads 57.06 against
70.0 in the paper.

`results/paper_results.json` now holds the real per-condition means, recovered
from the figure script. It is condition-level, not raw per-seed data, so the
SEM values in Table IV cannot be rebuilt from it. The full `results.json` written
by `experiments/train.py --preset all` is still the file worth committing.

## 2. Default run settings are smaller than the paper's

| setting | `train.py` default | paper |
|---|---|---|
| training steps | 200,000 | 500,000 |
| seeds | 3 | 5 |
| evaluation episodes | 10 | 20 |

All three are command-line arguments, so the paper's configuration is reachable:

```bash
python experiments/train.py --preset all --total_steps 500000 --n_seeds 5 --n_eval 20
```

**To close:** either change the defaults to match the paper, or state the exact
command used, which is what the Quick Start in the README now does.

## 3. MMPP transition probability

`noise_wrapper.py` defaults to `q01 = 0.02` (normal to congested) and
`q10 = 0.30`. `NOISE_PRESETS` does not override `q01`, so 0.02 is what runs.

The paper, Section IV, describes the chain as "0.05 into congestion and
0.30 out of it." `q10 = 0.30` matches, and the stated mean congestion duration of
3.3 steps equals 1/q10, so that is consistent. `q01` is not.

**To close:** set `q01 = 0.05` in the presets if 0.05 is what produced the
results, or correct the figure in any future version of the paper. These give
different congestion frequencies, so it is worth settling which one the published
numbers used.

## 4. Drift sigma is absolute, not scaled by the feature range

The paper defines sigma as a fraction of the empirical feature range `r`, and the
preset table reads `0.01 / 0.05 / 0.15 r`. The docstring in `noise_wrapper.py`
says the same thing.

The implementation does not multiply by `r`:

```python
obs = obs + self._rng.normal(0.0, self.sigma, size=obs.shape)
```

In practice the gap is small, because `fjsp_env._get_obs()` normalises every
feature (utilisations and fractions to roughly [0,1], slack clipped to [-2,2]),
so `r` is near 1 for most features. But it is not exactly what the paper
describes, and for any feature whose range is not 1 the noise level differs.

**To close:** either multiply by a measured per-feature range, or change the
docstring and paper wording to "absolute standard deviation on the normalised
observation."

## 5. Three paper experiments have no code here

| paper element | where | status |
|---|---|---|
| GRU-noisy baseline (−45.98 at delay high) | Section VI | not implemented |
| Brandimarte MK01/MK03/MK05/MK10 transfer | Table VIII | no instance loader |
| Ablation: 1-layer LSTM, EMA filter | Table VII | not implemented |

`train.py` trains FT-RLS, MLP-noisy and PPO-clean, and evaluates SPT, EDD and
FIFO. The GRU baseline, the Brandimarte loader and the two ablation variants are
absent.

**To close:** add them, or mark those three results as not reproducible from this
repository. The GRU baseline matters most, because it is the evidence that the
structured belief projection adds something beyond bare recurrence.

## 6. Observation dimension

The paper states the observation is `o_t ∈ R^35`, with three features
per machine, four per job and two global. At the paper's own configuration of 10
jobs and 5 machines that arithmetic gives

    3 x 5 + 4 x 10 + 2 = 57

which is what `fjsp_env.py` produces. The code is self-consistent and 57 is the
correct number for that configuration; the 35 in the paper does not follow from
the feature counts it lists.

The belief estimator's input is twice the observation width, since the
missingness indicator is concatenated, so the LSTM sees 114 inputs at this
configuration.

**To close:** nothing in the code. Worth correcting in any future version of the
paper.

## 7. Baseline sets differ between paper and code

The code includes `PPO-clean` (an upper bound on clean observations) and `FIFO`,
which the PEEIACON paper does not report. The paper reports `GRU-noisy`, which
the code does not implement.

**To close:** no action needed, but a reader comparing the two will notice.
