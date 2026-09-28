# Benchmark results

Hard-budget numbers come from `jevforest eval-afa` (official seed=0) and
`scripts/sample_reliability.py` (multi-seed). Do not paste sklearn RF / XGBoost
scores here. Cube Acc@3 is saturated — it is a smoke test, not method evidence.

## Protocol

| Item | Value |
|------|--------|
| Scoring | `jevtree.eval.metrics` via `HardBudgetProtocol` (prefix scoring ≡ protocol) |
| MiniBooNE | UCI PID, `n_train=2000`, `n_test=500`, 50 features, `logistic_impute` |
| Cube | `cube_without_noise`, `n_samples=256`, `n_test=77`, `match_majority` |
| Locked policy | `ig_forest`, T=16, `vote=ig_weighted`, MiniBooNE `max_features=sqrt`, cube `max_features=all` |
| Reliability | 5 cube seeds; 3 MiniBooNE seeds; 1000× episode bootstrap CI on seed 0 |

jevtree baselines below are from `jevtree/docs/RESULTS.md` (same predictor and split recipe, seed 0).

## MiniBooNE — official seed 0

Snapshot: [`docs/snapshots/miniboone_ig_forest_20260928T170020Z__summary.json`](snapshots/miniboone_ig_forest_20260928T170020Z__summary.json)

| Budget | Forest Acc | Forest F1 | Disc Acc | IG_static Acc | Random Acc | Sequential Acc |
|-------:|-----------:|----------:|---------:|--------------:|-----------:|---------------:|
| 5 | **0.798** | 0.659 | 0.824 | 0.814 | 0.746 | 0.724 |
| 10 | **0.856** | **0.797** | 0.820 | 0.822 | 0.766 | 0.728 |
| 20 | **0.878** | **0.840** | 0.814 | 0.822 | 0.760 | 0.792 |
| 40 | **0.894** | **0.867** | 0.828 | 0.814 | 0.800 | 0.822 |

Seed-0 bootstrap 95% CI (n_test=500): Acc@5 [0.762, 0.834]; Acc@10 [0.826, 0.886]; Acc@20 [0.848, 0.906]; Acc@40 [0.866, 0.922].

Forest trails discriminative IG at budget 5, then leads from budget 10 onward.

## MiniBooNE — 3-seed reliability (`ig_weighted`, T=16, sqrt)

Snapshot: [`docs/snapshots/sample_miniboone_round3_lock.json`](snapshots/sample_miniboone_round3_lock.json)

| Budget | Acc mean±std | Acc range | F1 mean±std |
|-------:|-------------:|----------:|------------:|
| 5 | 0.785±0.021 | 0.760–0.798 | 0.615±0.040 |
| 10 | 0.845±0.010 | 0.836–0.856 | 0.770±0.025 |
| 20 | 0.869±0.013 | 0.854–0.878 | 0.823±0.015 |
| 40 | 0.885±0.008 | 0.878–0.894 | 0.850±0.015 |

Acc@5 always beat the jevtree random baseline (0.746). Acc@40 is the most stable (±0.008).

Round-1 plurality (before IG-weighted default): Acc@5 0.779±0.012, Acc@10 0.810±0.013. Locked `ig_weighted` keeps Acc@5 and lifts Acc@10 by ~3.5 points.

## Vote-rule ablation (MiniBooNE seed 0)

| Vote | max_features | Acc@5 | Acc@10 | Acc@20 | Acc@40 |
|------|--------------|------:|-------:|-------:|-------:|
| plurality | sqrt | 0.798 | 0.836 | **0.882** | 0.894 |
| **ig_weighted** | **sqrt** | **0.798** | **0.856** | 0.878 | 0.894 |
| oob | sqrt | 0.774 | 0.802 | 0.854 | 0.894 |
| oob_ig | sqrt | 0.798 | 0.856 | 0.878 | 0.894 |
| plurality | all | 0.770 | 0.816 | 0.854 | **0.914** |
| ig_weighted | all | 0.790 | 0.822 | 0.860 | **0.914** |

Discarded: OOB-weighted plurality (hurts low budget). `max_features=all` helps Acc@40 but loses Acc@5. Kept: `ig_weighted` + `sqrt`.

## Cube without noise

Official seed 0 snapshot: [`docs/snapshots/cube_without_noise_ig_forest_20260928T170017Z__summary.json`](snapshots/cube_without_noise_ig_forest_20260928T170017Z__summary.json)

| Policy | Acc@1 | Acc@2 | Acc@3 |
|--------|------:|------:|------:|
| ig_forest (locked) | 0.688 | 0.909 | **1.000** |
| jevtree `ig_static` | 0.688 | 0.909 | **1.000** |
| jevtree `sequential` | 0.688 | 0.909 | **1.000** |
| jevtree `random` | 0.701 | 0.740 | 0.766 |

5-seed forest: Acc@3 = 1.000±0.000; Acc@2 = 0.839±0.139 (n_test=77, seed 3 dropped to 0.597). Cube@3 is solved by any informed order of three features. Do not cite it as a forest advantage.

## Reproduce

```bash
python -m jevforest eval-afa --config configs/eval_cube_forest.json
python -m jevforest eval-afa --config configs/eval_miniboone_forest_logistic.json

python scripts/sample_reliability.py --dataset cube --seeds 0,1,2,3,4 \
  --vote ig_weighted --max-features all --out results/sample_cube.json
python scripts/sample_reliability.py --dataset miniboone --seeds 0,1,2 \
  --vote ig_weighted --max-features sqrt --out results/sample_miniboone.json
```
