# jevforest

Language: [中文](README.md) | English

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![pytest](https://img.shields.io/badge/tests-pytest-green.svg)](#)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](./LICENSE)

**Bagged IG trees that vote on the next budgeted question, not only on y.** Under a hard feature budget, each tree walks to its first unobserved node; votes are aggregated with information-gain weighting. Evaluation `predict()` uses the shared jevtree predictor so numbers are comparable to the single-tree AFA table.

```text
FeatureTable / tabular rows
        ↓ freeze quantile bins on the full train set
   T bootstrap IG trees (optional max_features)
        ↓ act(): path-walk + ig_weighted vote
   next feature to buy  →  predict() when the budget is spent
```

**What ships today is tabular AFA.** `jevforest decide` and few-shot any2jevclass (one-sentence Jev rule synthesis) are planned, not implemented. The optional [Jev Decisions client](#jev-decisions-api) wraps OpenRouter typed questions; it does not build a rule system.

Numbers live in [docs/RESULTS.md](docs/RESULTS.md). On MiniBooNE with shared `logistic_impute`, Acc@10=0.856 beats jevtree Disc / IG_static under the same protocol; Acc@5 still trails Disc. Cube Acc@3=1.0 is a saturated smoke test, not method evidence.

## Install

Requires sibling [`../jevtree`](../jevtree) for the protocol and predictor.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ../jevtree
pip install -e .
```

Optional: copy `.env.example` to `.env` and set `OPENROUTER_API_KEY` (Jev client only).

## Five minutes

```bash
python -m jevforest version
python -m jevforest eval-afa --config configs/eval_cube_forest.json
```

MiniBooNE fair table (same `logistic_impute` as jevtree; cache under `data/cache/`):

```bash
python -m jevforest eval-afa --config configs/eval_miniboone_forest_logistic.json
```

Multi-seed sampling:

```bash
python scripts/sample_reliability.py --dataset miniboone --seeds 0,1,2 \
  --vote ig_weighted --max-features sqrt
```

## Algorithm

1. Freeze `bin_edges_` on the training set, then grow T bootstrap / `max_features` trees with `IGDecisionTreeGrower`.
2. `act()` walks each tree to its first unobserved node. Default `vote=ig_weighted` (votes × global IG), then feature name. OOB-weighted plurality is available and failed the low-budget ablation.
3. Evaluation `predict()` uses the shared jevtree predictor; product `predict()` uses a soft vote over leaf counts.
4. T=1 with `max_features=all` and `bootstrap=False` matches a single IG tree on `predict()`.

Configs: `configs/eval_cube_forest.json`, `configs/eval_miniboone_forest_logistic.json`.

## Results (locked config)

MiniBooNE seed=0, `n_train=2000`, `n_test=500`, T=16, `sqrt`, `ig_weighted`:

| Budget | Forest Acc | Forest F1 | jevtree Disc Acc |
|-------:|-----------:|----------:|-----------------:|
| 5 | 0.798 | 0.659 | 0.824 |
| 10 | 0.856 | 0.797 | 0.820 |
| 20 | 0.878 | 0.840 | 0.814 |
| 40 | 0.894 | 0.867 | 0.828 |

3-seed Acc@10 = 0.845±0.010; Acc@40 = 0.885±0.008. Full tables, CIs, ablations and reproduce commands: [docs/RESULTS.md](docs/RESULTS.md).

## Jev Decisions API

Jev (`~typesafe/jev-latest`) answers typed questions about a state: noul (yes/no probability), choice, or score. Your code owns the workflow. Requires an OpenRouter key:

```python
from jevforest.sdk.jev import JevDecisionsClient

client = JevDecisionsClient()  # reads OPENROUTER_API_KEY
decision = client.decide(
    "Help! My payouts have been failing for 3 days.",
    {
        "is_urgent": {
            "type": "noul",
            "instructions": "Does this message convey urgency?",
            "criteria": {"true": "Explicitly time-sensitive", "false": "No urgency expressed"},
        }
    },
)
print(decision.get("answers"))
```

This is not `jevforest decide` and does not train the forest. Model card: [TypeSafe Jev Latest](https://openrouter.ai/typesafe/jev-latest).

## Status

| Capability | Status |
|------------|--------|
| `eval-afa` / `ForestAcquisitionPolicy` | ships |
| Multi-seed sampling script | ships |
| `JevDecisionsClient` | ships (API key required) |
| `jevforest decide` / SDK `decide()` | not implemented |
| few-shot any2jevclass | planned, not implemented |

## Tests

```bash
python -m pytest -q
```

## License

MIT. See [LICENSE](LICENSE).
