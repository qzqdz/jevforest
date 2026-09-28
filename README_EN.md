# jevforest

Language: [中文](README.md) | English

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![pytest](https://img.shields.io/badge/tests-pytest-green.svg)](#)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](./LICENSE)

**Bagged IG trees that vote on the next budgeted question, not only on y.**

```text
FeatureTable / tabular rows
        ↓ freeze quantile bins on the full train set
   T bootstrap IG trees
        ↓ act(): path-walk + ig_weighted vote
   next feature to buy
        ↓ budget spent
      predict() + optional trace
```

```mermaid
flowchart LR
  A["tabular rows"] --> B[freeze bins]
  B --> C["T IG trees"]
  C --> D["act: path-walk vote"]
  D --> E["predict at budget"]
```

Requires sibling [`jevtree`](https://github.com/qzqdz/jevtree) for the protocol and predictor. `jevforest decide` is not implemented.

---

## Five-minute quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ../jevtree
pip install -e .

# One-shot cube smoke
./scripts/quickstart.sh

# Or run eval by hand
python -m jevforest eval-afa --config configs/eval_cube_forest.json
```

MiniBooNE fair table (same `logistic_impute` as jevtree):

```bash
python -m jevforest eval-afa --config configs/eval_miniboone_forest_logistic.json
```

Multi-seed sampling:

```bash
python scripts/sample_reliability.py --dataset miniboone --seeds 0,1,2
```

Outputs land in `results/<dataset>_ig_forest_<timestamp>/`:

| File | Contents |
|------|----------|
| `summary.json` | Acc / F1 @ budget |
| `episodes.jsonl` | Per-episode acquisition traces |

## CLI

```bash
jevforest eval-afa --config configs/…   # hard-budget AFA
jevforest version
jevforest decide                        # not implemented
```

## Benchmark results

On MiniBooNE, forest leads jevtree Disc / IG_static from budget 10; Acc@5 still trails Disc. Cube Acc@3 is saturated and is not method evidence. Full tables, CIs and ablations: [`docs/RESULTS.md`](docs/RESULTS.md).

### MiniBooNE · Acc@budget (seed 0, logistic_impute)

| Budget | Forest (`ig_weighted`) | Disc | IG_static | Random |
|-------:|-----------------------:|-----:|----------:|-------:|
| 5 | 0.798 | **0.824** | 0.814 | 0.746 |
| 10 | **0.856** | 0.820 | 0.822 | 0.766 |
| 20 | **0.878** | 0.814 | 0.822 | 0.760 |
| 40 | **0.894** | 0.828 | 0.814 | 0.800 |

### Cube · Acc@3

| Policy | Acc@3 |
|--------|------:|
| `ig_forest` | 1.000 |
| jevtree `ig_static` | 1.000 |
| jevtree `sequential` | 1.000 |
| jevtree `random` | 0.766 |

## Environment

Python ≥ 3.11. Eval does not need an LLM. The optional Jev client reads `OPENROUTER_API_KEY` from `.env` (model default `~typesafe/jev-latest`) and answers typed noul / choice / score questions; it does not replace `eval-afa`.

More scripts: [`examples/README.md`](examples/README.md).
