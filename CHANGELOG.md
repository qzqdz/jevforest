# Changelog

## 0.1.0 — 2026-09-29

Public snapshot: tabular forest AFA plus N0–N4 JevClass path.

- `ForestAcquisitionPolicy` / `IGDecisionForestGrower`: bagged IG trees, frozen train bins, path-walk acquisition.
- Default acquisition vote: `ig_weighted`. MiniBooNE seed-0 Acc@10 = 0.856 vs jevtree Disc 0.820.
- N0–N4: v2 DAG runtime, `author` / `run` / `eval-synth`, frozen-evaluator search, 0/1/3/5-shot isolation.
- Default synthesis provider is `stub_keyword` (pipeline consistency, not live Jev quality). `--provider live` uses OpenRouter.
- `JevDecisionsClient` for typed noul / choice / score.

Not in this release: `jevforest decide`.

