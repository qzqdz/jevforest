# Changelog

## 0.1.0 — 2026-09-28

First public snapshot of tabular forest AFA.

- `ForestAcquisitionPolicy` / `IGDecisionForestGrower`: bagged IG trees, frozen train bins, path-walk acquisition.
- Default acquisition vote: `ig_weighted` (votes × global IG). `plurality` and `oob` remain available.
- `jevforest eval-afa` for cube and MiniBooNE (shared jevtree predictor).
- `scripts/sample_reliability.py`: multi-seed sampling + episode bootstrap CI.
- `JevDecisionsClient`: OpenRouter Decisions API wrapper (`OPENROUTER_API_KEY`). Does not implement `decide`.
- Documented MiniBooNE seed-0 Acc@10 = 0.856 vs jevtree Disc 0.820 under the same protocol.

Not in this release: `jevforest decide`, any2jevclass / one-sentence rule synthesis.
