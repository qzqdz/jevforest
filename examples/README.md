# examples

Demos only — not a leaderboard.

```bash
python -m jevforest eval-afa --config configs/eval_cube_forest.json
python -m jevforest eval-afa --config configs/eval_miniboone_forest_logistic.json

python -m jevforest author --goal "Route support tickets to billing technical sales" --out results/authored.json
python -m jevforest run --jevclass results/authored.json --input '{"text":"Please refund the payout"}'
python -m jevforest eval-synth --task-dir examples/tasks/tickets --shot 0
```

Cube Acc@3 is saturated. Sample AFA output: [`../results/sample/cube_eval_summary.json`](../results/sample/cube_eval_summary.json).
