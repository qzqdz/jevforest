# jevforest

语言: 中文 | [English](README_EN.md)

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![pytest](https://img.shields.io/badge/tests-pytest-green.svg)](#)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](./LICENSE)

**Bagged IG 树对「下一个预算问题」投票，而不只对 y 投票。** 给定硬特征预算，森林沿每棵树走到第一个未观测节点，按信息增益加权聚合后提问；预测复用 jevtree 的共享 predictor，保证和单树 AFA 表可比。

```text
FeatureTable / 表格行
        ↓ 全训练集冻结 quantile bins
   T 棵 bootstrap IG 树（可选 max_features）
        ↓ act()：path-walk + ig_weighted 投票
   下一个要买的特征  →  预算用尽后 predict()
```

**表格 AFA 仍可用。** N0–N4 JevClass 路径已落地：`author` / `run` / `eval-synth`（默认 stub provider）。`jevforest decide` 仍未实现。stub 上 Acc=1.0 只说明流水线自洽，不是 live Jev 质量。详见 [docs/N0N4.md](docs/N0N4.md)。

评测数字见 [docs/RESULTS.md](docs/RESULTS.md)。MiniBooNE 在共享 `logistic_impute` 下 Acc@10=0.856，高于同协议的 jevtree Disc/IG_static；Acc@5 仍落后 Disc。Cube Acc@3=1.0 是饱和烟雾测试，不是方法证据。

## 安装

需要 sibling [`../jevtree`](../jevtree)（同版本协议与 predictor）。

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ../jevtree
pip install -e .
```

可选：复制 `.env.example` 为 `.env`，填 `OPENROUTER_API_KEY`（仅 Jev 客户端需要）。

## 五分钟

```bash
python -m jevforest version
python -m jevforest eval-afa --config configs/eval_cube_forest.json

# N0–N4: 一句话生成并跑一张工单（stub，无需 API）
python -m jevforest author --goal "Route support tickets to billing technical sales" --out results/authored.json
python -m jevforest run --jevclass results/authored.json --input '{"text":"Please refund the payout"}'
python -m jevforest eval-synth --task-dir examples/tasks/tickets --shot 0
```

MiniBooNE 公平表（与 jevtree 同一 `logistic_impute`，数据缓存 `data/cache/`）：

```bash
python -m jevforest eval-afa --config configs/eval_miniboone_forest_logistic.json
```

多 seed 抽样可靠性：

```bash
python scripts/sample_reliability.py --dataset miniboone --seeds 0,1,2 \
  --vote ig_weighted --max-features sqrt
```

## 算法

1. 在全训练集上冻结 `bin_edges_`，再 bootstrap + `max_features` 长 T 棵 `IGDecisionTreeGrower`。
2. `act()`：每棵树从根走到第一个未观测节点；默认 `vote=ig_weighted`（票数 × 全局 IG）。平票再按特征名。OOB 加权作为可选项，低预算上未通过消融。
3. 评测 `predict()` 走共享 jevtree predictor；产品 `predict()` 用叶子 `counts` 软投票。
4. T=1 且 `max_features=all`、`bootstrap=False` 时，与单棵 IG 树预测对齐。

配置入口：`configs/eval_cube_forest.json`、`configs/eval_miniboone_forest_logistic.json`。

## 结果摘要（锁定配置）

MiniBooNE seed=0，`n_train=2000`，`n_test=500`，T=16，`sqrt`，`ig_weighted`：

| Budget | Forest Acc | Forest F1 | jevtree Disc Acc |
|-------:|-----------:|----------:|-----------------:|
| 5 | 0.798 | 0.659 | 0.824 |
| 10 | 0.856 | 0.797 | 0.820 |
| 20 | 0.878 | 0.840 | 0.814 |
| 40 | 0.894 | 0.867 | 0.828 |

3-seed Acc@10 = 0.845±0.010；Acc@40 = 0.885±0.008。完整表、CI、消融与复现命令见 [docs/RESULTS.md](docs/RESULTS.md)。

## Jev Decisions API

Jev（`~typesafe/jev-latest`）回答关于 state 的类型化问题：noul（是/否概率）、choice、score。工作流由调用方代码拥有。需要 OpenRouter key：

```python
import os
from jevforest.sdk.jev import JevDecisionsClient

client = JevDecisionsClient()  # reads OPENROUTER_API_KEY
decision = client.decide(
    "Help! My payouts have been failing for 3 days.",
    {
        "is_urgent": {
            "type": "noul",
            "instructions": "Does this message convey urgency?",
            "criteria": {"true": "Explicitly time-sensitive", "false": "No urgency expressed"},
        },
        "department": {
            "type": "choice",
            "instructions": "Which team should handle this?",
            "criteria": {
                "billing": "Payments, invoicing, refunds",
                "technical": "Bugs, outages, integrations",
                "sales": "Pricing, upgrades, new accounts",
            },
        },
    },
)
print(decision.get("answers"))
```

这不是 `jevforest decide`，也不训练森林。模型卡：[TypeSafe Jev Latest](https://openrouter.ai/typesafe/jev-latest)。

## 状态与路线

| 能力 | 状态 |
|------|------|
| `eval-afa` / `ForestAcquisitionPolicy` | 可用 |
| 多 seed 抽样脚本 | 可用 |
| `JevDecisionsClient` | 可用（需 API key） |
| `jevforest decide` / SDK `decide()` | 未实现 |
| few-shot any2jevclass / 一句话建规则 | 规划中，未实现 |

## 测试

```bash
python -m pytest -q
```

## License

MIT. 见 [LICENSE](LICENSE).
