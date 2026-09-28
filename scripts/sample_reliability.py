#!/usr/bin/env python3
"""Multi-seed sampling reliability for forest AFA.

Prefix scoring is equivalent to HardBudgetProtocol when act() is deterministic
given the current mask: one max-budget episode, then predict on each prefix.
Use --mode protocol to call the official loop (slower, same numbers).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def _mean_std(xs: list[float]) -> tuple[float, float]:
    if not xs:
        return float("nan"), float("nan")
    m = sum(xs) / len(xs)
    if len(xs) == 1:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return m, math.sqrt(var)


def bootstrap_ci(correct: list[int], *, n_boot: int = 1000, seed: int = 0) -> dict[str, float]:
    n = len(correct)
    if n == 0:
        return {"mean": float("nan"), "boot_std": float("nan"), "ci95_lo": float("nan"), "ci95_hi": float("nan")}
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(n_boot):
        acc = sum(correct[rng.randrange(n)] for _ in range(n)) / n
        means.append(acc)
    means.sort()
    obs = sum(correct) / n
    lo = means[int(0.025 * n_boot)]
    hi = means[min(n_boot - 1, int(0.975 * n_boot))]
    _, std = _mean_std(means)
    return {"mean": obs, "boot_std": std, "ci95_lo": lo, "ci95_hi": hi}


def _policy_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    mf: Any = args.max_features
    if mf.replace(".", "", 1).isdigit():
        mf = float(mf) if "." in mf else int(mf)
    return dict(
        n_estimators=args.n_estimators,
        max_features=mf,
        max_depth=args.max_depth,
        min_samples=args.min_samples,
        n_bins=args.n_bins,
        criterion=args.criterion,
        bootstrap=not args.no_bootstrap,
        force_acquisition=True,
        predictor_name=args.predictor,
        vote=args.vote,
    )


def load_split(dataset: str, seed: int, args: argparse.Namespace) -> dict[str, Any]:
    if dataset == "cube":
        from jevtree.data.cube import load_cube_split

        split = load_cube_split(
            n_samples=args.n_samples, n_features=args.n_features, seed=seed
        )
        split["continuous_keys"] = []
        return split

    from jevtree.data.tabular import load_tabular_split

    return load_tabular_split(
        dataset,
        seed=seed,
        n_train=args.n_train,
        n_test=args.n_test,
        cache_dir=args.cache_dir,
        allow_fallback=args.allow_fallback,
    )


def run_prefix(
    policy: Any,
    test_rows: list[dict[str, Any]],
    feature_keys: list[str],
    label_key: str,
    budgets: list[int],
) -> dict[str, Any]:
    from jevtree.eval.metrics import accuracy_at_budget, f1_at_budget

    max_b = max(budgets)
    fkeys = list(feature_keys)
    records: list[dict[str, Any]] = []
    for ep_i, row in enumerate(test_rows):
        feats = [row[k] for k in fkeys]
        lab = row[label_key]
        n = len(feats)
        feature_mask = [False] * n
        masked: list[Any] = [None] * n
        acquired: list[int] = []
        policy.force_acquisition = True
        while len(acquired) < max_b and not all(feature_mask):
            action = int(policy.act(masked_features=masked, feature_mask=feature_mask))
            if action == 0:
                break
            idx = action - 1
            if idx < 0 or idx >= n or feature_mask[idx]:
                break
            feature_mask[idx] = True
            masked[idx] = feats[idx]
            acquired.append(idx)
        preds: dict[int, Any] = {}
        for b in budgets:
            m2 = [False] * n
            x2: list[Any] = [None] * n
            for j in acquired[:b]:
                m2[j] = True
                x2[j] = feats[j]
            preds[b] = policy.predict(masked_features=x2, feature_mask=m2)
        records.append(
            {
                "episode_index": ep_i,
                "acquired": acquired,
                "label": lab,
                "preds": {str(b): preds[b] for b in budgets},
            }
        )

    curve = []
    for b in budgets:
        y_true = [r["label"] for r in records]
        y_pred = [r["preds"][str(b)] for r in records]
        correct = [int(a == p) for a, p in zip(y_true, y_pred)]
        ci = bootstrap_ci(correct, n_boot=1000, seed=0)
        curve.append(
            {
                "budget": b,
                "accuracy": accuracy_at_budget(y_true, y_pred, budget=b),
                "f1": f1_at_budget(y_true, y_pred, budget=b),
                "n_episodes": len(records),
                "bootstrap": ci,
            }
        )
    return {"curve": curve, "n_test": len(records)}


def run_protocol(
    policy: Any,
    test_rows: list[dict[str, Any]],
    feature_keys: list[str],
    label_key: str,
    budgets: list[int],
    dataset_id: str,
    dataset_hash: str,
    seed: int,
) -> dict[str, Any]:
    from jevtree.eval.protocol import HardBudgetEpisodeConfig, HardBudgetProtocol

    cfg = HardBudgetEpisodeConfig(
        dataset_id=dataset_id,
        hard_budget=budgets[0],
        split_seed=seed,
        budget_schedule=budgets,
        dataset_hash=dataset_hash,
        policy_name="ig_forest",
        feature_keys=feature_keys,
        label_key=label_key,
    )
    summary = HardBudgetProtocol().run_eval(
        policy,
        cfg,
        test_rows=test_rows,
        feature_keys=feature_keys,
        label_key=label_key,
        budget_schedule=budgets,
    )
    summary.pop("episodes", None)
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Forest AFA multi-seed sampling")
    p.add_argument("--dataset", choices=["cube", "miniboone"], required=True)
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--mode", choices=["prefix", "protocol"], default="prefix")
    p.add_argument("--n-estimators", type=int, default=16)
    p.add_argument("--max-features", default="sqrt")
    p.add_argument("--max-depth", type=int, default=6)
    p.add_argument("--min-samples", type=int, default=4)
    p.add_argument("--n-bins", type=int, default=4)
    p.add_argument("--criterion", default="gain")
    p.add_argument("--no-bootstrap", action="store_true")
    p.add_argument("--predictor", default="match_majority")
    p.add_argument("--n-samples", type=int, default=256)
    p.add_argument("--n-features", type=int, default=5)
    p.add_argument("--n-train", type=int, default=2000)
    p.add_argument("--n-test", type=int, default=500)
    p.add_argument("--budgets", default="")
    p.add_argument("--cache-dir", default=str(ROOT / "data" / "cache"))
    p.add_argument("--allow-fallback", action="store_true")
    p.add_argument("--vote", default="ig_weighted")
    p.add_argument("--tag", default="")
    p.add_argument("--out", default="")
    return p.parse_args()


def main() -> int:
    from jevforest.policy.forest_afa import ForestAcquisitionPolicy

    args = parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]
    if args.dataset == "cube":
        budgets = [int(x) for x in (args.budgets or "1,2,3").split(",")]
        if args.max_features == "sqrt" and "--max-features" not in sys.argv:
            args.max_features = "all"
        if args.min_samples == 4 and "--min-samples" not in sys.argv:
            args.min_samples = 1
        if args.predictor == "match_majority" and "--predictor" not in sys.argv:
            args.predictor = "match_majority"
    else:
        budgets = [int(x) for x in (args.budgets or "5,10,20,40").split(",")]
        if args.max_depth == 6 and "--max-depth" not in sys.argv:
            args.max_depth = 8
        if args.min_samples == 4 and "--min-samples" not in sys.argv:
            args.min_samples = 8
        if args.predictor == "match_majority" and "--predictor" not in sys.argv:
            args.predictor = "logistic_impute"

    per_seed: list[dict[str, Any]] = []
    t0 = time.time()
    for seed in seeds:
        st = time.time()
        split = load_split(args.dataset, seed, args)
        train, test = split["train"], split["test"]
        fkeys, lkey = split["feature_keys"], split["label_key"]
        kw = _policy_kwargs(args)
        kw["seed"] = seed
        cont = list(split.get("continuous_keys") or [])
        if args.dataset != "cube":
            kw["continuous_keys"] = cont or None
        policy = ForestAcquisitionPolicy(**kw)
        policy.fit(train, lkey, fkeys)
        if args.mode == "protocol":
            summary = run_protocol(
                policy,
                test,
                fkeys,
                lkey,
                budgets,
                split["dataset_id"],
                split["dataset_hash"],
                seed,
            )
            curve = summary.get("curve") or []
        else:
            summary = run_prefix(policy, test, fkeys, lkey, budgets)
            curve = summary["curve"]
        rec = {
            "seed": seed,
            "n_train": len(train),
            "n_test": len(test),
            "n_features": len(fkeys),
            "dataset_id": split.get("dataset_id"),
            "dataset_hash": split.get("dataset_hash"),
            "fallback_used": split.get("fallback_used"),
            "elapsed_sec": round(time.time() - st, 3),
            "curve": curve,
        }
        per_seed.append(rec)
        pts = " ".join(
            f"b{c['budget']}={c['accuracy']:.4f}" for c in curve
        )
        print(f"seed={seed}  {pts}  {rec['elapsed_sec']}s", flush=True)

    # aggregate
    agg = []
    for i, c0 in enumerate(per_seed[0]["curve"]):
        b = c0["budget"]
        accs = [s["curve"][i]["accuracy"] for s in per_seed]
        f1s = [s["curve"][i]["f1"] for s in per_seed]
        am, asd = _mean_std(accs)
        fm, fsd = _mean_std(f1s)
        agg.append(
            {
                "budget": b,
                "acc_mean": am,
                "acc_std": asd,
                "acc_min": min(accs),
                "acc_max": max(accs),
                "f1_mean": fm,
                "f1_std": fsd,
                "seeds": accs,
            }
        )

    out = {
        "dataset": args.dataset,
        "mode": args.mode,
        "vote": args.vote,
        "n_estimators": args.n_estimators,
        "max_features": args.max_features,
        "max_depth": args.max_depth,
        "min_samples": args.min_samples,
        "predictor": args.predictor if args.dataset != "cube" else args.predictor,
        "seeds": seeds,
        "budgets": budgets,
        "tag": args.tag or None,
        "elapsed_sec": round(time.time() - t0, 3),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "aggregate": agg,
        "per_seed": per_seed,
    }
    out_path = Path(
        args.out
        or (
            ROOT
            / "results"
            / f"sample_{args.dataset}_{args.tag or args.vote}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
        )
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")
    print("aggregate:")
    for row in agg:
        print(
            f"  budget={row['budget']:>3}  acc={row['acc_mean']:.4f}±{row['acc_std']:.4f}  "
            f"range=[{row['acc_min']:.4f},{row['acc_max']:.4f}]  "
            f"f1={row['f1_mean']:.4f}±{row['f1_std']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
