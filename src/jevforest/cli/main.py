"""Official jevforest CLI.

Research entry: `jevforest eval-afa --config …`.
Product `decide` is not implemented.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jevforest import __version__


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here] + list(here.parents):
        if (p / "pyproject.toml").is_file() and (p / "src").is_dir():
            return p
    return Path.cwd()


def _git_commit(repo: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or None
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


def _load_config(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".json", ""} or text.lstrip().startswith("{"):
        return json.loads(text)
    raise ValueError(f"jevforest eval-afa expects JSON config, got {path.suffix}")


def cmd_eval_afa(args: argparse.Namespace) -> int:
    from jevtree.data.cube import load_cube_split
    from jevtree.eval.afabench import AFABenchAdapter
    from jevtree.eval.protocol import HardBudgetEpisodeConfig, HardBudgetProtocol
    from jevtree.eval.provenance import collect_provenance
    from jevtree.policy.predictor import normalize_predictor_name

    from jevforest.policy.forest_afa import ForestAcquisitionPolicy

    if not args.config:
        print("jevforest eval-afa: --config is required", file=sys.stderr)
        return 2

    cfg_path = Path(args.config)
    if not cfg_path.is_file():
        alt = _repo_root() / args.config
        if alt.is_file():
            cfg_path = alt
        else:
            print(f"config not found: {args.config}", file=sys.stderr)
            return 2

    cfg = _load_config(cfg_path)
    dataset_id = cfg.get("dataset_id", "cube_without_noise")
    split_seed = int(cfg.get("split_seed", 0))
    hard_budgets = cfg.get("hard_budgets") or [cfg.get("hard_budget", 3)]
    hard_budgets = [int(b) for b in hard_budgets]

    policy_cfg = cfg.get("policy") or {}
    if isinstance(policy_cfg, str):
        policy_cfg = {"name": policy_cfg}

    output_cfg = cfg.get("output") or {}
    results_dir = Path(output_cfg.get("results_dir") or "results/")
    if not results_dir.is_absolute():
        results_dir = _repo_root() / results_dir

    continuous_keys: list[str] = list(policy_cfg.get("continuous_keys") or [])
    data_provenance: dict[str, Any] = {}

    if dataset_id in ("cube_without_noise", "cube"):
        n_samples = int(cfg.get("n_samples", 256))
        n_features = int(cfg.get("n_features", 5))
        split = load_cube_split(
            n_samples=n_samples, n_features=n_features, seed=split_seed
        )
        train, test = split["train"], split["test"]
        feature_keys = split["feature_keys"]
        label_key = split["label_key"]
        dataset_hash = split["dataset_hash"]
        data_provenance = {
            "loader": "cube",
            "n_samples": n_samples,
            "n_features": n_features,
        }
    elif dataset_id in ("miniboone", "diabetes", "bank_marketing"):
        from jevtree.data.tabular import load_tabular_split

        n_train = cfg.get("n_train")
        n_test = cfg.get("n_test")
        split = load_tabular_split(
            dataset_id,
            seed=split_seed,
            n_train=int(n_train) if n_train is not None else None,
            n_test=int(n_test) if n_test is not None else None,
            cache_dir=cfg.get("cache_dir"),
            allow_fallback=bool(cfg.get("allow_fallback", True)),
        )
        train, test = split["train"], split["test"]
        feature_keys = split["feature_keys"]
        label_key = split["label_key"]
        dataset_hash = split["dataset_hash"]
        continuous_keys = list(split.get("continuous_keys") or feature_keys)
        dataset_id = split["dataset_id"]
        data_provenance = {
            "loader": "tabular",
            "requested_dataset_id": split.get("requested_dataset_id"),
            "fallback_used": split.get("fallback_used"),
            "n_train": split.get("n_train"),
            "n_test": split.get("n_test"),
        }
    else:
        print(f"unsupported dataset_id: {dataset_id}", file=sys.stderr)
        return 2

    raw_pred_name = (cfg.get("predictor") or {}).get("name") or "match_majority"
    predictor_name = normalize_predictor_name(raw_pred_name)
    max_features = policy_cfg.get("max_features", "sqrt")
    if isinstance(max_features, str) and max_features.replace(".", "", 1).isdigit():
        max_features = float(max_features) if "." in max_features else int(max_features)

    policy = ForestAcquisitionPolicy(
        n_estimators=int(policy_cfg.get("n_estimators", 16)),
        max_features=max_features,
        max_depth=policy_cfg.get("max_depth", 6),
        min_samples=int(policy_cfg.get("min_samples", 4)),
        n_bins=int(policy_cfg.get("n_bins", 4)),
        seed=int(policy_cfg.get("seed", split_seed)),
        criterion=policy_cfg.get("criterion", "gain"),
        continuous_keys=continuous_keys or None,
        bootstrap=bool(policy_cfg.get("bootstrap", True)),
        force_acquisition=bool(policy_cfg.get("force_acquisition", True)),
        predictor_name=predictor_name,
        bin_seed=int(policy_cfg.get("bin_seed", 0)),
        vote=str(policy_cfg.get("vote", "ig_weighted")),
    )
    policy.fit(train, label_key, feature_keys)
    policy_name = "ig_forest"

    adapter = AFABenchAdapter(
        policy,
        force_acquisition=bool(policy_cfg.get("force_acquisition", True)),
        has_builtin_classifier=True,
        predictor_name=predictor_name,
    )

    repo = _repo_root()
    prov = collect_provenance(
        repo=repo,
        predictor_name=predictor_name,
        policy_name=policy_name,
        dataset_hash=dataset_hash,
        split_seed=split_seed,
        budgets=hard_budgets,
        config_path=str(cfg_path),
    )
    prov["jevforest_version"] = __version__
    try:
        import jevtree as _jt

        prov["jevtree_version"] = getattr(_jt, "__version__", None)
    except Exception:
        prov["jevtree_version"] = None
    prov["n_estimators"] = policy.n_estimators
    prov["max_features"] = policy.max_features
    prov["vote_rule"] = policy.vote

    ep_cfg = HardBudgetEpisodeConfig(
        dataset_id=dataset_id,
        hard_budget=hard_budgets[0],
        split_seed=split_seed,
        budget_schedule=hard_budgets,
        config_path=str(cfg_path),
        git_commit=prov.get("git_commit") or _git_commit(repo),
        dataset_hash=dataset_hash,
        policy_name=policy_name,
        feature_keys=feature_keys,
        label_key=label_key,
    )
    summary = HardBudgetProtocol().run_eval(
        adapter,
        ep_cfg,
        test_rows=test,
        feature_keys=feature_keys,
        label_key=label_key,
        budget_schedule=hard_budgets,
    )
    summary["predictor_name"] = predictor_name
    summary["data_provenance"] = data_provenance
    summary["n_train"] = len(train)
    summary["n_test"] = len(test)
    summary["n_features"] = len(feature_keys)
    summary["provenance"] = prov
    summary["git_commit"] = prov.get("git_commit")
    summary["git_dirty"] = bool(prov.get("git_dirty"))

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{dataset_id}_{policy_name}_{ts}"
    run_dir = results_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    episodes = summary.pop("episodes", [])
    with open(run_dir / "summary.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        f.write("\n")
    with open(run_dir / "episodes.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for ep in episodes:
            f.write(json.dumps(ep, ensure_ascii=False, default=str) + "\n")

    print(f"eval-afa  dataset={dataset_id}  policy={policy_name}  n_test={len(test)}")
    print(f"results → {run_dir}")
    print("Acc@budget / F1@budget:")
    for pt in summary.get("curve", []):
        print(
            f"  budget={pt['budget']:>3}  acc={pt['accuracy']:.4f}  "
            f"f1={pt['f1']:.4f}  n={pt.get('n_episodes')}"
        )
    return 0


def cmd_version(_: argparse.Namespace) -> int:
    print(__version__)
    return 0


def cmd_decide(_: argparse.Namespace) -> int:
    print(
        "jevforest decide: not implemented. "
        "Use jevforest eval-afa for forest AFA, or JevDecisionsClient for typed Jev questions.",
        file=sys.stderr,
    )
    return 2


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="jevforest",
        description=(
            "jevforest — bagged IG trees that vote on the next budgeted question. "
            "Scoring: jevforest eval-afa --config configs/eval_cube_forest.json"
        ),
    )
    sub = p.add_subparsers(dest="command", required=True)

    e = sub.add_parser("eval-afa", help="AFA hard-budget eval (config-driven)")
    e.add_argument("--config", required=False, help="Eval JSON config")
    e.set_defaults(func=cmd_eval_afa)

    d = sub.add_parser("decide", help="Product decide (not implemented)")
    d.set_defaults(func=cmd_decide)

    ver = sub.add_parser("version", help="Print package version")
    ver.set_defaults(func=cmd_version)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
