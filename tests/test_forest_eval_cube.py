"""Cube hard-budget smoke via jevtree HardBudgetProtocol (no copied metrics)."""

from __future__ import annotations

from jevtree.data.cube import load_cube_split
from jevtree.eval.protocol import HardBudgetEpisodeConfig, HardBudgetProtocol

from jevforest.policy.forest_afa import ForestAcquisitionPolicy


def test_forest_cube_budget3_beats_chance() -> None:
    split = load_cube_split(n_samples=256, n_features=5, seed=0)
    policy = ForestAcquisitionPolicy(
        n_estimators=8,
        max_features="all",
        bootstrap=True,
        seed=0,
        min_samples=1,
        max_depth=6,
        force_acquisition=True,
        predictor_name="match_majority",
    )
    policy.fit(split["train"], split["label_key"], split["feature_keys"])
    cfg = HardBudgetEpisodeConfig(
        dataset_id=split["dataset_id"],
        hard_budget=3,
        split_seed=0,
        budget_schedule=[3],
        dataset_hash=split["dataset_hash"],
        policy_name="ig_forest",
        feature_keys=split["feature_keys"],
        label_key=split["label_key"],
    )
    summary = HardBudgetProtocol().run_eval(
        policy,
        cfg,
        test_rows=split["test"],
        feature_keys=split["feature_keys"],
        label_key=split["label_key"],
        budget_schedule=[3],
    )
    acc = summary["curve"][0]["accuracy"]
    assert acc > 0.6, f"expected acc>0.6 at budget=3, got {acc}"
