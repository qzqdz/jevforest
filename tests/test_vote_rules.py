"""Vote aggregation rules for forest acquisition."""

from __future__ import annotations

from jevforest.core.aggregate import aggregate_feature_votes
from jevforest.core.forest import IGDecisionForestGrower


def test_ig_weighted_prefers_high_ig_over_raw_count() -> None:
    proposals = ["weak", "weak", "strong"]
    ig = {"weak": 0.01, "strong": 0.9}
    plurality = aggregate_feature_votes(proposals, ig_scores=ig, remaining=["weak", "strong"])
    weighted = aggregate_feature_votes(
        proposals, ig_scores=ig, remaining=["weak", "strong"], vote="ig_weighted"
    )
    assert plurality["chosen"] == "weak"
    assert weighted["chosen"] == "strong"


def test_oob_weights_break_plurality() -> None:
    proposals = ["a", "a", "b"]
    ig = {"a": 0.2, "b": 0.2}
    out = aggregate_feature_votes(
        proposals,
        ig_scores=ig,
        remaining=["a", "b"],
        vote="oob",
        tree_weights=[0.1, 0.1, 1.0],
    )
    assert out["chosen"] == "b"


def test_oob_weights_stored_on_fit() -> None:
    rows = []
    for a in (0, 1):
        for b in (0, 1):
            for _ in range(8):
                rows.append({"a": a, "b": b, "c": 0, "y": a ^ b})
    forest = IGDecisionForestGrower(
        n_estimators=4, max_features="sqrt", bootstrap=True, seed=0, min_samples=1
    )
    forest.fit(rows, "y", ["a", "b", "c"])
    assert all(0.0 < ft.oob_weight <= 1.0 for ft in forest.trees_)
