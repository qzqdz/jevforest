"""P0: forest grow, T=1 parity, freeze edges, serialization."""

from __future__ import annotations

import json
from pathlib import Path

from jevtree.core.tree import IGDecisionTreeGrower

from jevforest.core.forest import IGDecisionForestGrower


def _xor_rows() -> list[dict]:
    rows = []
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                rows.append({"a": a, "b": b, "c": c, "y": a ^ b})
    return rows * 4


def test_t1_matches_single_tree() -> None:
    rows = _xor_rows()
    fkeys = ["a", "b", "c"]
    tree = IGDecisionTreeGrower(continuous_keys=[], min_samples=1)
    tree.fit(rows, "y", fkeys, max_depth=6)
    forest = IGDecisionForestGrower(
        n_estimators=1,
        max_features="all",
        bootstrap=False,
        min_samples=1,
        max_depth=6,
        seed=0,
    )
    forest.fit(rows, "y", fkeys)
    for row in rows:
        assert forest.predict(row) == tree.predict(row)


def test_serialization_roundtrip(tmp_path: Path) -> None:
    rows = _xor_rows()
    forest = IGDecisionForestGrower(
        n_estimators=4, max_features="sqrt", bootstrap=True, seed=3, min_samples=1
    )
    forest.fit(rows, "y", ["a", "b", "c"])
    path = tmp_path / "forest.json"
    forest.save(str(path))
    loaded = IGDecisionForestGrower.load(str(path))
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["kind"] == "IGDecisionForest"
    for row in rows[:8]:
        assert loaded.predict(row) == forest.predict(row)


def test_same_seed_deterministic() -> None:
    rows = _xor_rows()
    kwargs = dict(
        n_estimators=6, max_features="sqrt", bootstrap=True, seed=11, min_samples=1
    )
    a = IGDecisionForestGrower(**kwargs)
    b = IGDecisionForestGrower(**kwargs)
    a.fit(rows, "y", ["a", "b", "c"])
    b.fit(rows, "y", ["a", "b", "c"])
    assert [tuple(t.feature_subset) for t in a.trees_] == [
        tuple(t.feature_subset) for t in b.trees_
    ]
    assert [a.predict(r) for r in rows] == [b.predict(r) for r in rows]


def test_feature_subsets_are_diverse() -> None:
    rows = _xor_rows()
    forest = IGDecisionForestGrower(
        n_estimators=8, max_features="sqrt", bootstrap=True, seed=0, min_samples=1
    )
    forest.fit(rows, "y", ["a", "b", "c"])
    subsets = {tuple(t.feature_subset) for t in forest.trees_}
    assert forest.k_features_ == 1  # sqrt(3) == 1
    assert len(subsets) > 1


def test_continuous_predict_does_not_collapse_to_q0() -> None:
    rows = []
    for i in range(80):
        rows.append({"x": float(i), "z": i % 3, "y": "lo" if i < 40 else "hi"})
    forest = IGDecisionForestGrower(
        n_estimators=4,
        max_features="all",
        bootstrap=False,
        min_samples=2,
        n_bins=4,
        seed=0,
        continuous_keys=["x"],
    )
    forest.fit(rows, "y", ["x", "z"])
    low = forest._prepared_row({"x": 2.0, "z": 0})["x"]
    high = forest._prepared_row({"x": 78.0, "z": 0})["x"]
    assert str(low).startswith("q")
    assert str(high).startswith("q")
    assert low != high
    # Contrast: jevtree product grower bins a SINGLE row → always q0
    broken = IGDecisionTreeGrower(continuous_keys=["x"], n_bins=4, min_samples=2)
    broken.fit(rows, "y", ["x", "z"])
    # The forest still produces a prediction (not required to match broken grower).
    assert forest.predict({"x": 2.0, "z": 0}) is not None
    assert forest.bin_edges_["x"]
