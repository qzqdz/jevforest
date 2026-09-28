"""FeatureTable → forest.json (product grow, no LLM)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jevtree.data.feature_table import FeatureTable

from jevforest.core.forest import IGDecisionForestGrower, MaxFeatures


def grow_forest_from_table(
    table: FeatureTable,
    *,
    out: str | Path | None = None,
    n_estimators: int = 16,
    max_features: MaxFeatures = "sqrt",
    max_depth: int | None = 6,
    min_samples: int = 4,
    seed: int = 0,
    criterion: str = "gain",
    bootstrap: bool = True,
    continuous_keys: list[str] | None = None,
) -> dict[str, Any]:
    forest = IGDecisionForestGrower(
        n_estimators=n_estimators,
        max_features=max_features,
        max_depth=max_depth,
        min_samples=min_samples,
        seed=seed,
        criterion=criterion,
        bootstrap=bootstrap,
        continuous_keys=continuous_keys or [],
    )
    forest.fit(table.rows, table.label_key, table.feature_keys, criterion=criterion)
    result: dict[str, Any] = {
        "forest": forest,
        "n_rows": table.n_rows,
        "n_features": table.n_features,
        "n_estimators": len(forest.trees_),
        "label_key": table.label_key,
    }
    if out:
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        forest.save(str(out_path))
        result["out"] = str(out_path)
    return result


__all__ = ["grow_forest_from_table"]
