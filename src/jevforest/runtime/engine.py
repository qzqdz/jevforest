"""Local forest execution. Does not call Jev nodes."""

from __future__ import annotations

from typing import Any

from jevforest.core.forest import IGDecisionForestGrower


class ForestRuntimeEngine:
    """Run a fitted / serialized IGDecisionForest on one observation."""

    def run_forest(self, forest: IGDecisionForestGrower, row: dict[str, Any]) -> Any:
        return forest.predict(row)

    def run_forest_traced(
        self, forest: IGDecisionForestGrower, row: dict[str, Any]
    ) -> dict[str, Any]:
        pred, per_tree = forest.predict_path(row)
        return {"prediction": pred, "trees": per_tree}

    def run_payload(self, payload: dict[str, Any], row: dict[str, Any]) -> Any:
        forest = IGDecisionForestGrower.from_json(payload)
        return forest.predict(row)


__all__ = ["ForestRuntimeEngine"]
