"""Path-conditional forest acquisition policy (AFA act / predict)."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Sequence

from jevtree.policy.ig_acquisition import IGAcquisitionPolicy
from jevtree.policy.predictor import (
    build_predictor,
    configure_policy_predictor,
    detect_continuous_keys,
    normalize_predictor_name,
)

from jevforest.core.forest import IGDecisionForestGrower, MaxFeatures


class ForestAcquisitionPolicy(IGAcquisitionPolicy):
    """AFA policy: each tree walks to the first unobserved node and votes.

    Fair-curve ``predict()`` delegates to the shared jevtree predictor
    (default ``match_majority``; MiniBooNE comparisons use ``logistic_impute``).
    """

    force_acquisition: bool = False
    has_builtin_classifier: bool = True

    def __init__(
        self,
        *,
        n_estimators: int = 16,
        max_features: MaxFeatures = "sqrt",
        max_depth: int | None = 6,
        min_samples: int = 4,
        n_bins: int = 4,
        seed: int = 0,
        criterion: str = "gain",
        continuous_keys: Sequence[str] | None = None,
        bootstrap: bool = True,
        force_acquisition: bool = False,
        predictor_name: str = "match_majority",
        bin_seed: int = 0,  # config symmetry with jevtree policies; edges are frozen on train
        vote: str = "ig_weighted",
    ) -> None:
        self.n_estimators = int(n_estimators)
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples = int(min_samples)
        self.n_bins = int(n_bins)
        self.seed = int(seed)
        self.criterion = criterion
        self.continuous_keys = list(continuous_keys) if continuous_keys else []
        self.bootstrap = bool(bootstrap)
        self.force_acquisition = bool(force_acquisition)
        self.predictor_name = normalize_predictor_name(predictor_name)
        self.bin_seed = int(bin_seed)
        self.vote = str(vote or "plurality")
        self.forest = IGDecisionForestGrower(
            n_estimators=self.n_estimators,
            max_features=self.max_features,
            max_depth=self.max_depth,
            min_samples=self.min_samples,
            n_bins=self.n_bins,
            seed=self.seed,
            criterion=self.criterion,
            continuous_keys=self.continuous_keys,
            bootstrap=self.bootstrap,
            vote=self.vote,
        )
        self.feature_keys_: list[str] = []
        self.label_key_: str | None = None
        self.ranking_: list[str] = []
        self.raw_train_rows_: list[dict[str, Any]] = []
        self._predictor = build_predictor(self.predictor_name)
        self.last_vote_: dict[str, Any] | None = None

    def fit(self, rows: list[dict[str, Any]], label_key: str, feature_keys: list[str]) -> None:
        self.feature_keys_ = list(feature_keys)
        self.label_key_ = label_key
        self.raw_train_rows_ = list(rows)
        cont = list(self.continuous_keys)
        if not cont:
            cont = detect_continuous_keys(rows, feature_keys, n_bins=self.n_bins)
        self.continuous_keys = cont
        self.forest.continuous_keys = cont
        self.forest.fit(rows, label_key, feature_keys, criterion=self.criterion)
        self.ranking_ = list(self.forest.global_ranking_)
        self._predictor = build_predictor(self.predictor_name)
        configure_policy_predictor(
            self._predictor,
            feature_keys=self.feature_keys_,
            label_key=label_key,
            binned_rows=self._binned_train(),
            raw_rows=self.raw_train_rows_,
            bin_edges=self.forest.bin_edges_,
            ranking=self.ranking_,
        )

    def _binned_train(self) -> list[dict[str, Any]]:
        from jevtree.core.tree import bin_rows_with_edges

        edges = self.forest.bin_edges_
        if not edges:
            return [dict(r) for r in self.raw_train_rows_]
        return bin_rows_with_edges(self.raw_train_rows_, edges)

    def _row_from_masked(
        self,
        masked_features: Sequence[Any],
        feature_mask: Sequence[bool],
    ) -> dict[str, Any]:
        row: dict[str, Any] = {}
        for i, fk in enumerate(self.feature_keys_):
            if i < len(feature_mask) and feature_mask[i] and i < len(masked_features):
                val = masked_features[i]
                if val is None:
                    continue
                row[fk] = val
        return row

    def _remaining_names(
        self,
        feature_mask: Sequence[bool],
        selection_mask: Sequence[bool] | None = None,
    ) -> list[str]:
        names: list[str] = []
        for i, fk in enumerate(self.feature_keys_):
            if i < len(feature_mask) and feature_mask[i]:
                continue
            if selection_mask is not None and i < len(selection_mask) and not selection_mask[i]:
                continue
            names.append(fk)
        return names

    def act(
        self,
        masked_features: Sequence[Any],
        feature_mask: Sequence[bool],
        selection_mask: Sequence[bool] | None = None,
        label: Any = None,  # noqa: ARG002
        feature_shape: Any = None,  # noqa: ARG002
    ) -> int:
        if not self.forest.trees_:
            raise RuntimeError("ForestAcquisitionPolicy.fit must be called first")
        remaining = self._remaining_names(feature_mask, selection_mask)
        if not remaining:
            return 0
        row = self._row_from_masked(masked_features, feature_mask)
        vote = self.forest.propose_next(row, remaining)
        self.last_vote_ = vote
        chosen = vote.get("chosen")
        if chosen is None:
            if self.force_acquisition:
                chosen = remaining[0]
            else:
                return 0
        if chosen not in remaining:
            if self.force_acquisition:
                return self.feature_keys_.index(remaining[0]) + 1
            return 0
        return self.feature_keys_.index(chosen) + 1

    def predict(
        self,
        masked_features: Sequence[Any],
        feature_mask: Sequence[bool],
        label: Any = None,
        feature_shape: Any = None,
    ) -> Any:
        if self.label_key_ is None:
            raise RuntimeError("ForestAcquisitionPolicy.fit must be called first")
        return self._predictor.predict(
            masked_features, feature_mask, label=label, feature_shape=feature_shape
        )

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.suffix.lower() == ".json":
            payload = {
                "type": "ForestAcquisitionPolicy",
                "n_estimators": self.n_estimators,
                "max_features": self.max_features,
                "max_depth": self.max_depth,
                "min_samples": self.min_samples,
                "n_bins": self.n_bins,
                "seed": self.seed,
                "criterion": self.criterion,
                "continuous_keys": self.continuous_keys,
                "bootstrap": self.bootstrap,
                "force_acquisition": self.force_acquisition,
                "predictor_name": self.predictor_name,
                "vote": self.vote,
                "feature_keys": self.feature_keys_,
                "label_key": self.label_key_,
                "ranking": self.ranking_,
                "forest": self.forest.to_json() if self.forest.trees_ else None,
            }
            with open(p, "w", encoding="utf-8", newline="\n") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
                f.write("\n")
        else:
            with open(p, "wb") as f:
                pickle.dump(self, f)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "ForestAcquisitionPolicy":  # noqa: ARG002
        p = Path(path)
        if p.suffix.lower() == ".json":
            with open(p, encoding="utf-8") as f:
                payload = json.load(f)
            obj = cls(
                n_estimators=int(payload.get("n_estimators", 16)),
                max_features=payload.get("max_features", "sqrt"),
                max_depth=payload.get("max_depth"),
                min_samples=int(payload.get("min_samples", 4)),
                n_bins=int(payload.get("n_bins", 4)),
                seed=int(payload.get("seed", 0)),
                criterion=str(payload.get("criterion", "gain")),
                continuous_keys=payload.get("continuous_keys") or [],
                bootstrap=bool(payload.get("bootstrap", True)),
                force_acquisition=bool(payload.get("force_acquisition", False)),
                predictor_name=str(payload.get("predictor_name", "match_majority")),
                vote=str(payload.get("vote", "ig_weighted")),
            )
            obj.feature_keys_ = list(payload.get("feature_keys") or [])
            obj.label_key_ = payload.get("label_key")
            obj.ranking_ = list(payload.get("ranking") or [])
            forest_payload = payload.get("forest")
            if forest_payload:
                obj.forest = IGDecisionForestGrower.from_json(forest_payload)
                obj.continuous_keys = list(obj.forest.continuous_keys)
            if obj.label_key_ is not None and obj.feature_keys_:
                obj._predictor = build_predictor(obj.predictor_name)
                # predictor needs train rows; JSON save omitted raw rows to stay small.
                # Reconfigure is best-effort if forest has no raw rows.
            return obj
        with open(p, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, ForestAcquisitionPolicy):
            raise TypeError(f"expected ForestAcquisitionPolicy, got {type(obj)}")
        return obj


__all__ = ["ForestAcquisitionPolicy"]
