"""Bagged IG decision forest. Leaves are jevtree IGDecisionTreeGrower instances.

Continuous features are binned once on the full train set (frozen edges).
Trees are fitted on already-discrete rows so we never hit jevtree's
single-row quantile collapse (everything → q0).
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

from jevtree.core.entropy import best_split
from jevtree.core.tree import (
    IGDecisionTreeGrower,
    TreeLeaf,
    TreeNode,
    apply_quantile_edges,
    bin_rows_with_edges,
    compute_quantile_edges,
)
from jevtree.policy.predictor import detect_continuous_keys

from jevforest.core.aggregate import aggregate_feature_votes

MaxFeatures = int | float | Literal["sqrt", "log2", "all"] | None


def resolve_max_features(max_features: MaxFeatures, n_features: int) -> int:
    """Map sklearn-style max_features to an integer k in [1, p]."""
    p = int(n_features)
    if p <= 0:
        return 0
    if max_features in (None, "all"):
        return p
    if max_features == "sqrt":
        return max(1, int(math.sqrt(p)))
    if max_features == "log2":
        return max(1, int(math.log2(p))) if p > 1 else 1
    if isinstance(max_features, float):
        if max_features >= 1.0:
            return p
        if max_features <= 0.0:
            return 1
        return max(1, min(p, int(max_features * p)))
    if isinstance(max_features, int):
        return max(1, min(p, max_features))
    raise ValueError(f"unsupported max_features: {max_features!r}")


def _bin_row(row: dict[str, Any], bin_edges: dict[str, Sequence[float]]) -> dict[str, Any]:
    out = dict(row)
    for fk, edges in bin_edges.items():
        if fk in out and out[fk] is not None:
            try:
                out[fk] = apply_quantile_edges(float(out[fk]), edges)
            except (TypeError, ValueError):
                pass
    return out


def leaf_counts(node: TreeNode | TreeLeaf, row: dict[str, Any]) -> dict[Any, float]:
    """Walk to a leaf (or node default) and return mass on the raw prediction.

    Keys keep the original label type (TreeLeaf.counts is str-keyed; using it
    would turn 0 into "0" and break T=1 parity with IGDecisionTreeGrower).
    """
    cur: TreeNode | TreeLeaf = node
    while isinstance(cur, TreeNode):
        v = row.get(cur.feature)
        child = cur.children.get(v)
        if child is None:
            return {cur.default: float(cur.n_samples or 1)}
        cur = child
    pred = getattr(cur, "prediction", None)
    mass = float(getattr(cur, "n_samples", 0) or 1)
    return {pred: mass}


def propose_feature(
    node: TreeNode | TreeLeaf,
    row: dict[str, Any],
    unobserved: set[str],
) -> str | None:
    """First unobserved feature on this tree's path; None = abstain (leaf)."""
    cur: TreeNode | TreeLeaf = node
    while isinstance(cur, TreeNode):
        if cur.feature in unobserved:
            return cur.feature
        v = row.get(cur.feature)
        child = cur.children.get(v)
        if child is None:
            return None
        cur = child
    return None


def _oob_accuracy(
    grower: IGDecisionTreeGrower,
    prepared: list[dict[str, Any]],
    boot_idx: Sequence[int],
    label_key: str,
) -> float:
    """Laplace-smoothed accuracy on rows not in the bootstrap sample."""
    seen = set(int(i) for i in boot_idx)
    oob = [i for i in range(len(prepared)) if i not in seen]
    if not oob:
        return 0.5
    correct = 0
    for i in oob:
        pred = grower.predict(prepared[i])
        if pred == prepared[i].get(label_key):
            correct += 1
    return (correct + 1.0) / (len(oob) + 2.0)


@dataclass
class FittedTree:
    """One bagged IG tree plus its acquisition fallback ranking."""

    grower: IGDecisionTreeGrower
    feature_subset: list[str]
    ranking: list[str]
    ig_scores: dict[str, float]
    bootstrap_indices: list[int] = field(default_factory=list)
    oob_weight: float = 1.0

    def tree(self) -> TreeNode | TreeLeaf:
        if self.grower.tree_ is None:
            raise RuntimeError("tree not fitted")
        return self.grower.tree_


class IGDecisionForestGrower:
    """Bootstrap + feature-subspace forest of IGDecisionTreeGrower leaves."""

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
        vote: str = "ig_weighted",
    ) -> None:
        if n_estimators < 1:
            raise ValueError("n_estimators must be >= 1")
        self.n_estimators = int(n_estimators)
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples = int(min_samples)
        self.n_bins = int(n_bins)
        self.seed = int(seed)
        self.criterion = criterion
        self.continuous_keys = list(continuous_keys) if continuous_keys else []
        self.bootstrap = bool(bootstrap)
        self.vote = vote
        self.trees_: list[FittedTree] = []
        self.label_key_: str | None = None
        self.feature_keys_: list[str] = []
        self.bin_edges_: dict[str, list[float]] = {}
        self.global_ranking_: list[str] = []
        self.global_ig_scores_: dict[str, float] = {}
        self.k_features_: int = 0

    def fit(
        self,
        rows: list[dict[str, Any]],
        label_key: str,
        feature_keys: list[str],
        *,
        criterion: str | None = None,
        max_depth: int | None = None,
    ) -> "IGDecisionForestGrower":
        if not rows:
            raise ValueError("rows must be non-empty")
        if not feature_keys:
            raise ValueError("feature_keys must be non-empty")
        crit = criterion or self.criterion
        depth = self.max_depth if max_depth is None else max_depth
        self.label_key_ = label_key
        self.feature_keys_ = list(feature_keys)

        cont = list(self.continuous_keys)
        if not cont:
            cont = detect_continuous_keys(rows, feature_keys, n_bins=self.n_bins)
        self.continuous_keys = cont

        bin_edges: dict[str, list[float]] = {}
        for fk in cont:
            if fk not in feature_keys:
                continue
            bin_edges[fk] = compute_quantile_edges(
                [float(r[fk]) for r in rows if r.get(fk) is not None],
                n_bins=self.n_bins,
            )
        self.bin_edges_ = bin_edges
        prepared = bin_rows_with_edges(rows, bin_edges) if bin_edges else [dict(r) for r in rows]

        global_split = best_split(prepared, label_key, feature_keys, criterion=crit)  # type: ignore[arg-type]
        ranking_rows = global_split.get("ranking") or []
        self.global_ranking_ = [r["feature"] for r in ranking_rows] or list(feature_keys)
        self.global_ig_scores_ = {
            r["feature"]: float(r["score"]) for r in ranking_rows
        }

        p = len(feature_keys)
        k = resolve_max_features(self.max_features, p)
        self.k_features_ = k
        n = len(prepared)
        trees: list[FittedTree] = []
        for t in range(self.n_estimators):
            rng = random.Random(self.seed + t + 1)
            if self.bootstrap:
                boot_idx = [rng.randrange(n) for _ in range(n)]
            else:
                boot_idx = list(range(n))
            boot_rows = [prepared[i] for i in boot_idx]
            if k >= p:
                subset = list(feature_keys)
            else:
                subset = sorted(rng.sample(feature_keys, k))
            grower = IGDecisionTreeGrower(
                min_samples=self.min_samples,
                continuous_keys=[],
                n_bins=self.n_bins,
                bin_seed=self.seed,
            )
            grower.fit(
                boot_rows,
                label_key,
                subset,
                criterion=crit,
                max_depth=depth,
            )
            split = best_split(boot_rows, label_key, subset, criterion=crit)  # type: ignore[arg-type]
            local_rank = [r["feature"] for r in (split.get("ranking") or [])] or list(subset)
            local_ig = {r["feature"]: float(r["score"]) for r in (split.get("ranking") or [])}
            oob_w = _oob_accuracy(grower, prepared, boot_idx, label_key) if self.bootstrap else 1.0
            trees.append(
                FittedTree(
                    grower=grower,
                    feature_subset=subset,
                    ranking=local_rank,
                    ig_scores=local_ig,
                    bootstrap_indices=boot_idx,
                    oob_weight=float(oob_w),
                )
            )
        self.trees_ = trees
        return self

    def _prepared_row(self, row: dict[str, Any]) -> dict[str, Any]:
        if self.bin_edges_:
            return _bin_row(row, self.bin_edges_)
        return dict(row)

    def predict_proba(self, row: dict[str, Any]) -> dict[str, float]:
        if not self.trees_:
            raise RuntimeError("IGDecisionForestGrower.fit must be called first")
        prepared = self._prepared_row(row)
        acc: Counter[str] = Counter()
        for ft in self.trees_:
            mass = leaf_counts(ft.tree(), prepared)
            tot = sum(mass.values()) or 1.0
            for lab, c in mass.items():
                acc[lab] += c / tot
        n = float(len(self.trees_))
        return {k: v / n for k, v in acc.items()}

    def predict(self, row: dict[str, Any]) -> Any:
        proba = self.predict_proba(row)
        if not proba:
            return None
        return sorted(proba.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]

    def predict_path(self, row: dict[str, Any]) -> tuple[Any, list[dict[str, Any]]]:
        """Native forest prediction plus per-tree paths (product audit)."""
        if not self.trees_:
            raise RuntimeError("IGDecisionForestGrower.fit must be called first")
        prepared = self._prepared_row(row)
        per_tree: list[dict[str, Any]] = []
        for i, ft in enumerate(self.trees_):
            pred, steps = ft.tree().predict_path(prepared)
            per_tree.append({"tree": i, "prediction": pred, "path": steps})
        return self.predict(row), per_tree

    def propose_next(
        self,
        row: dict[str, Any],
        unobserved: Sequence[str],
    ) -> dict[str, Any]:
        """Path-conditional plurality vote for the next feature to acquire."""
        if not self.trees_:
            raise RuntimeError("IGDecisionForestGrower.fit must be called first")
        prepared = self._prepared_row(row)
        unseen = set(unobserved)
        proposals: list[str | None] = []
        ig_pool: dict[str, float] = dict(self.global_ig_scores_)
        tree_weights: list[float] = []
        rule = (self.vote or "plurality").lower()
        for ft in self.trees_:
            q = propose_feature(ft.tree(), prepared, unseen)
            proposals.append(q)
            tree_weights.append(float(ft.oob_weight if rule.startswith("oob") else 1.0))
            for feat, score in ft.ig_scores.items():
                ig_pool[feat] = ig_pool.get(feat, 0.0) + score
        n = max(1, len(self.trees_))
        mean_ig = {k: v / n for k, v in ig_pool.items()}
        # Prefer global IG for ranking — subspace IG is biased by max_features.
        ig_for_vote = dict(self.global_ig_scores_) or mean_ig
        result = aggregate_feature_votes(
            proposals,
            ig_scores=ig_for_vote,
            remaining=list(unobserved),
            fallback_ranking=self.global_ranking_,
            vote=rule,
            tree_weights=tree_weights,
        )
        result["proposals"] = proposals
        result["tree_weights"] = tree_weights
        return result

    def to_json(self) -> dict[str, Any]:
        if not self.trees_:
            raise RuntimeError("no forest fitted")
        return {
            "type": "IGDecisionForest",
            "kind": "IGDecisionForest",
            "n_estimators": self.n_estimators,
            "max_features": self.max_features,
            "k_features": self.k_features_,
            "max_depth": self.max_depth,
            "min_samples": self.min_samples,
            "n_bins": self.n_bins,
            "seed": self.seed,
            "criterion": self.criterion,
            "bootstrap": self.bootstrap,
            "vote": self.vote,
            "label_key": self.label_key_,
            "feature_keys": list(self.feature_keys_),
            "continuous_keys": list(self.continuous_keys),
            "bin_edges": {k: list(v) for k, v in self.bin_edges_.items()},
            "global_ranking": list(self.global_ranking_),
            "global_ig_scores": dict(self.global_ig_scores_),
            "trees": [
                {
                    "feature_subset": list(ft.feature_subset),
                    "ranking": list(ft.ranking),
                    "ig_scores": dict(ft.ig_scores),
                    "bootstrap_indices": list(ft.bootstrap_indices),
                    "oob_weight": float(ft.oob_weight),
                    "grower": ft.grower.to_json(),
                }
                for ft in self.trees_
            ],
        }

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(self.to_json(), f, indent=2, ensure_ascii=False)
            f.write("\n")

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "IGDecisionForestGrower":
        if payload.get("type") not in ("IGDecisionForest",) and payload.get("kind") not in (
            "IGDecisionForest",
        ):
            if payload.get("type") != "IGDecisionForest":
                raise ValueError("expected an IGDecisionForest payload")
        obj = cls(
            n_estimators=int(payload.get("n_estimators", 1)),
            max_features=payload.get("max_features", "sqrt"),
            max_depth=payload.get("max_depth"),
            min_samples=int(payload.get("min_samples", 4)),
            n_bins=int(payload.get("n_bins", 4)),
            seed=int(payload.get("seed", 0)),
            criterion=str(payload.get("criterion", "gain")),
            continuous_keys=payload.get("continuous_keys") or [],
            bootstrap=bool(payload.get("bootstrap", True)),
            vote=str(payload.get("vote", "ig_weighted")),
        )
        obj.label_key_ = payload.get("label_key")
        obj.feature_keys_ = list(payload.get("feature_keys") or [])
        raw_edges = payload.get("bin_edges") or {}
        obj.bin_edges_ = {str(k): [float(e) for e in v] for k, v in raw_edges.items()}
        obj.global_ranking_ = list(payload.get("global_ranking") or [])
        obj.global_ig_scores_ = {
            str(k): float(v) for k, v in (payload.get("global_ig_scores") or {}).items()
        }
        obj.k_features_ = int(payload.get("k_features") or 0)
        trees: list[FittedTree] = []
        for td in payload.get("trees") or []:
            grower = IGDecisionTreeGrower.from_json(td["grower"])
            trees.append(
                FittedTree(
                    grower=grower,
                    feature_subset=list(td.get("feature_subset") or []),
                    ranking=list(td.get("ranking") or []),
                    ig_scores={str(k): float(v) for k, v in (td.get("ig_scores") or {}).items()},
                    bootstrap_indices=[int(i) for i in (td.get("bootstrap_indices") or [])],
                    oob_weight=float(td.get("oob_weight", 1.0)),
                )
            )
        obj.trees_ = trees
        obj.n_estimators = len(trees) or obj.n_estimators
        return obj

    @classmethod
    def load(cls, path: str) -> "IGDecisionForestGrower":
        with open(path, encoding="utf-8") as f:
            return cls.from_json(json.load(f))


__all__ = [
    "FittedTree",
    "IGDecisionForestGrower",
    "leaf_counts",
    "propose_feature",
    "resolve_max_features",
    "_oob_accuracy",
    "MaxFeatures",
]
