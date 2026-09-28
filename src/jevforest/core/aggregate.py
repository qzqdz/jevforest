"""Forest vote aggregation: plurality, IG-weighted, OOB-weighted."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Mapping, Sequence

VoteRule = str  # "plurality" | "ig_weighted" | "oob"


def disagreement_entropy(votes: Mapping[str, float]) -> float:
    """Shannon entropy (bits) of a vote histogram. Empty → 0."""
    total = sum(float(v) for v in votes.values())
    if total <= 0:
        return 0.0
    h = 0.0
    for c in votes.values():
        if c <= 0:
            continue
        p = c / total
        h -= p * math.log2(p)
    return float(h)


def aggregate_feature_votes(
    proposals: Sequence[str | None],
    *,
    ig_scores: Mapping[str, float] | None = None,
    remaining: Sequence[str] | None = None,
    fallback_ranking: Sequence[str] | None = None,
    vote: VoteRule = "plurality",
    tree_weights: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Aggregate tree proposals into one next-feature choice.

    Abstentions (None) are ignored. If every tree abstains, use
    *fallback_ranking* among *remaining*.

    Vote rules:
      plurality   — most votes; ties by mean IG then name
      ig_weighted — votes × (IG + eps); ties by raw votes then name
      oob         — same as plurality but *tree_weights* scale each ballot
    """
    allowed = set(remaining) if remaining is not None else None
    scores = ig_scores or {}
    weights = list(tree_weights) if tree_weights is not None else [1.0] * len(proposals)
    if len(weights) < len(proposals):
        weights = weights + [1.0] * (len(proposals) - len(weights))

    counts: Counter[str] = Counter()
    ig_sum: Counter[str] = Counter()
    n_abstain = 0
    raw_votes: Counter[str] = Counter()

    for q, w in zip(proposals, weights, strict=False):
        if q is None or (allowed is not None and q not in allowed):
            n_abstain += 1
            continue
        ww = float(w)
        if ww < 0:
            ww = 0.0
        counts[q] += ww
        raw_votes[q] += 1
        ig_sum[q] += ww * float(scores.get(q, 0.0))

    chosen: str | None = None
    if counts:
        rule = (vote or "plurality").lower()
        if rule in ("oob", "plurality"):

            def _key(feat: str) -> tuple:
                n = counts[feat]
                mean_ig = ig_sum[feat] / n if n else 0.0
                return (-n, -mean_ig, feat)

        elif rule in ("ig_weighted", "oob_ig"):

            def _key(feat: str) -> tuple:
                n = counts[feat]
                ig = float(scores.get(feat, 0.0))
                return (-(n * (ig + 1e-12)), -n, feat)

        else:
            raise ValueError(f"unsupported vote rule: {vote!r}")

        chosen = sorted(counts, key=_key)[0]
    elif fallback_ranking:
        pool = list(remaining) if remaining is not None else list(fallback_ranking)
        rank = {fk: i for i, fk in enumerate(fallback_ranking)}
        pool_sorted = sorted(pool, key=lambda f: (rank.get(f, len(rank)), f))
        chosen = pool_sorted[0] if pool_sorted else None

    return {
        "chosen": chosen,
        "votes": {k: float(v) for k, v in counts.items()},
        "raw_votes": dict(raw_votes),
        "n_abstain": n_abstain,
        "disagreement_entropy": disagreement_entropy(counts),
        "vote_rule": (vote or "plurality"),
    }


__all__ = ["aggregate_feature_votes", "disagreement_entropy", "VoteRule"]
