"""P1: path-walk plurality act()."""

from __future__ import annotations

from jevtree.core.tree import TreeNode

from jevforest.core.forest import propose_feature
from jevforest.policy.forest_afa import ForestAcquisitionPolicy


def _cubeish() -> tuple[list[dict], list[str]]:
    rows = []
    for f0 in (0, 1):
        for f1 in (0, 1):
            for f2 in (0, 1):
                for f3 in (0, 1):
                    y = int((f0 ^ f1) or (f0 & f2))
                    rows.append({"f0": f0, "f1": f1, "f2": f2, "f3": f3, "y": y})
    return rows * 4, ["f0", "f1", "f2", "f3"]


def test_t1_act_matches_root_then_path() -> None:
    rows, fkeys = _cubeish()
    policy = ForestAcquisitionPolicy(
        n_estimators=1,
        max_features="all",
        bootstrap=False,
        min_samples=1,
        max_depth=6,
        seed=0,
        force_acquisition=True,
    )
    policy.fit(rows, "y", fkeys)
    tree = policy.forest.trees_[0].tree()
    assert isinstance(tree, TreeNode)
    mask = [False] * len(fkeys)
    masked = [None] * len(fkeys)
    action = policy.act(masked, mask)
    assert action == fkeys.index(tree.feature) + 1

    # reveal root and ask again — proposal must be the next unobserved node
    idx = action - 1
    mask[idx] = True
    masked[idx] = 0
    row = {fkeys[i]: masked[i] for i in range(len(fkeys)) if mask[i]}
    expected = propose_feature(tree, row, {fkeys[i] for i, s in enumerate(mask) if not s})
    action2 = policy.act(masked, mask)
    if expected is None:
        assert action2 != 0  # force_acquisition fallback
    else:
        assert fkeys[action2 - 1] == expected


def test_force_acquisition_never_returns_zero() -> None:
    rows, fkeys = _cubeish()
    policy = ForestAcquisitionPolicy(
        n_estimators=4,
        max_features="sqrt",
        bootstrap=True,
        seed=1,
        force_acquisition=True,
        min_samples=1,
    )
    policy.fit(rows, "y", fkeys)
    mask = [False] * len(fkeys)
    masked: list = [None] * len(fkeys)
    acquired = []
    for _ in range(len(fkeys)):
        action = policy.act(masked, mask)
        assert action != 0
        i = action - 1
        assert not mask[i]
        mask[i] = True
        masked[i] = rows[0][fkeys[i]]
        acquired.append(i)
    assert policy.act(masked, mask) == 0
    assert len(acquired) == len(fkeys)


def test_plurality_is_deterministic() -> None:
    rows, fkeys = _cubeish()
    kwargs = dict(
        n_estimators=8,
        max_features="sqrt",
        bootstrap=True,
        seed=2,
        force_acquisition=True,
        min_samples=1,
    )
    a = ForestAcquisitionPolicy(**kwargs)
    b = ForestAcquisitionPolicy(**kwargs)
    a.fit(rows, "y", fkeys)
    b.fit(rows, "y", fkeys)
    mask = [False] * len(fkeys)
    masked = [None] * len(fkeys)
    assert a.act(masked, mask) == b.act(masked, mask)
