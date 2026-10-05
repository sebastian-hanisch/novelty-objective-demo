"""Orakel-Tests für die Auswahlwerte: unabhängige Gegenproben.

* **Fronten** gegen ein Abschälen in zwei Schleifen (jedes Mal die nicht dominierten Punkte entfernen; ein anderer Algorithmus als das Zählen der Dominierer).
* **Crowding** gegen eine skalare Neuberechnung nur aus der Definition.
* **Lokaler Wettbewerb** gegen scikit-learns `NearestNeighbors` (ohne Gleichstände) und eine skalare Neuberechnung (auch mit doppelten Punkten).
* **Novelty** gegen SciPys `cKDTree`.
* **Besuchte Zellen und Anteil nahe der Falle** gegen eine Zählung über Tupel."""

import math

import numpy as np
import pytest
from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors

import nobj_constants as C
import nobj_maze as M
import nobj_search as S
import nobj_selection as X


def dominates(a, b):
    return all(x >= y for x, y in zip(a, b)) and any(x > y for x, y in zip(a, b))


def peel_fronts(obj):
    """Front 0 = alle nicht dominierten Punkte, dann entfernen und wiederholen."""
    remaining = list(range(len(obj)))
    rank = [None] * len(obj)
    r = 0
    while remaining:
        front = [i for i in remaining if not any(dominates(obj[j], obj[i]) for j in remaining if j != i)]
        for i in front:
            rank[i] = r
        remaining = [i for i in remaining if i not in front]
        r += 1
    return np.array(rank)


def scalar_crowding(obj, rank):
    out = [0.0] * len(obj)
    for r in sorted(set(rank)):
        members = [i for i in range(len(obj)) if rank[i] == r]
        if len(members) <= 2:
            for i in members:
                out[i] = float(len(obj[0]))
            continue
        for m in range(len(obj[0])):
            ordered = sorted(members, key=lambda i: (obj[i][m], i))
            span = obj[ordered[-1]][m] - obj[ordered[0]][m]
            if span <= 0:
                continue
            out[ordered[0]] += 1.0
            out[ordered[-1]] += 1.0
            for pos in range(1, len(ordered) - 1):
                out[ordered[pos]] += (obj[ordered[pos + 1]][m] - obj[ordered[pos - 1]][m]) / span
    return np.array(out)


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("m", [2, 3])
def test_fronts_agree_with_peeling(seed, m):
    obj = np.random.default_rng(seed * 10 + m).uniform(0, 1, (35, m))
    assert list(X.fronts(obj)) == list(peel_fronts(obj))


def test_fronts_agree_with_peeling_for_points_on_a_grid_with_many_ties():
    obj = np.random.default_rng(1).integers(0, 4, (40, 2)).astype(float)
    assert list(X.fronts(obj)) == list(peel_fronts(obj))


@pytest.mark.parametrize("seed", range(6))
def test_crowding_agrees_with_the_scalar_definition(seed):
    obj = np.random.default_rng(seed).uniform(0, 1, (35, 2))
    rank = X.fronts(obj)
    assert X.crowding(obj, rank) == pytest.approx(scalar_crowding(obj, list(rank)), abs=1e-12)


def scalar_local_competition(points, fit, k):
    out = []
    for i, p in enumerate(points):
        order = sorted((math.dist(p, q), j) for j, q in enumerate(points) if j != i)[:k]
        out.append(sum(1 for _, j in order if fit[j] < fit[i]) / len(order))
    return np.array(out)


def test_local_competition_agrees_with_scikit_learn_without_ties():
    rng = np.random.default_rng(4)
    pts, fit = rng.uniform(0, 24, (50, 2)), rng.uniform(-20, 0, 50)
    for k in (1, 5, 15):
        _, nn = NearestNeighbors(n_neighbors=k + 1).fit(pts).kneighbors(pts)
        ref = (fit[nn[:, 1:]] < fit[:, None]).mean(axis=1)
        assert X.local_competition(pts, fit, k) == pytest.approx(ref, abs=1e-12)


def test_local_competition_agrees_with_the_scalar_definition_with_duplicates():
    rng = np.random.default_rng(8)
    pts = rng.uniform(0, 24, (8, 2))[rng.integers(0, 8, 40)]
    fit = rng.uniform(-20, 0, 40)
    for k in (1, 5, 15):
        assert X.local_competition(pts, fit, k) == pytest.approx(scalar_local_competition(pts, fit, k), abs=1e-12)


@pytest.mark.parametrize("k", [1, 3, 15, 30])
def test_novelty_agrees_with_a_kd_tree(k):
    pts = np.random.default_rng(k).uniform(0, 24, (40, 2))
    dist, _ = cKDTree(pts).query(pts, k=min(k + 1, len(pts)))
    assert X.novelty_scores(pts, k) == pytest.approx(dist[:, 1:].mean(axis=1), abs=1e-12)


def test_visited_cells_and_trap_share_agree_with_a_set_and_a_scalar_count():
    grid = M.make_maze(4)
    res = S.run_search(grid, 5, "pareto", gens=30, keep_positions=True)
    seen, counts = set(), []
    for end in res.end_positions:
        seen.update((int(x), int(y)) for x, y in end)
        counts.append(len(seen))
    assert res.visited_cells == counts and res.visited_cells[-1] == int(res.visited_grid.sum())
    centers = [(i + 0.5, j + 0.5) for (i, j), _ in M.local_minima(grid)]
    expected = [sum(1 for x, y in end if any(math.hypot(x - cx, y - cy) <= C.TRAP_RADIUS for cx, cy in centers)) / len(end) for end in res.end_positions]
    assert res.trap_share == pytest.approx(expected, abs=1e-12) and len(centers) == 2
