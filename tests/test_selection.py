"""Auswahlwerte von Hand an Mini-Mengen: Skalierung, Fronten, Crowding, lokaler Wettbewerb, gewichtete Summe, Minimalkriterium, Zuordnung der zwölf Verfahren."""

import numpy as np
import pytest

import nobj_constants as C
import nobj_selection as X

LINE = np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]])


def test_minmax_scales_to_zero_and_one_and_is_zero_for_constants():
    assert X.minmax([2.0, 4.0, 6.0]) == pytest.approx([0.0, 0.5, 1.0]) and X.minmax([5.0, 5.0, 5.0]) == pytest.approx([0.0, 0.0, 0.0])


def test_novelty_by_hand_on_a_line():
    assert X.novelty_scores(np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0], [10.0, 0.0]]), 2) == pytest.approx([2.0, 1.5, 2.5, 8.0])
    assert X.novelty_scores(LINE, 1) == pytest.approx([1.0, 1.0, 2.0])
    with pytest.raises(ValueError):
        X.novelty_scores(LINE[:1], 1)


def test_fronts_by_hand():
    """B (2, 2) dominiert A (1, 1) und E (2, 1); E dominiert A. C (3, 0) und D (0, 3) werden von niemandem dominiert."""
    obj = np.array([[1, 1], [2, 2], [3, 0], [0, 3], [2, 1]], float)       # A, B, C, D, E
    assert list(X.fronts(obj)) == [2, 0, 0, 0, 1]


def test_equal_points_do_not_dominate_each_other():
    assert list(X.fronts(np.array([[1.0, 1.0], [1.0, 1.0], [0.0, 0.0]]))) == [0, 0, 1]


def test_a_single_objective_makes_every_front_a_single_value():
    assert list(X.fronts(np.array([[3.0], [1.0], [2.0], [3.0]]))) == [0, 2, 1, 0]


def test_crowding_by_hand_on_four_points_of_one_front():
    """Vier Punkte auf der Geraden x + y = 3: je Ziel Randpunkte 1, innere (2 / 3); ergibt 2, 4/3, 4/3, 2."""
    obj = np.array([[0.0, 3.0], [1.0, 2.0], [2.0, 1.0], [3.0, 0.0]])
    assert X.crowding(obj, X.fronts(obj)) == pytest.approx([2.0, 4 / 3, 4 / 3, 2.0])


def test_crowding_of_small_fronts_is_one_per_objective():
    obj = np.array([[1.0, 1.0], [0.0, 2.0]])
    assert X.crowding(obj, np.array([0, 0])) == pytest.approx([2.0, 2.0])
    assert X.crowding(np.array([[1.0, 1.0]]), np.array([0])) == pytest.approx([2.0])


def test_crowding_ignores_an_objective_without_spread():
    """Zweites Ziel konstant (Spannweite 0): nur das erste zählt, Randpunkte 1, innere (2 / 3); die Front wird hier von Hand vorgegeben (sonst bildete jeder Punkt eine eigene Front)."""
    obj = np.array([[0.0, 5.0], [1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])
    assert X.crowding(obj, np.zeros(4, int)) == pytest.approx([1.0, 2 / 3, 2 / 3, 1.0])


def test_pareto_score_orders_by_front_first_then_by_crowding():
    """Ränge 0 (B, C, D), 1 (E), 2 (A): der Crowding-Anteil ist höchstens 0.5 und ändert nie die Reihenfolge der Fronten."""
    obj = np.array([[1, 1], [2, 2], [3, 0], [0, 3], [2, 1]], float)
    s = X.pareto_score(obj)
    assert all(0.0 <= s[i] <= 0.5 for i in (1, 2, 3)) and -1.0 <= s[4] <= -0.5 and -2.0 <= s[0] <= -1.5


def test_local_competition_by_hand():
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    fit = np.array([3.0, 1.0, 2.0, 0.0])
    assert X.local_competition(pts, fit, 1) == pytest.approx([1.0, 0.0, 1.0, 0.0])
    assert X.local_competition(pts, fit, 2) == pytest.approx([1.0, 0.0, 1.0, 0.0])
    assert X.local_competition(pts, fit, 30) == pytest.approx([1.0, 1 / 3, 2 / 3, 0.0])            # k größer als die Population: alle anderen Punkte


def test_weighted_sum_by_hand():
    """Abstände 2, 4, 6 (Fitness −2, −4, −6, skaliert 1, 0.5, 0); Novelty 1, 1, 2 (skaliert 0, 0, 1); Gewicht 0.25: 0.25 · (1, 0.5, 0) + 0.75 · (0, 0, 1)."""
    dist = np.array([2.0, 4.0, 6.0])
    assert X.selection_score("w25", LINE, dist, 1) == pytest.approx([0.25, 0.125, 0.75])
    assert X.selection_score("w90", LINE, dist, 1) == pytest.approx(0.9 * np.array([1.0, 0.5, 0.0]) + 0.1 * np.array([0.0, 0.0, 1.0]))


def test_the_extremes_are_the_pure_scores():
    dist = np.array([2.0, 4.0, 6.0])
    assert X.selection_score("fitness", LINE, dist, 1) == pytest.approx([-2.0, -4.0, -6.0])
    assert X.selection_score("novelty", LINE, dist, 1) == pytest.approx([1.0, 1.0, 2.0])


def test_minimal_criterion_by_start_distance_marks_the_failing_ones_and_keeps_novelty_for_the_rest():
    end = np.array([[3.5, 3.5], [3.6, 3.5], [10.0, 3.5], [20.0, 3.5]])          # Abstand vom Start: 0, 0.1, 6.5, 16.5; Kriterium: mindestens 3
    nov = X.novelty_scores(end, 1)
    score = X.selection_score("mc_start3", end, np.full(4, 9.0), 1)
    assert list(score[:2]) == [-1.0, -1.0] and score[2:] == pytest.approx(nov[2:])


def test_minimal_criterion_by_goal_distance():
    end = np.array([[3.5, 18.0], [3.5, 12.0], [3.5, 3.5]])
    dist = np.array([2.5, 8.5, 17.0])                                           # Kriterium mc_goal12: Abstand höchstens 12
    score = X.selection_score("mc_goal12", end, dist, 1)
    nov = X.novelty_scores(end, 1)
    assert score[:2] == pytest.approx(nov[:2]) and score[2] == -1.0
    assert X.selection_score("mc_goal19", end, dist, 1) == pytest.approx(nov)    # 17 <= 19: alle erfüllen es


def test_minimal_criterion_falls_back_to_pure_novelty_if_nobody_passes():
    end = np.array([[3.5, 3.5], [3.6, 3.5], [3.7, 3.5]])
    assert X.selection_score("mc_start3", end, np.full(3, 9.0), 1) == pytest.approx(X.novelty_scores(end, 1))


def test_every_selection_has_a_spec_a_label_and_a_group():
    assert set(C.SELECTION_SPECS) == set(C.SELECTION_LABELS) == set(C.SELECTION_GROUP) and len(C.SELECTIONS) == 12
    assert set(C.SELECTION_GROUP.values()) == set(C.GROUPS) and set(C.LIVE_SELECTIONS) == set(C.SELECTIONS) - {"fitness", "novelty"} and C.BASELINE == "novelty"
    assert [C.WEIGHTS[s] for s in C.WEIGHT_SELECTIONS] == [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    for s in C.SELECTIONS:
        kind, par = C.SELECTION_SPECS[s]
        assert (kind == "weighted") == s.startswith("w") and (par == C.WEIGHTS[s] if kind == "weighted" else True)


def test_unknown_selection_is_refused():
    with pytest.raises(ValueError):
        X.selection_score("zufall", LINE, np.ones(3), 1)


@pytest.mark.parametrize("selection", C.SELECTIONS)
def test_every_selection_returns_one_finite_value_per_individual(selection):
    rng = np.random.default_rng(3)
    end = rng.uniform(1, 23, (40, 2))
    dist = np.hypot(end[:, 0] - C.GOAL[0], end[:, 1] - C.GOAL[1])
    score = X.selection_score(selection, end, dist, 15)
    assert score.shape == (40,) and np.isfinite(score).all()
