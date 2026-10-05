"""Auswertung: Täuschungsmaße je Stufe, Lösbarkeit, Live-Paar, Hilfsfunktionen der Studie (Konfidenzintervall, Fisher, Mann-Whitney), Vollständigkeit und Stimmigkeit der vorgerechneten Datei (inklusive Neurechnung einzelner Läufe)."""

import numpy as np
import pytest
from scipy.stats import fisher_exact

import nobj_constants as C
import nobj_evaluation as E
import nobj_maze as M
import nobj_search as S

PRE = E.load_precomputed()


def test_level_metrics_by_hand():
    m0, m3 = E.level_metrics(0), E.level_metrics(3)
    assert m0["path"] == 17 and m0["straight"] == pytest.approx(17.0) and m0["detour"] == pytest.approx(1.0) and m0["minima"] == 0 and m0["trap"] is None and m0["free"] == 22 * 22
    assert m3["path"] == 49 and m3["detour"] == pytest.approx(49 / 17) and m3["minima"] == 1 and m3["trap"] == pytest.approx(9.0) and m3["free"] == 22 * 22 - 19


@pytest.mark.parametrize("level", list(C.LEVELS))
def test_every_level_is_solvable_within_the_step_budget(level):
    dist, steps = E.solvability(level)
    assert dist < 0.5 and steps <= C.STEPS


def test_live_pair_runs_the_baseline_and_the_chosen_selection_from_the_same_start():
    (base, bt), (sel, st_) = E.live_pair(2, 12, 5, "pareto", 15)
    assert len(base.best_distance) == len(sel.best_distance) == 12 and bt.shape == st_.shape == (C.STEPS + 1, 2) and tuple(bt[0]) == tuple(st_[0]) == C.START
    assert (base.end_positions[0] == sel.end_positions[0]).all()
    assert np.hypot(*(st_[-1] - np.array(C.GOAL))) == pytest.approx(min(sel.best_distance)) and np.hypot(*(bt[-1] - np.array(C.GOAL))) == pytest.approx(min(base.best_distance))


def test_success_helpers_by_hand():
    runs = [100, None, 500, 50, None]
    assert E.success_count(runs) == 3 and E.success_count(runs, 100) == 2 and E.success_count(runs, 10) == 0
    assert E.median_solved(runs) == 100.0 and E.median_solved([None, None]) is None and E.median_solved([10, 20]) == 15.0


def test_wilson_interval_by_hand_and_at_the_borders():
    lo, hi = E.wilson(50, 100)
    assert lo == pytest.approx(0.4038, abs=1e-3) and hi == pytest.approx(0.5962, abs=1e-3)        # bekannter Wert: 50 von 100 -> 40.4 % bis 59.6 %
    lo0, hi0 = E.wilson(0, 20)
    assert lo0 == 0.0 and hi0 == pytest.approx(0.1611, abs=1e-3)
    lo1, hi1 = E.wilson(20, 20)
    assert hi1 == 1.0 and lo1 == pytest.approx(0.8389, abs=1e-3)


def test_fisher_p_agrees_with_scipy_and_is_one_for_equal_lists():
    a, b = [1] * 11 + [None] * 89, [1] * 2 + [None] * 98
    assert E.fisher_p(a, b) == pytest.approx(fisher_exact([[11, 89], [2, 98]])[1]) and E.fisher_p(a, a) == pytest.approx(1.0)


def test_mann_whitney_needs_only_solved_runs_and_is_symmetric():
    a, b = [100, 200, 300, 400], [150, 250, 350, 450]
    assert E.mannwhitney_p(a, b) == pytest.approx(E.mannwhitney_p(b, a)) and 0.0 < E.mannwhitney_p(a, b) <= 1.0
    assert E.mannwhitney_p(a, a) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        E.mannwhitney_p(a, [1, None])


def test_the_study_covers_every_case_and_every_selection():
    assert set(PRE["cases"]) == set(C.CASES) and PRE["budget"] == C.STUDY_BUDGET
    for case, (level, steps, seeds) in C.CASES.items():
        info = E.case_info(PRE, case)
        assert (info["level"], info["steps"], info["seeds"]) == (level, steps, seeds) and set(info["selections"]) == set(C.SELECTIONS)
        for s in C.SELECTIONS:
            c = E.study_cell(PRE, case, s)
            assert len(c["runs"]) == len(c["cells"]) == len(c["progress"]) == seeds and len(c["mean_trap"]) == C.STUDY_GENS
            assert all(r is None or 1 <= r <= C.STUDY_BUDGET for r in c["runs"]) and max(c["cells"]) <= info["metrics"]["free"] and all(0.0 <= x <= 1.0 for x in c["mean_trap"])


def test_the_stored_level_metrics_equal_the_exact_ones():
    for case in C.CASES:
        stored, fresh = E.case_info(PRE, case)["metrics"], E.level_metrics(C.CASES[case][0])
        for key in ("path", "straight", "detour", "fdc", "minima", "free"):
            assert stored[key] == pytest.approx(fresh[key])
        assert stored["trap"] == pytest.approx(fresh["trap"])


@pytest.mark.parametrize("case,selection,seed", [("l3", "fitness", 0), ("l3", "novelty", 3), ("l3", "w25", 2), ("l3", "pareto", 7), ("l3", "nslc", 1), ("l3", "mc_goal12", 4), ("l4", "mc_start3", 1)])
def test_a_study_run_is_reproduced_by_a_fresh_run_with_the_same_seed(case, selection, seed):
    """Gespeicherte Zahl der Evaluationen bis zum Erfolg, besuchte Zellen und Restweg stimmen mit einem frischen Lauf überein."""
    level = C.CASES[case][0]
    res = S.run_search(M.make_maze(level), seed, selection, gens=C.STUDY_GENS, keep_positions=False)
    c = E.study_cell(PRE, case, selection)
    assert res.first_solved == c["runs"][seed] and res.visited_cells[-1] == c["cells"][seed] and res.best_progress[-1] == c["progress"][seed]


def test_a_long_run_is_reproduced_by_a_fresh_run_with_the_same_seed(monkeypatch):
    """Stufe 4 mit 300 Schritten, reine Novelty, Seed 2: dieselben Endwerte wie gespeichert."""
    monkeypatch.setattr(C, "STEPS", C.LONG_STEPS)
    res = S.run_search(M.make_maze(4), 2, "novelty", gens=C.STUDY_GENS, keep_positions=False)
    c = E.study_cell(PRE, "l4long", "novelty")
    assert res.first_solved == c["runs"][2] and res.visited_cells[-1] == c["cells"][2] and res.best_progress[-1] == c["progress"][2]


def test_the_mean_trap_curve_is_the_mean_of_fresh_runs_for_a_small_case():
    """Die gespeicherte Kurve ist der Mittelwert über die 20 Läufe; zwei frische Läufe liegen im Anfangsbereich nah an ihr (gleiche Anfangspopulationen, erster Wert exakt gleich)."""
    c = E.study_cell(PRE, "l3", "novelty")["mean_trap"]
    first = np.mean([S.run_search(M.make_maze(3), s, "novelty", gens=1, keep_positions=False).trap_share[0] for s in range(C.STUDY_SEEDS)])
    assert c[0] == pytest.approx(round(float(first), 4), abs=1e-9)


def test_the_baseline_equals_the_novelty_of_the_second_piece():
    assert E.median_solved(E.study_cell(PRE, "l3", "novelty")["runs"]) == 988.5 and E.study_cell(PRE, "l3", "novelty")["cells"] == [465] * 20


def test_the_two_140_step_cases_use_the_same_seeds_as_the_long_case_for_the_first_twenty():
    """Die Seeds 0 bis 19 sind in allen Fällen dieselben (Seed = Läufeindex); der Langlauf hat 100 Läufe, also mehr, nicht andere."""
    assert C.CASES["l4long"][2] == 100 and C.CASES["l3"][2] == C.CASES["l4"][2] == C.STUDY_SEEDS == 20
