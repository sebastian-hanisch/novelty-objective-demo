"""JEDE Zahl aus README und App-Texten wird hier nachgerechnet: Täuschungsmaße exakt aus dem Labyrinth, Studienzahlen aus der vorgerechneten Datei (Stufe 3 und 4 mit 140 Schritten: 20 Läufe je Verfahren, Seeds 0 bis 19;
Stufe 4 mit 300 Schritten: 100 Läufe, Seeds 0 bis 99; je 40 000 Evaluationen). Die Datei ist fest; ändern sich die Zahlen nach einer neuen Rechnung, müssen README und Hilfetexte nachgezogen werden. Aufwand in Evaluationen."""

import numpy as np
import pytest

import nobj_constants as C
import nobj_evaluation as E

PRE = E.load_precomputed()
S = C.SELECTIONS


def cell(case, sel):
    return E.study_cell(PRE, case, sel)


def runs(case, sel):
    return cell(case, sel)["runs"]


def succ(case, sel):
    return E.success_count(runs(case, sel))


def mean(case, sel, key):
    return float(np.mean(cell(case, sel)[key]))


def late_trap(case, sel):
    return float(np.mean(cell(case, sel)["mean_trap"][100:]))


def test_exact_deception_measures_of_the_levels_as_in_the_first_pieces():
    """README: kürzester Weg 49 / 77 Zellen (Stufe 3 / 4), Umwegfaktor 2.88 / 4.53, Korrelation 0.69 / 0.68, Fallen 1 / 2, freie Zellen 465 / 427."""
    m3, m4 = E.level_metrics(3), E.level_metrics(4)
    assert (m3["path"], m4["path"]) == (49, 77) and (round(m3["detour"], 2), round(m4["detour"], 2)) == (2.88, 4.53)
    assert (round(m3["fdc"], 2), round(m4["fdc"], 2)) == (0.69, 0.68) and (m3["minima"], m4["minima"]) == (1, 2) and (m3["free"], m4["free"]) == (465, 427)


def test_the_references_equal_the_first_pieces():
    """README: Fitness Stufe 3: 11 von 20, Median 1 267, schlimmster Lauf 3 355, 343.9 Zellen, Restweg 5.55; reine Novelty: 20 von 20, Median 988.5, schlimmster Lauf 3 373, alle 465 Zellen; beide Stufe 4: 0 von 20, Restweg 34.75 / 26.25."""
    assert (succ("l3", "fitness"), E.median_solved(runs("l3", "fitness")), max(r for r in runs("l3", "fitness") if r is not None)) == (11, 1267.0, 3355)
    assert round(mean("l3", "fitness", "cells"), 2) == 343.9 and round(mean("l3", "fitness", "progress"), 2) == 5.55
    assert (succ("l3", "novelty"), E.median_solved(runs("l3", "novelty")), max(runs("l3", "novelty"))) == (20, 988.5, 3373) and cell("l3", "novelty")["cells"] == [465] * 20
    assert succ("l4", "fitness") == succ("l4", "novelty") == 0 and round(mean("l4", "fitness", "progress"), 2) == 34.75 and round(mean("l4", "novelty", "progress"), 2) == 26.25


def test_success_and_median_on_level_three_for_all_twelve_selections():
    """README: Erfolge von 20 / Median auf Stufe 3: Fitness 11 / 1 267, Novelty 20 / 988.5, Summe 0.1 / 0.25 / 0.5: 20 / 709, 20 / 570, 20 / 827, Summe 0.75: 16 / 1 231, Summe 0.9: 12 / 931.5, Pareto 20 / 925, NSLC 20 / 925.5,
    Minimalkriterium Start 3: 20 / 1 064, Ziel 19: 20 / 794.5, Ziel 12: 20 / 1 114."""
    expect = {"fitness": (11, 1267.0), "novelty": (20, 988.5), "w10": (20, 709.0), "w25": (20, 570.0), "w50": (20, 827.0), "w75": (16, 1231.0), "w90": (12, 931.5), "pareto": (20, 925.0),
              "nslc": (20, 925.5), "mc_start3": (20, 1064.0), "mc_goal19": (20, 794.5), "mc_goal12": (20, 1114.0)}
    assert set(expect) == set(S)
    for s, (n, med) in expect.items():
        assert (succ("l3", s), E.median_solved(runs("l3", s))) == (n, med), s


def test_slowest_runs_and_cells_on_level_three():
    """README: schlimmster Lauf auf Stufe 3: Summe 0.1 / 0.25 / 0.5: 1 796 / 1 796 / 2 356, Summe 0.75: 7 977, Summe 0.9: 7 826, Pareto 2 395, NSLC 3 796, Minimalkriterium Start 3: 4 845, Ziel 12: 11 915; besuchte Zellen im Mittel
    Summe 0.75: 398.35, 0.9: 357.1, 0.5: 464.8, Minimalkriterium Ziel 19: 463.65, Ziel 12: 457.55; Restweg Summe 0.75: 3.45, 0.9: 5.5."""
    mx = {s: max(r for r in runs("l3", s) if r is not None) for s in ("w10", "w25", "w50", "w75", "w90", "pareto", "nslc", "mc_start3", "mc_goal12")}
    assert mx == {"w10": 1796, "w25": 1796, "w50": 2356, "w75": 7977, "w90": 7826, "pareto": 2395, "nslc": 3796, "mc_start3": 4845, "mc_goal12": 11915}
    assert [round(mean("l3", s, "cells"), 2) for s in ("w75", "w90", "w50", "mc_goal19", "mc_goal12")] == [398.35, 357.1, 464.8, 463.65, 457.55]
    assert [round(mean("l3", s, "progress"), 2) for s in ("w75", "w90")] == [3.45, 5.5]


def test_the_weight_tips_the_search_back_into_the_trap():
    """README: Fisher-Test gegen reine Novelty auf Stufe 3: Fitness p = 0.001, Summe 0.9 p = 0.003, Summe 0.75 p = 0.106; alle anderen p = 1 (jeweils 20 von 20)."""
    nov = runs("l3", "novelty")
    assert round(E.fisher_p(nov, runs("l3", "fitness")), 3) == 0.001 and round(E.fisher_p(nov, runs("l3", "w90")), 3) == 0.003 and round(E.fisher_p(nov, runs("l3", "w75")), 3) == 0.106
    assert all(E.fisher_p(nov, runs("l3", s)) == pytest.approx(1.0) for s in S if succ("l3", s) == 20)


def test_no_median_difference_to_pure_novelty_is_proven_on_level_three():
    """README: Mann-Whitney-Test der Laufzeiten gegen reine Novelty (alle Verfahren mit 20 von 20): p zwischen 0.23 (Summe 0.25) und 0.89 (Minimalkriterium Start 3); keiner unter 0.05."""
    ps = {s: E.mannwhitney_p(runs("l3", "novelty"), runs("l3", s)) for s in S if s != "novelty" and succ("l3", s) == 20}
    assert len(ps) == 8 and set(ps) == {"w10", "w25", "w50", "pareto", "nslc", "mc_start3", "mc_goal19", "mc_goal12"}
    assert round(min(ps.values()), 2) == 0.23 and min(ps, key=ps.get) == "w25" and round(max(ps.values()), 2) == 0.89 and max(ps, key=ps.get) == "mc_start3" and all(p > 0.05 for p in ps.values())


def test_the_population_near_the_trap_grows_with_the_weight():
    """README: mittlerer Anteil der Population nahe der Falle (Generationen 101 bis 400, Stufe 3): Fitness 52 %, Summe 0.9: 46 %, 0.75: 28 %, 0.5: 10 %, Pareto 8 %, NSLC 8 %, reine Novelty 7 %."""
    pct = {s: round(100 * late_trap("l3", s)) for s in ("fitness", "w90", "w75", "w50", "pareto", "nslc", "novelty")}
    assert pct == {"fitness": 52, "w90": 46, "w75": 28, "w50": 10, "pareto": 8, "nslc": 8, "novelty": 7}
    assert late_trap("l3", "novelty") < late_trap("l3", "w10") < late_trap("l3", "w25") < late_trap("l3", "w50") < late_trap("l3", "w75") < late_trap("l3", "w90") < late_trap("l3", "fitness")


def test_pareto_does_not_hold_the_population_at_the_trap():
    """README: Pareto 8.1 % gegen reine Novelty 7.0 % (Stufe 3, Generationen 101 bis 400): die Vermutung, Pareto halte die Population an der Falle fest, ist nicht bestätigt."""
    assert round(100 * late_trap("l3", "pareto"), 1) == 8.1 and round(100 * late_trap("l3", "novelty"), 1) == 7.0


def test_level_four_with_140_steps_is_solved_by_no_selection():
    """README: Stufe 4, 140 Schritte: 0 von 240 Läufen für die zwölf Verfahren; kleinster wahrer Restweg im Mittel zwischen 24.55 (Minimalkriterium Ziel 12) und 33.9 (Summe 0.9) für die elf Verfahren mit Novelty, 34.75 für die Fitness."""
    assert all(succ("l4", s) == 0 for s in S) and sum(len(runs("l4", s)) for s in S) == 240
    prog = {s: round(mean("l4", s, "progress"), 2) for s in S if s != "fitness"}
    assert (min(prog.values()), min(prog, key=prog.get)) == (24.55, "mc_goal12") and (max(prog.values()), max(prog, key=prog.get)) == (33.9, "w90") and round(mean("l4", "fitness", "progress"), 2) == 34.75


def test_level_four_with_300_steps_in_one_hundred_runs():
    """README: Stufe 4, 300 Schritte, Erfolge von 100: Fitness 0, Novelty 11 (Median 27 378), Summe 0.1 / 0.25 / 0.5 / 0.75 / 0.9: 10 / 15 / 14 / 2 / 1, Pareto 5, NSLC 6, Minimalkriterium Start 3 / Ziel 19 / Ziel 12: 9 / 4 / 3."""
    expect = {"fitness": 0, "novelty": 11, "w10": 10, "w25": 15, "w50": 14, "w75": 2, "w90": 1, "pareto": 5, "nslc": 6, "mc_start3": 9, "mc_goal19": 4, "mc_goal12": 3}
    assert {s: succ("l4long", s) for s in S} == expect and all(len(runs("l4long", s)) == 100 for s in S)
    assert E.median_solved(runs("l4long", "novelty")) == 27378.0


def test_fisher_tests_on_the_long_runs_against_pure_novelty():
    """README: Fisher-Test gegen reine Novelty (11 von 100): Fitness p = 0.001, Summe 0.1: 1.0, 0.25: 0.53, 0.5: 0.67, 0.75: 0.018, 0.9: 0.005, Pareto 0.19, NSLC 0.31, Minimalkriterium Start 3: 0.81, Ziel 19: 0.10, Ziel 12: 0.049."""
    nov = runs("l4long", "novelty")
    p = {s: E.fisher_p(nov, runs("l4long", s)) for s in S if s != "novelty"}
    assert {s: round(v, 3) for s, v in p.items()} == {"fitness": 0.001, "w10": 1.0, "w25": 0.529, "w50": 0.67, "w75": 0.018, "w90": 0.005, "pareto": 0.191, "nslc": 0.311, "mc_start3": 0.814, "mc_goal19": 0.105, "mc_goal12": 0.049}
    assert sorted(s for s in p if p[s] < 0.05) == ["fitness", "mc_goal12", "w75", "w90"] and not any(p[s] < 0.05 and succ("l4long", s) > 11 for s in p)


def test_the_wilson_interval_of_pure_novelty_on_the_long_run():
    """README: 11 von 100: 95-%-Intervall 6.3 % bis 18.6 %; 0 von 100 (Fitness): 0 % bis 3.7 %."""
    lo, hi = E.wilson(11, 100)
    assert (round(100 * lo, 1), round(100 * hi, 1)) == (6.3, 18.6) and tuple(round(100 * x, 1) for x in E.wilson(0, 100)) == (0.0, 3.7)


def test_progress_on_the_long_runs():
    """README: kleinster wahrer Restweg im Mittel (300 Schritte): Fitness 17.4, reine Novelty 7.97, Summe 0.25: 7.34, Summe 0.9: 16.43."""
    assert [round(mean("l4long", s, "progress"), 2) for s in ("fitness", "novelty", "w25", "w90")] == [17.4, 7.97, 7.34, 16.43]


def test_preset_help_quotes_the_study():
    """PRESET_HELP: Summe 0.25 (20 von 20, Median 570 gegen 988.5, p = 0.23; Fitness 11 von 20), Summe 0.9 (12 von 20, 46 % gegen 7 %), Pareto (20 von 20, Median 925, 8 % gegen 7 %), Schlangenlinie (NSLC 0 von 20; 300 Schritte: 6 von 100
    gegen 11)."""
    assert (succ("l3", "w25"), E.median_solved(runs("l3", "w25")), E.median_solved(runs("l3", "novelty")), succ("l3", "fitness")) == (20, 570.0, 988.5, 11)
    assert round(E.mannwhitney_p(runs("l3", "novelty"), runs("l3", "w25")), 2) == 0.23
    assert succ("l3", "w90") == 12 and round(100 * late_trap("l3", "w90")) == 46 and round(100 * late_trap("l3", "novelty")) == 7
    assert succ("l3", "pareto") == 20 and E.median_solved(runs("l3", "pareto")) == 925.0 and round(100 * late_trap("l3", "pareto")) == 8
    assert succ("l4", "nslc") == 0 and succ("l4long", "nslc") == 6 and succ("l4long", "novelty") == 11
    assert C.STUDY_BUDGET == 40_000
    for name, text in C.PRESET_HELP.items():
        assert text and name in C.PRESETS
