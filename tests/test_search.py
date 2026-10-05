"""Suche: Reproduzierbarkeit je Verfahren, Zählungen (Evaluationen, besuchte Zellen, Restweg, Anteil nahe der Falle), wer das Ziel sieht, Bezug gleich Stück 2."""

import numpy as np
import pytest

import nobj_constants as C
import nobj_maze as M
import nobj_robot as R
import nobj_search as S


def test_next_generation_keeps_the_elite_unchanged_and_keeps_the_size():
    genomes = R.random_genomes(np.random.default_rng(1), 10)
    score = np.array([0.0, 5.0, 1.0, 9.0, 2.0, 3.0, 4.0, 8.0, 7.0, 6.0])
    nxt = S.next_generation(np.random.default_rng(2), genomes, score)
    assert nxt.shape == genomes.shape and (nxt[0] == genomes[3]).all() and (nxt[1] == genomes[7]).all()


def test_without_mutation_every_child_is_a_copy_of_a_parent():
    genomes = R.random_genomes(np.random.default_rng(1), 8)
    nxt = S.next_generation(np.random.default_rng(2), genomes, np.arange(8, dtype=float), rate=0.0)
    assert all(any((child == parent).all() for parent in genomes) for child in nxt)


def test_the_tournament_prefers_higher_scores():
    genomes = R.random_genomes(np.random.default_rng(5), 60)
    nxt = S.next_generation(np.random.default_rng(6), genomes, np.arange(60, dtype=float), rate=0.0)
    parents = [next(i for i in range(60) if (child == genomes[i]).all()) for child in nxt[C.ELITE:]]
    assert len(parents) == 58 and np.mean(parents) > 36


def test_unknown_selection_is_refused():
    with pytest.raises(ValueError):
        S.run_search(M.make_maze(0), 1, "zufall", gens=2)


@pytest.mark.parametrize("selection", C.SELECTIONS)
def test_runs_are_reproducible_and_depend_on_the_seed(selection):
    grid = M.make_maze(2)
    a, b, c = (S.run_search(grid, s, selection, gens=12, keep_positions=False) for s in (4, 4, 5))
    assert a.best_distance == b.best_distance and a.first_solved == b.first_solved and a.visited_cells == b.visited_cells and a.trap_share == b.trap_share and a.best_distance != c.best_distance


@pytest.mark.parametrize("selection", C.SELECTIONS)
def test_counts_and_series_have_the_right_shape(selection):
    res = S.run_search(M.make_maze(3), 3, selection, gens=25, keep_positions=True)
    assert res.evaluations == 25 * C.POP and len(res.best_distance) == len(res.visited_cells) == len(res.best_progress) == len(res.trap_share) == len(res.end_positions) == 25
    assert all(b >= a for a, b in zip(res.visited_cells, res.visited_cells[1:])) and res.visited_cells[-1] == int(res.visited_grid.sum())
    assert all(b <= a for a, b in zip(res.best_progress, res.best_progress[1:])) and res.last_scores.shape == (C.POP,) and all(0.0 <= x <= 1.0 for x in res.trap_share)


def test_the_baseline_equals_the_second_piece():
    """Reine Novelty = Novelty ohne Archiv auf der Endposition aus Stück 2 und 3: dieselben Evaluationen bis zum Erfolg auf Stufe 3, Seeds 0 bis 5."""
    runs = [S.run_search(M.make_maze(3), s, "novelty", keep_positions=False).first_solved for s in range(6)]
    assert runs == [1246, 1737, 515, 631, 2035, 1641]


def test_every_selection_starts_from_the_same_initial_population_but_runs_differently():
    """Generation 1 ist für alle Verfahren gleich (die Anfangspopulation hängt nur vom Seed ab); schon die zweite Generation unterscheidet Fitness und Novelty."""
    grid = M.make_maze(3)
    firsts = [S.run_search(grid, 2, s, gens=2, keep_positions=True).end_positions for s in C.SELECTIONS]
    assert all((f[0] == firsts[0][0]).all() for f in firsts)
    nov, fit = firsts[C.SELECTIONS.index("novelty")], firsts[C.SELECTIONS.index("fitness")]
    assert (nov[1] != fit[1]).any()


def test_only_the_selections_that_use_the_goal_react_to_a_moved_goal():
    """Mit verschobenem Ziel bleiben die Läufe von Novelty und vom Minimalkriterium über den Start gleich; alle Verfahren mit Fitness oder Zielabstand ändern sich."""
    grid = M.make_maze(3)

    def positions(selection):
        return S.run_search(grid, 2, selection, gens=12, keep_positions=True).end_positions

    base = {s: positions(s) for s in C.SELECTIONS}
    old_goal = C.GOAL
    try:
        C.GOAL = (20.5, 20.5)
        moved = {s: positions(s) for s in C.SELECTIONS}
    finally:
        C.GOAL = old_goal
    for s in C.SELECTIONS:
        same = all((a == b).all() for a, b in zip(base[s], moved[s]))
        assert same == (s in ("novelty", "mc_start3")), s


def test_trap_share_is_zero_without_a_trap_and_follows_the_definition_with_one():
    free = S.run_search(M.make_maze(0), 1, "fitness", gens=15, keep_positions=False)
    assert free.trap_share == [0.0] * 15 and len(S.trap_centers(M.make_maze(0))) == 0
    centers = S.trap_centers(M.make_maze(3))
    assert centers.shape == (1, 2)
    pts = np.array([centers[0], centers[0] + [3.0, 0.0], centers[0] + [3.1, 0.0], centers[0] + [0.0, -2.9]])           # drei von vier Punkten höchstens 3 Zellen entfernt
    assert S.near_trap_share(pts, centers) == pytest.approx(0.75)


def test_the_fitness_search_gets_stuck_in_the_trap_and_novelty_does_not():
    """Auf Stufe 3 sammelt sich die Fitness-Population nahe der Falle (Anteil am Ende deutlich über dem von Novelty)."""
    grid = M.make_maze(3)
    fit = np.mean([S.run_search(grid, s, "fitness", gens=60, keep_positions=False).trap_share[-1] for s in range(3)])
    nov = np.mean([S.run_search(grid, s, "novelty", gens=60, keep_positions=False).trap_share[-1] for s in range(3)])
    assert fit > nov + 0.15


def test_first_solved_counts_evaluations_by_hand_on_the_free_field():
    res = S.run_search(M.make_maze(0), 3, "novelty", gens=30, keep_positions=True)
    for gen, end in enumerate(res.end_positions):
        hit = np.flatnonzero(R.distance_to_goal(end) < C.SUCCESS_RADIUS)
        if len(hit):
            assert res.first_solved == gen * C.POP + int(hit[0]) + 1
            break
    else:
        assert res.first_solved is None


def test_stop_when_solved_ends_the_run_early():
    res = S.run_search(M.make_maze(0), 3, "novelty", gens=300, keep_positions=False, stop_when_solved=True)
    assert res.first_solved is not None and len(res.best_distance) < 300 and len(res.best_distance) == (res.first_solved - 1) // C.POP + 1
