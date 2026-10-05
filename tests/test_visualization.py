"""Abbildungen: gesperrte Achsen, Spuren, Beschriftungen, Farben der Gruppen."""

import numpy as np
import pytest

import nobj_constants as C
import nobj_evaluation as E
import nobj_maze as M
import nobj_visualization as V

PRE = E.load_precomputed()


def _locked(fig):
    return all(ax.fixedrange for ax in fig.select_xaxes()) and all(ax.fixedrange for ax in fig.select_yaxes())


def test_all_charts_lock_their_axes():
    grid = M.make_maze(3)
    (base, bt), (sel, st_) = E.live_pair(3, 10, 1, "pareto", 15)
    figs = [V.build_run_figure(grid, base.end_positions[-1], bt, base.visited_grid, base.last_scores), V.build_run_figure(grid, sel.end_positions[-1], st_, sel.visited_grid, sel.last_scores),
            V.build_progress_chart(base.best_distance, sel.best_distance, "pareto", 9.0), V.build_coverage_chart(base.visited_cells, sel.visited_cells, "pareto", 465),
            V.build_trap_chart(base.trap_share, sel.trap_share, "pareto"), V.build_median_chart(PRE, "l3"), V.build_median_chart(PRE, "l4"), V.build_weight_chart(PRE, "l3"),
            V.build_long_chart(PRE), V.build_trap_curves(PRE, "l3")]
    assert all(_locked(f) for f in figs)


def test_run_figure_colors_the_end_positions_by_their_selection_score():
    grid = M.make_maze(3)
    (base, bt), _ = E.live_pair(3, 10, 1, "w25", 15)
    fig = V.build_run_figure(grid, base.end_positions[-1], bt, base.visited_grid, base.last_scores)
    pts = next(t for t in fig.data if (t.name or "").startswith("Endpositionen"))
    assert list(pts.marker.color) == list(base.last_scores) and len(pts.x) == C.POP and pts.marker.showscale


def test_run_figure_marks_walls_and_visited_cells_with_different_values():
    grid = M.make_maze(3)
    (base, bt), _ = E.live_pair(3, 10, 1, "w25", 15)
    z = np.array(V.build_run_figure(grid, base.end_positions[-1], bt, base.visited_grid, base.last_scores).data[0].z).T
    assert (z[grid] == 2.0).all() and (z[base.visited_grid & ~grid] == 1.0).all() and np.isnan(z[~grid & ~base.visited_grid]).all()


def test_pair_charts_name_the_baseline_and_the_chosen_selection():
    fig = V.build_progress_chart([5.0, 4.0], [6.0, 3.0], "pareto", None)
    assert [t.name for t in fig.data] == [C.SELECTION_LABELS["novelty"], C.SELECTION_LABELS["pareto"]]
    trap = V.build_trap_chart([0.1, 0.2], [0.3, 0.4], "w90")
    assert list(trap.data[1].y) == [0.3, 0.4]


def test_median_chart_has_one_bar_per_selection_in_the_group_colors():
    bars = V.build_median_chart(PRE, "l3").data[0]
    assert len(bars.y) == 12 and bars.marker.color[0] == V.FITNESS_COLOR
    assert [bars.marker.color[i] for i in range(1, 12)] == [V.GROUP_COLORS[C.SELECTION_GROUP[s]] for s in C.SELECTIONS[1:]]
    assert "11 von 20" in bars.text[0] and "20 von 20 · Median 988.5" in bars.text[1]


def test_median_chart_on_level_four_has_no_bar_length():
    bars = V.build_median_chart(PRE, "l4").data[0]
    assert all(x == 0 for x in bars.x) and all(t == "0 von 20" for t in bars.text)


def test_weight_chart_lists_the_seven_weights_with_successes_and_medians():
    fig = V.build_weight_chart(PRE, "l3")
    assert list(fig.data[0].x) == ["0", "0.1", "0.25", "0.5", "0.75", "0.9", "1"] and list(fig.data[0].y) == [20, 20, 20, 20, 16, 12, 11]
    assert fig.data[1].y[0] == 988.5 and fig.data[1].y[-1] == 1267.0 and fig.data[1].yaxis == "y2"


def test_long_chart_has_a_wilson_interval_for_every_selection():
    bars = V.build_long_chart(PRE).data[0]
    assert len(bars.y) == 12 and bars.text[1] == "11 von 100" and bars.text[0] == "0 von 100"
    lo, hi = E.wilson(11, 100)
    assert bars.error_x.array[1] == pytest.approx(hi - 0.11) and bars.error_x.arrayminus[1] == pytest.approx(0.11 - lo)


def test_trap_curves_have_one_line_per_listed_selection():
    fig = V.build_trap_curves(PRE, "l3")
    assert [t.name for t in fig.data] == [C.SELECTION_LABELS[s] for s in V.CURVE_SELECTIONS] and all(len(t.x) == C.STUDY_GENS for t in fig.data)
