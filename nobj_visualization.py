"""Plotly-Abbildungen der Novelty-plus-Ziel-Demo: Lauf im Labyrinth (Endpositionen nach Auswahlwert gefärbt, besuchte Zellen), Verlauf von Abstand, besuchten Zellen und Anteil nahe der Falle, Vergleich der zwölf Verfahren,
Gewichts-Sweep, Langlauf mit Konfidenzintervallen, mittlere Falle-Anteile. Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht zoomen. Das Raster `grid[i, j]` wird mit i als x-Achse gezeichnet."""

import numpy as np
import plotly.graph_objects as go

import nobj_constants as C
import nobj_evaluation as E

WALL_COLOR = "#3a3f47"
VISITED_COLOR = "#d6e4f5"
START_COLOR, GOAL_COLOR, PATH_COLOR, TRAP_COLOR = "#54a24b", "#e45756", "#f58518", "#b279a2"
BASE_COLOR, SEL_COLOR = "#4c78a8", "#f58518"
GROUP_COLORS = {"single": "#4c78a8", "weighted": "#f58518", "pareto": "#b279a2", "criterion": "#54a24b"}
FITNESS_COLOR = "#e45756"
CURVE_SELECTIONS = ("fitness", "w90", "w75", "w50", "pareto", "novelty")      # Verfahren der Falle-Anteil-Kurven


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height, legend_y=-0.25, top=10):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=top, b=10), legend=dict(orientation="h", y=legend_y), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def _arena_axes(fig):
    """Beide Achsen 0 bis N mit gleichem Maßstab (quadratische Zellen), Zeichenfläche passt sich der kleineren Seite an (constrain=domain)."""
    fig.update_xaxes(range=[0, C.N], showgrid=False, zeroline=False, title_text="x (Zellen)", constrain="domain")
    fig.update_yaxes(range=[0, C.N], showgrid=False, zeroline=False, title_text="y (Zellen)", scaleanchor="x", scaleratio=1, constrain="domain")


def _markers(fig):
    fig.add_trace(go.Scatter(x=[C.START[0]], y=[C.START[1]], mode="markers", marker=dict(color=START_COLOR, size=13, symbol="square", line=dict(color="white", width=1.5)), name="Start"))
    fig.add_trace(go.Scatter(x=[C.GOAL[0]], y=[C.GOAL[1]], mode="markers", marker=dict(color=GOAL_COLOR, size=16, symbol="star", line=dict(color="white", width=1.5)), name="Ziel"))
    fig.add_shape(type="circle", xref="x", yref="y", x0=C.GOAL[0] - C.SUCCESS_RADIUS, x1=C.GOAL[0] + C.SUCCESS_RADIUS, y0=C.GOAL[1] - C.SUCCESS_RADIUS, y1=C.GOAL[1] + C.SUCCESS_RADIUS,
                  line=dict(color=GOAL_COLOR, dash="dot", width=1.5))


def build_run_figure(grid, end_positions, track, visited, scores):
    """Ein Lauf: Wände (dunkel), alle besuchten Zellen (hellblau), Endpositionen der letzten Generation gefärbt nach ihrem Auswahlwert (hell = bevorzugt) und die beste Fahrt des Laufs (orange)."""
    z = np.where(grid, 2.0, np.where(visited, 1.0, np.nan))
    fig = go.Figure(go.Heatmap(z=z.T, x0=0.5, dx=1, y0=0.5, dy=1, zmin=1, zmax=2, colorscale=[[0, VISITED_COLOR], [0.5, VISITED_COLOR], [0.5, WALL_COLOR], [1, WALL_COLOR]], showscale=False, hoverinfo="skip",
                               xgap=0, ygap=0))
    fig.add_trace(go.Scatter(x=end_positions[:, 0], y=end_positions[:, 1], mode="markers", name="Endpositionen der letzten Generation (Farbe: Auswahlwert)",
                             marker=dict(color=scores, colorscale="Viridis", size=7, line=dict(color="white", width=0.5), showscale=True,
                                         colorbar=dict(title=dict(text="Auswahlwert", side="top"), orientation="h", thickness=10, len=0.9, y=-0.2, yanchor="top"))))
    fig.add_trace(go.Scatter(x=track[:, 0], y=track[:, 1], mode="lines", line=dict(color=PATH_COLOR, width=2.5), name="beste Fahrt"))
    _markers(fig)
    _arena_axes(fig)
    return _base(fig, 480, legend_y=-0.45, top=0)


def _pair_chart(base, sel, selection, y_title, tickformat=None, y_range=None, height=320, trap=None):
    gens = list(range(1, len(base) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=gens, y=base, mode="lines", line=dict(color=BASE_COLOR, width=2.5), name=C.SELECTION_LABELS[C.BASELINE]))
    fig.add_trace(go.Scatter(x=gens, y=sel, mode="lines", line=dict(color=SEL_COLOR, width=2.5), name=C.SELECTION_LABELS[selection]))
    if trap is not None:
        fig.add_hline(y=C.SUCCESS_RADIUS, line=dict(color=GOAL_COLOR, dash="dot"), annotation_text="Zielkreis")
        fig.add_hline(y=trap, line=dict(color=TRAP_COLOR, dash="dash"), annotation_text="tiefste Falle")
    fig.update_xaxes(title_text="Generation")
    fig.update_yaxes(title_text=y_title, rangemode="tozero", tickformat=tickformat, range=y_range)
    return _base(fig, height, legend_y=-0.35)


def build_progress_chart(base_best, sel_best, selection, trap):
    """Kleinster Abstand zum Ziel je Generation, Bezug (reine Novelty) gegen das gewählte Verfahren; gestrichelt: Zielkreis und (wenn vorhanden) die Tiefe der tiefsten Falle."""
    fig = _pair_chart(base_best, sel_best, selection, "bester Abstand zum Ziel (Zellen)")
    fig.add_hline(y=C.SUCCESS_RADIUS, line=dict(color=GOAL_COLOR, dash="dot"), annotation_text="Zielkreis")
    if trap is not None:
        fig.add_hline(y=trap, line=dict(color=TRAP_COLOR, dash="dash"), annotation_text="tiefste Falle")
    return fig


def build_coverage_chart(base_visited, sel_visited, selection, free):
    """Anteil der freien Zellen, die die Suche bis zur jeweiligen Generation besucht hat (Endpositionen)."""
    return _pair_chart([v / free for v in base_visited], [v / free for v in sel_visited], selection, "besuchte Zellen (Anteil der freien)", tickformat=".0%", y_range=[0, 1.03])


def build_trap_chart(base_trap, sel_trap, selection):
    """Anteil der Population, deren Endposition höchstens 3 Zellen von einer Falle liegt, je Generation."""
    return _pair_chart(base_trap, sel_trap, selection, "Population nahe der Falle (Anteil)", tickformat=".0%", y_range=[0, 1.03])


def _label(selection):
    return C.SELECTION_LABELS[selection]


def _color(selection):
    return FITNESS_COLOR if selection == "fitness" else GROUP_COLORS[C.SELECTION_GROUP[selection]]


def build_median_chart(pre, case):
    """Median der Evaluationen bis zum Erfolg je Verfahren (nur über die erfolgreichen Läufe); Beschriftung: erfolgreiche Läufe von N. Farbe nach Gruppe des Verfahrens."""
    n = pre["cases"][case]["seeds"]
    rows = [(s, E.median_solved(E.study_cell(pre, case, s)["runs"]), E.success_count(E.study_cell(pre, case, s)["runs"])) for s in C.SELECTIONS]
    top = max((m or 0) for _, m, _ in rows)
    fig = go.Figure(go.Bar(y=[_label(s) for s, _, _ in rows], x=[m or 0 for _, m, _ in rows], orientation="h", marker_color=[_color(s) for s, _, _ in rows],
                           text=[f"{c} von {n}" + ("" if m is None else f" · Median {C.fmt_med(m)}") for _, m, c in rows], textposition="outside", cliponaxis=False))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title_text="Median der Evaluationen bis zum Erfolg (nur erfolgreiche Läufe)", range=[0, max(top, 1) * 1.55])
    return _base(fig, 450)


def build_weight_chart(pre, case):
    """Gewichts-Sweep: Zahl erfolgreicher Läufe (Balken, links) und Median der Evaluationen bis zum Erfolg (Linie, rechts) über dem Fitness-Gewicht w von 0 (reine Novelty) bis 1 (reine Fitness)."""
    n = pre["cases"][case]["seeds"]
    xs = [C.WEIGHTS[s] for s in C.WEIGHT_SELECTIONS]
    succ = [E.success_count(E.study_cell(pre, case, s)["runs"]) for s in C.WEIGHT_SELECTIONS]
    meds = [E.median_solved(E.study_cell(pre, case, s)["runs"]) for s in C.WEIGHT_SELECTIONS]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[f"{x:g}" for x in xs], y=succ, name=f"erfolgreiche Läufe (von {n})", marker_color=[FITNESS_COLOR if x == 1.0 else "#4c78a8" if x == 0.0 else "#f58518" for x in xs], text=[str(c) for c in succ], textposition="outside", cliponaxis=False))
    fig.add_trace(go.Scatter(x=[f"{x:g}" for x in xs], y=[m for m in meds], name="Median der Evaluationen bis zum Erfolg", mode="lines+markers", line=dict(color="#333", width=2), yaxis="y2"))
    fig.update_layout(yaxis=dict(title="erfolgreiche Läufe", range=[0, n * 1.15]), yaxis2=dict(title="Median der Evaluationen", overlaying="y", side="right", rangemode="tozero"))
    fig.update_xaxes(title_text="Gewicht w der Fitness (0 = reine Novelty, 1 = reine Fitness)", type="category")
    return _base(fig, 360, legend_y=-0.35)


def build_long_chart(pre):
    """Stufe 4 mit 300 Schritten: Erfolgsquote je Verfahren (100 Läufe) mit 95-%-Wilson-Intervall; Beschriftung: erfolgreiche Läufe von 100."""
    n = pre["cases"]["l4long"]["seeds"]
    succ = {s: E.success_count(E.study_cell(pre, "l4long", s)["runs"]) for s in C.SELECTIONS}
    lo_hi = {s: E.wilson(succ[s], n) for s in C.SELECTIONS}
    fig = go.Figure(go.Bar(y=[_label(s) for s in C.SELECTIONS], x=[succ[s] / n for s in C.SELECTIONS], orientation="h", marker_color=[_color(s) for s in C.SELECTIONS],
                           error_x=dict(type="data", symmetric=False, array=[lo_hi[s][1] - succ[s] / n for s in C.SELECTIONS], arrayminus=[succ[s] / n - lo_hi[s][0] for s in C.SELECTIONS], color="#555"),
                           text=[f"{succ[s]} von {n}" for s in C.SELECTIONS], textposition="outside", cliponaxis=False))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title_text="Anteil erfolgreicher Läufe (Balken) mit 95-%-Intervall", tickformat=".0%", range=[0, max(lo_hi[s][1] for s in C.SELECTIONS) * 1.35])
    return _base(fig, 450)


def build_trap_curves(pre, case):
    """Mittlerer Anteil der Population nahe der Falle über den Generationen für ausgewählte Verfahren (Fitness, Summe 0.9 / 0.75 / 0.5, Pareto, Novelty), 20 Läufe je Kurve."""
    fig = go.Figure()
    for s in CURVE_SELECTIONS:
        curve = E.study_cell(pre, case, s)["mean_trap"]
        fig.add_trace(go.Scatter(x=list(range(1, len(curve) + 1)), y=curve, mode="lines", line=dict(color=_color(s), width=2.5, dash="dot" if s in ("w90", "w75") else "solid"), name=_label(s)))
    fig.update_xaxes(title_text="Generation")
    fig.update_yaxes(title_text="Population nahe der Falle (Anteil)", tickformat=".0%", rangemode="tozero")
    return _base(fig, 340, legend_y=-0.35)
