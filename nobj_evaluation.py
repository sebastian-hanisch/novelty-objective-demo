"""Auswertung: Täuschungsmaße je Stufe (exakt), Lösbarkeitsnachweis, Live-Paar (Bezug neben gewähltem Verfahren), Lesen und Aggregieren der vorgerechneten Studie, Konfidenzintervalle und Fisher-Test.
Die Studie (zwölf Verfahren auf Stufe 3 und 4, Stufe 4 mit 300 Schritten und 100 Läufen) wird beim Bau gerechnet (`generate_precomputed.py`) und hier nur gelesen."""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.stats import fisher_exact, mannwhitneyu

import nobj_constants as C
import nobj_maze as M
import nobj_robot as R
import nobj_search as S

PRECOMPUTED_PATH = Path(__file__).resolve().parent / "precomputed_sweep.json"


def level_metrics(level):
    """Exakte Täuschungsmaße der Stufe: kürzester Weg (Zellen), Luftlinie Start–Ziel, Umwegfaktor, Fitness-Distanz-Korrelation, lokale Minima (Zahl), Abstand des tiefsten Minimums zum Ziel, freie Zellen."""
    g = M.make_maze(level)
    path = M.shortest_path(g)
    return {"path": len(path) - 1, "straight": float(np.hypot(C.START[0] - C.GOAL[0], C.START[1] - C.GOAL[1])), "detour": M.detour_ratio(g), "fdc": M.fitness_distance_correlation(g),
            "minima": len(M.local_minima(g)), "trap": M.trap_distance(g), "free": int((~g).sum())}


def solvability(level):
    """Lösbarkeitsnachweis: eine aus dem kürzesten Pfad abgeleitete Steuerfolge fährt in diesem Labyrinth ins Ziel. Gibt (Abstand am Ende, benötigte Schritte) zurück."""
    g = M.make_maze(level)
    genome = R.controller_from_path(M.shortest_path(g))
    end = R.simulate(genome[None], g)
    return float(R.distance_to_goal(end)[0]), int((np.abs(genome).sum(axis=1) > 0).sum())


def live_run(level, gens, seed, selection, k):
    """Ein Lauf mit dem Auswahlverfahren `selection` auf der gewählten Stufe, samt bester Fahrt (Positionen je Schritt)."""
    g = M.make_maze(level)
    res = S.run_search(g, seed, selection, gens=gens, k=k)
    _, track = R.simulate(res.best_genome[None], g, trajectories=True)
    return res, track[0]


def live_pair(level, gens, seed, selection, k):
    """Derselbe Seed, dieselbe Stufe: der Lauf mit reiner Novelty (Bezug) und der Lauf mit dem gewählten Verfahren. Gibt ((Lauf, Fahrt) des Bezugs, (Lauf, Fahrt) des Verfahrens) zurück."""
    return live_run(level, gens, seed, C.BASELINE, k), live_run(level, gens, seed, selection, k)


@lru_cache(maxsize=1)
def load_precomputed():
    return json.loads(PRECOMPUTED_PATH.read_text(encoding="utf-8"))


def success_count(runs, budget=None):
    """Zahl der Läufe, die den Zielkreis erreichten (höchstens `budget` Evaluationen, wenn angegeben)."""
    return sum(1 for r in runs if r is not None and (budget is None or r <= budget))


def median_solved(runs):
    """Median der Evaluationen bis zum Erfolg über die erfolgreichen Läufe (None, wenn kein Lauf erfolgreich war)."""
    ok = [r for r in runs if r is not None]
    return float(np.median(ok)) if ok else None


def wilson(successes, n, z=1.96):
    """Wilson-Intervall (95 %) für einen Anteil: (untere, obere Grenze). Auch bei 0 oder n Erfolgen brauchbar."""
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def fisher_p(runs_a, runs_b):
    """Zweiseitiger exakter Test nach Fisher: unterscheiden sich die Erfolgsquoten zweier Läufe-Listen?"""
    a, b = success_count(runs_a), success_count(runs_b)
    return float(fisher_exact([[a, len(runs_a) - a], [b, len(runs_b) - b]])[1])


def mannwhitney_p(runs_a, runs_b):
    """Zweiseitiger Mann-Whitney-Test auf die Evaluationen bis zum Erfolg zweier Läufe-Listen; nur sinnvoll, wenn in beiden Listen jeder Lauf erfolgreich war (sonst ValueError)."""
    if any(r is None for r in list(runs_a) + list(runs_b)):
        raise ValueError("Mann-Whitney braucht in beiden Listen lauter erfolgreiche Läufe")
    return float(mannwhitneyu(runs_a, runs_b, alternative="two-sided").pvalue)


def case_info(pre, case):
    return pre["cases"][case]


def study_cell(pre, case, selection):
    return pre["cases"][case]["selections"][selection]
