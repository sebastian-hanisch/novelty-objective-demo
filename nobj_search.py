"""Suche mit wählbarem Auswahlverfahren: derselbe Genetische Algorithmus wie in Stück 1 bis 3 (Turnier 3, die zwei Besten bleiben, Mutation ohne Kreuzung); gewählt wird nach dem Wert aus `nobj_selection.selection_score`
(Fitness, Novelty, gewichtete Summe, Pareto, NSLC, Novelty mit Minimalkriterium). Verhalten ist die Endposition, es gibt kein Archiv. Das Ziel geht nur über die Fitness (und die Minimalkriterien) in die Auswahl ein.

Zufall nur über `numpy.random.default_rng(seed)` (PCG64, plattformstabil)."""

from dataclasses import dataclass, field

import numpy as np

import nobj_constants as C
import nobj_maze as M
import nobj_robot as R
import nobj_selection as X


@dataclass
class SearchResult:
    best_distance: list                  # kleinster Abstand zum Ziel je Generation
    visited_cells: list                  # Zahl der bisher besuchten Zellen (Endpositionen) nach jeder Generation
    best_progress: list                  # kleinster wahrer Restweg (Zellen) aller bisher besuchten Zellen nach jeder Generation
    trap_share: list                     # Anteil der Population nahe einer Falle je Generation (0 auf Stufen ohne Falle)
    first_solved: object                 # Zahl der Evaluationen bis zum ersten Individuum im Zielkreis (None: nie)
    end_positions: list = field(default_factory=list)    # Endpositionen (pop, 2) je Generation
    last_scores: object = None           # Auswahlwert jedes Individuums der letzten Generation
    visited_grid: object = None          # bool-Raster der besuchten Zellen
    best_genome: object = None           # Steuerfolge mit dem kleinsten Abstand über den ganzen Lauf
    evaluations: int = 0


def next_generation(rng, genomes, score, sigma=C.MUT_SIGMA, rate=C.MUT_RATE):
    """Neue Population: die ELITE höchsten `score` bleiben unverändert, der Rest sind Mutationen von Turnier-Siegern (Turniergröße TOURNAMENT). Größer ist besser."""
    pop = len(genomes)
    order = np.argsort(-score, kind="stable")
    nxt = [genomes[i].copy() for i in order[:C.ELITE]]
    while len(nxt) < pop:
        cand = rng.integers(0, pop, C.TOURNAMENT)
        parent = cand[np.argmax(score[cand])]
        child = genomes[parent] + rng.normal(0.0, sigma, genomes[parent].shape) * (rng.random(genomes[parent].shape) < rate)
        nxt.append(R.clip_genomes(child))
    return np.array(nxt)


def trap_centers(grid):
    """Mittelpunkte der Fallen (lokale Minima der Luftlinie zum Ziel, ohne das Ziel selbst); leeres Feld (0, 2) auf Stufen ohne Falle."""
    cells = [m[0] for m in M.local_minima(grid)]
    return np.array([[i + 0.5, j + 0.5] for i, j in cells]).reshape(-1, 2)


def near_trap_share(end, centers):
    """Anteil der Endpositionen, die höchstens C.TRAP_RADIUS von einem Fallen-Mittelpunkt entfernt liegen (0 ohne Fallen)."""
    if not len(centers):
        return 0.0
    d = np.linalg.norm(end[:, None, :] - centers[None, :, :], axis=2).min(axis=1)
    return float((d <= C.TRAP_RADIUS).mean())


def run_search(grid, seed, selection=C.BASELINE, gens=C.STUDY_GENS, pop=C.POP, k=C.K_DEFAULT, sigma=C.MUT_SIGMA, rate=C.MUT_RATE, keep_positions=True, stop_when_solved=False):
    """Ein Lauf mit dem Auswahlverfahren `selection` (Schlüssel aus C.SELECTIONS). Zählt Evaluationen (pop je Generation) bis die erste Endposition im Zielkreis liegt, und verfolgt besuchte Zellen, den wahren Restweg
    (kürzester Weg von der besten besuchten Zelle zum Ziel) und den Anteil der Population nahe einer Falle."""
    if selection not in C.SELECTION_SPECS:
        raise ValueError(f"unbekannte Auswahl: {selection}")
    rng = np.random.default_rng(seed)
    genomes = R.random_genomes(rng, pop)
    true_path = M.bfs_distances(grid, M.cell_of(C.GOAL))
    centers = trap_centers(grid)
    visited = np.zeros(grid.shape, bool)
    res = SearchResult(best_distance=[], visited_cells=[], best_progress=[], trap_share=[], first_solved=None)
    best_d, best_p = np.inf, np.inf
    for gen in range(gens):
        end = R.simulate(genomes, grid)
        dist = R.distance_to_goal(end)
        res.evaluations += pop
        if res.first_solved is None and dist.min() < C.SUCCESS_RADIUS:
            res.first_solved = gen * pop + int(np.argmax(dist < C.SUCCESS_RADIUS)) + 1
        idx = np.clip(end.astype(int), 0, grid.shape[0] - 1)
        visited[idx[:, 0], idx[:, 1]] = True
        best_p = min(best_p, float(true_path[idx[:, 0], idx[:, 1]].min()))
        res.best_distance.append(float(dist.min()))
        res.visited_cells.append(int(visited.sum()))
        res.best_progress.append(best_p)
        res.trap_share.append(near_trap_share(end, centers))
        if keep_positions:
            res.end_positions.append(end.copy())
        if dist.min() < best_d:
            best_d = float(dist.min())
            res.best_genome = genomes[int(np.argmin(dist))].copy()
        score = X.selection_score(selection, end, dist, k)
        res.last_scores = score
        if stop_when_solved and res.first_solved is not None:
            break
        genomes = next_generation(rng, genomes, score, sigma, rate)
    res.visited_grid = visited
    return res
