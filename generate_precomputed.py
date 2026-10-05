"""Rechnet die Studie vor (Build-Zeit, nicht in der App): `python generate_precomputed.py [Prozesse]` schreibt `precomputed_sweep.json`.

  cases   drei Fälle, je zwölf Auswahlverfahren (Fitness, Novelty, gewichtete Summe mit fünf Gewichten, Pareto, NSLC, drei Minimalkriterien), Novelty auf der Endposition ohne Archiv, k = 15, 100 Individuen, 400 Generationen:
          l3      Stufe 3, 140 Schritte, 20 Läufe
          l4      Stufe 4, 140 Schritte, 20 Läufe
          l4long  Stufe 4, 300 Schritte, 100 Läufe (bei 20 Läufen verschwinden die Unterschiede im Rauschen)
          je Zelle: Evaluationen bis zum ersten Erfolg (None = nie), besuchte Zellen und kleinster wahrer Restweg am Ende, mittlerer Verlauf des Anteils der Population nahe einer Falle"""

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import nobj_constants as C
import nobj_evaluation as E
import nobj_maze as M
import nobj_search as S


def _task(args):
    case, selection, seed = args
    level, steps, _ = C.CASES[case]
    old = C.STEPS
    C.STEPS = steps                                      # Genomlänge gilt je Prozessaufruf; danach zurück
    try:
        res = S.run_search(M.make_maze(level), seed, selection, gens=C.STUDY_GENS, keep_positions=False)
    finally:
        C.STEPS = old
    return case, selection, seed, res.first_solved, res.visited_cells[-1], res.best_progress[-1], res.trap_share


def main(workers):
    t0 = time.time()
    tasks = [(case, sel, s) for case, (_, _, seeds) in C.CASES.items() for sel in C.SELECTIONS for s in range(seeds)]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(_task, tasks, chunksize=2))
    cases = {}
    for case, (level, steps, seeds) in C.CASES.items():
        selections = {}
        for sel in C.SELECTIONS:
            rows = sorted([r for r in res if r[0] == case and r[1] == sel], key=lambda r: r[2])
            selections[sel] = {"runs": [r[3] for r in rows], "cells": [r[4] for r in rows], "progress": [r[5] for r in rows],
                               "mean_trap": [round(float(x), 4) for x in np.mean([r[6] for r in rows], axis=0)]}
        cases[case] = {"level": level, "steps": steps, "seeds": seeds, "metrics": E.level_metrics(level), "selections": selections}
    data = {"budget": C.STUDY_BUDGET, "cases": cases}
    E.PRECOMPUTED_PATH.write_text(json.dumps(data), encoding="utf-8")
    print(f"fertig in {time.time() - t0:.0f} s -> {E.PRECOMPUTED_PATH}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else min(14, os.cpu_count() or 1))
