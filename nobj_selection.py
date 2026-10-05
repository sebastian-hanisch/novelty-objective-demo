"""Auswahlwerte: wie aus Fitness und Novelty ein Wert wird, nach dem der Genetische Algorithmus auswählt (Turnier, Elite). Größer ist besser.

* **fitness:** −Abstand der Endposition zum Ziel (Bezug aus Stück 1).
* **novelty:** mittlerer Abstand der Endposition zu den `k` nächsten Endpositionen der Population (Lehman und Stanley 2008, 2011; ohne Archiv, wie in Stück 2 und 3).
* **weighted:** `w · Fitness' + (1 − w) · Novelty'`, beide in jeder Generation auf [0, 1] skaliert (Minimum und Maximum der Population; sind alle Werte gleich, ist die skalierte Größe überall 0).
* **pareto:** zwei Ziele (Novelty, Fitness), beide maximieren; Auswahlwert = −(Front) + halbes normiertes Crowding (NSGA-II, Deb et al. 2002): nicht dominierte zuerst, innerhalb einer Front die weniger gedrängten.
* **nslc:** Novelty mit lokalem Wettbewerb (Lehman und Stanley 2011): zwei Ziele (Novelty, lokaler Wettbewerb), lokaler Wettbewerb = Anteil der `k` nächsten Nachbarn (im Verhaltensraum), die schlechtere Fitness haben; dann Pareto wie oben.
* **mc_start, mc_goal:** Novelty mit Minimalkriterium (Lehman und Stanley 2010): wer das Kriterium (Abstand vom Start mindestens τ beziehungsweise Abstand zum Ziel höchstens τ) nicht erfüllt, bekommt den Wert −1 (unter jeder Novelty) und
  pflanzt sich nur fort, wenn gar kein Individuum das Kriterium erfüllt (dann gilt reine Novelty)."""

import numpy as np

import nobj_constants as C


def novelty_scores(points, k):
    """Novelty jedes Punkts: mittlerer Abstand zu den `k` nächsten anderen Punkten (euklidisch). Der Punkt selbst zählt nicht, ein anderer Punkt am selben Ort zählt mit Abstand 0;
    gibt es weniger als k andere Punkte, gehen alle ein."""
    if len(points) < 2 or k < 1:
        raise ValueError("Novelty braucht mindestens zwei Punkte und k >= 1")
    d = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    m = min(k + 1, d.shape[1])
    d = np.partition(d, m - 1, axis=1)[:, :m]            # die m kleinsten Abstände je Zeile (statt die ganze Zeile zu sortieren; gleiches Ergebnis)
    d.sort(axis=1)
    return d[:, 1:k + 1].mean(axis=1)


def minmax(x):
    """Auf [0, 1] skalieren (Minimum 0, Maximum 1); sind alle Werte gleich, überall 0."""
    lo, hi = float(np.min(x)), float(np.max(x))
    return np.zeros(len(x)) if hi - lo < 1e-12 else (np.asarray(x, float) - lo) / (hi - lo)


def fronts(obj):
    """Nicht-dominierte Sortierung (alle Ziele maximieren): Front je Individuum, 0 = beste Front. i dominiert j, wenn i in keinem Ziel schlechter und in mindestens einem besser ist; gleiche Punkte dominieren einander nicht."""
    obj = np.asarray(obj, float)
    n = len(obj)
    dom = (obj[:, None, :] >= obj[None, :, :]).all(axis=2) & (obj[:, None, :] > obj[None, :, :]).any(axis=2)      # dom[i, j]: i dominiert j
    count = dom.sum(axis=0)                                                                                        # Zahl der Dominierer je Individuum
    rank = np.full(n, -1)
    cur, r = np.flatnonzero(count == 0), 0
    while len(cur):
        rank[cur] = r
        count = count - dom[cur].sum(axis=0)
        count[cur] = -1
        cur = np.flatnonzero(count == 0)
        r += 1
    return rank


def crowding(obj, rank):
    """Crowding je Front (NSGA-II): je Ziel bekommen die beiden Randpunkte der Front 1, die inneren den Abstand ihrer Nachbarn geteilt durch die Spannweite der Front; ein Ziel mit Spannweite 0 trägt nichts bei.
    Fronten mit höchstens zwei Punkten: jeder Punkt bekommt 1 je Ziel."""
    obj = np.asarray(obj, float)
    d = np.zeros(len(obj))
    for r in np.unique(rank):
        idx = np.flatnonzero(rank == r)
        if len(idx) <= 2:
            d[idx] = obj.shape[1]
            continue
        for m in range(obj.shape[1]):
            order = idx[np.argsort(obj[idx, m], kind="stable")]
            span = obj[order[-1], m] - obj[order[0], m]
            if span <= 0:
                continue
            d[order[0]] += 1.0
            d[order[-1]] += 1.0
            d[order[1:-1]] += (obj[order[2:], m] - obj[order[:-2], m]) / span
    return d


def pareto_score(obj):
    """Auswahlwert für die Pareto-Auswahl: −Front + 0.5 · Crowding / größtes Crowding (immer unter 1, ändert also nie die Front-Reihenfolge)."""
    rank = fronts(obj)
    cd = crowding(obj, rank)
    return -rank + 0.5 * cd / max(float(cd.max()), 1e-12)


def local_competition(points, fit, k):
    """Lokaler Wettbewerb: Anteil der `k` nächsten anderen Punkte (euklidisch, bei gleichem Abstand der kleinere Index), deren Fitness kleiner ist als die des Punkts. Gibt es weniger als k andere Punkte, gehen alle ein."""
    points, fit = np.asarray(points, float), np.asarray(fit, float)
    d = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    nn = np.argsort(d, axis=1, kind="stable")[:, :min(k, len(points) - 1)]
    return (fit[nn] < fit[:, None]).mean(axis=1)


def selection_score(selection, end, dist, k):
    """Auswahlwert (größer ist besser) jedes Individuums einer Population für das Verfahren `selection` (Schlüssel aus C.SELECTION_SPECS); `end` Endpositionen (P, 2), `dist` Abstände zum Ziel (P,)."""
    if selection not in C.SELECTION_SPECS:
        raise ValueError(f"unbekannte Auswahl: {selection}")
    kind, par = C.SELECTION_SPECS[selection]
    fit = -np.asarray(dist, float)
    if kind == "fitness":
        return fit
    nov = novelty_scores(end, k)
    if kind == "novelty":
        return nov
    if kind == "weighted":
        return par * minmax(fit) + (1.0 - par) * minmax(nov)
    if kind == "pareto":
        return pareto_score(np.column_stack([nov, fit]))
    if kind == "nslc":
        return pareto_score(np.column_stack([nov, local_competition(end, fit, k)]))
    ok = np.linalg.norm(np.asarray(end, float) - np.array(C.START), axis=1) >= par if kind == "mc_start" else np.asarray(dist, float) <= par
    return np.where(ok, nov, -1.0) if ok.any() else nov
