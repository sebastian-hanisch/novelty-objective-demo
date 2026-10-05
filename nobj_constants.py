"""Konstanten der Novelty-plus-Ziel-Demo: Labyrinth und Roboter wie in Stück 1 bis 3, zwölf Auswahlverfahren, Studien-Achsen, Regler, Voreinstellungen.
Längen in Zellen (Zellraster N × N, Zelle (i, j) belegt den Bereich [i, i + 1) × [j, j + 1))."""


def fmt_int(n):
    """Ganzzahl mit Leerzeichen als Tausendertrenner (10000 -> 10 000), wie in der README; so lässt sich ein halber Median (1 308.5) nicht mit einem Tausenderpunkt verwechseln."""
    return f"{n:,}".replace(",", " ")


def fmt_med(x):
    """Median der Evaluationen: ganze Zahl mit Leerzeichen als Tausendertrenner, ein halber Wert behält seine Stelle (988.5, 1 308.5); None = kein Lauf war erfolgreich."""
    if x is None:
        return "–"
    return fmt_int(int(x)) if x == int(x) else f"{x:,.1f}".replace(",", " ")


N = 24                                       # Kantenlänge des Labyrinths in Zellen (Rand ist Wand)
START = (3.5, 3.5)                           # Mitte der Startzelle (3, 3)
GOAL = (3.5, 20.5)                           # Mitte der Zielzelle (3, 20)
SUCCESS_RADIUS = 2.0                         # Erfolg: Endposition höchstens so weit vom Ziel
START_HEADING = 0.7853981633974483           # Anfangsrichtung π/4
STEPS = 140                                  # Länge der Steuerfolge
TURN_MAX = 0.6                               # Drehung je Schritt (rad), Gen ∈ [−1, 1]
SPEED_MAX = 0.7                              # Fahrstrecke je Schritt (Zellen), Gen ∈ [0, 1]

POP = 100                                    # Individuen je Generation
ELITE = 2
TOURNAMENT = 3
MUT_SIGMA = 0.25                             # Standardabweichung der Mutation je verändertem Gen
MUT_RATE = 0.15                              # Anteil der Gene, die je Kind mutieren

LEVELS = (0, 1, 2, 3, 4)
LEVEL_LABELS = {0: "0: freies Feld", 1: "1: Öffnung links", 2: "2: Öffnung in der Mitte", 3: "3: Öffnung rechts", 4: "4: Schlangenlinie"}
GAP = 3                                      # Breite einer Öffnung in der Wand (Zellen)

K_DEFAULT = 15                               # Nachbarn der Novelty (k); Verhalten = Endposition, kein Archiv (Stück 2 und 3)
TRAP_RADIUS = 3.0                            # „nahe der Falle“: Endposition höchstens so weit vom Mittelpunkt eines lokalen Minimums der Luftlinie

# Zwölf Auswahlverfahren. Schlüssel -> (Art, Parameter). Fitness = −Abstand zum Ziel; Novelty = mittlerer Abstand zu den k nächsten Endpositionen der Population.
SELECTION_SPECS = {
    "fitness": ("fitness", None),
    "novelty": ("novelty", None),
    "w10": ("weighted", 0.10), "w25": ("weighted", 0.25), "w50": ("weighted", 0.50), "w75": ("weighted", 0.75), "w90": ("weighted", 0.90),
    "pareto": ("pareto", None),
    "nslc": ("nslc", None),
    "mc_start3": ("mc_start", 3.0),          # Minimalkriterium: Endposition mindestens 3 Zellen vom Start
    "mc_goal19": ("mc_goal", 19.0),          # Minimalkriterium: Abstand zum Ziel höchstens 19 Zellen (Start: 17)
    "mc_goal12": ("mc_goal", 12.0),          # Minimalkriterium: Abstand zum Ziel höchstens 12 Zellen
}
SELECTIONS = tuple(SELECTION_SPECS)
BASELINE = "novelty"                         # Bezug im Live-Lauf: reine Novelty (wie Stück 2)
LIVE_SELECTIONS = tuple(s for s in SELECTIONS if s not in ("fitness", "novelty"))
WEIGHT_SELECTIONS = ("novelty", "w10", "w25", "w50", "w75", "w90", "fitness")   # Gewichtssweep: Gewicht w auf der Fitness 0 … 1
WEIGHTS = {"novelty": 0.0, "w10": 0.10, "w25": 0.25, "w50": 0.50, "w75": 0.75, "w90": 0.90, "fitness": 1.0}
SELECTION_LABELS = {
    "fitness": "Fitness (Bezug aus Stück 1)", "novelty": "Novelty allein (Bezug)",
    "w10": "Summe, Fitness-Gewicht 0.1", "w25": "Summe, Fitness-Gewicht 0.25", "w50": "Summe, Fitness-Gewicht 0.5", "w75": "Summe, Fitness-Gewicht 0.75", "w90": "Summe, Fitness-Gewicht 0.9",
    "pareto": "Pareto (Novelty × Fitness)", "nslc": "NSLC (Novelty × lokaler Wettbewerb)",
    "mc_start3": "Minimalkriterium: ≥ 3 Zellen vom Start", "mc_goal19": "Minimalkriterium: ≤ 19 Zellen vom Ziel", "mc_goal12": "Minimalkriterium: ≤ 12 Zellen vom Ziel",
}
GROUPS = ("single", "weighted", "pareto", "criterion")
GROUP_LABELS = {"single": "eine Größe", "weighted": "gewichtete Summe", "pareto": "Pareto-Auswahl", "criterion": "Minimalkriterium"}
SELECTION_GROUP = {"fitness": "single", "novelty": "single", "w10": "weighted", "w25": "weighted", "w50": "weighted", "w75": "weighted", "w90": "weighted",
                   "pareto": "pareto", "nslc": "pareto", "mc_start3": "criterion", "mc_goal19": "criterion", "mc_goal12": "criterion"}

STUDY_SEEDS = 20                             # Läufe je Zelle der Stufenstudie
STUDY_GENS = 400                             # Generationen je Lauf (bei POP = 100: 40 000 Evaluationen)
STUDY_BUDGET = POP * STUDY_GENS
LONG_SEEDS = 100                             # Läufe je Auswahl auf Stufe 4 mit längerer Steuerfolge (bei 20 Läufen verschwinden die Unterschiede im Rauschen)
LONG_STEPS = 300                             # Länge der Steuerfolge im Langlauf-Experiment (Stufe 4)
# Fälle der vorgerechneten Studie: Schlüssel -> (Stufe, Länge der Steuerfolge, Zahl der Seeds)
CASES = {"l3": (3, STEPS, STUDY_SEEDS), "l4": (4, STEPS, STUDY_SEEDS), "l4long": (4, LONG_STEPS, LONG_SEEDS)}
CASE_LABELS = {"l3": "Stufe 3 (140 Schritte)", "l4": "Stufe 4 (140 Schritte)", "l4long": "Stufe 4 (300 Schritte)"}

GENS_MIN, GENS_MAX, GENS_STEP, DEFAULT_GENS = 10, 200, 10, 100      # Regler: Generationen des Live-Laufs
K_MIN, K_MAX, K_STEP = 1, 30, 1
DEFAULT_LEVEL = 3
SEED_MAX = 999999
DEFAULT_SEED = 35
DEFAULT_SELECTION = "w25"

PRESET_ORDER = ("Falle: etwas Ziel dazu", "Zu viel Ziel", "Pareto", "Schlangenlinie")


def _preset(level=DEFAULT_LEVEL, gens=DEFAULT_GENS, selection=DEFAULT_SELECTION, k=K_DEFAULT):
    return {"level": level, "gens": gens, "seed": DEFAULT_SEED, "selection": selection, "k": k}


PRESETS = {
    "Falle: etwas Ziel dazu": _preset(),
    "Zu viel Ziel": _preset(selection="w90", gens=200),
    "Pareto": _preset(selection="pareto", gens=200),
    "Schlangenlinie": _preset(level=4, gens=200, selection="nslc"),
}
# Zahlen aus der vorgerechneten Studie; tests/test_claims.py rechnet jede nach
PRESET_HELP = {
    "Falle: etwas Ziel dazu": "Stufe 3, Summe mit Fitness-Gewicht 0.25: 20 von 20 Läufen, Median 570 Evaluationen gegen 988.5 bei reiner Novelty (der Unterschied ist nicht belegt, p = 0.23); die Fitness allein schafft 11 von 20.",
    "Zu viel Ziel": "Fitness-Gewicht 0.9: nur 12 von 20 Läufen auf Stufe 3; im Mittel stehen 46 % der Population nahe der Falle (reine Novelty: 7 %).",
    "Pareto": "Pareto-Auswahl (Novelty × Fitness) auf Stufe 3: 20 von 20 Läufen, Median 925; nur 8 % der Population stehen nahe der Falle, so wenig wie bei reiner Novelty (7 %).",
    "Schlangenlinie": "Stufe 4 mit NSLC: 0 von 20 Läufen bei 140 Schritten, wie bei jedem der zwölf Verfahren; mit 300 Schritten löst NSLC 6 von 100 Läufen, reine Novelty 11.",
}
