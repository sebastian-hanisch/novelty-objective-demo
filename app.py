"""Novelty plus Ziel – wie viel Ziel verträgt die Suche? - Stück 4 der Konzepte-Linie "Novelty Search und Quality-Diversity"
Sebastian Hanisch - Operations Research und Machine Learning

Derselbe Hof-Roboter, dasselbe täuschende Labyrinth und dieselbe Novelty wie in Stück 2 und 3 - aber die Auswahl nutzt wieder das Ziel: gewichtete Summe, Pareto-Auswahl, Novelty mit lokalem Wettbewerb (NSLC) und Novelty mit
Minimalkriterium. Siehe README für die Einordnung in die Linie.

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import nobj_constants as C
import nobj_evaluation as E
import nobj_maze as M
from nobj_presets import (apply_preset, bounds, init_session_state_defaults, load_permalink_settings, randomize_seed, sync_query_params)
from nobj_visualization import (build_coverage_chart, build_long_chart, build_median_chart, build_progress_chart, build_run_figure, build_trap_chart, build_trap_curves, build_weight_chart)

st.set_page_config(page_title="Novelty plus Ziel – Sebastian Hanisch", layout="wide")

PRE = E.load_precomputed()
N = PRE["cases"]["l3"]["seeds"]
NL = PRE["cases"]["l4long"]["seeds"]


@st.cache_data(show_spinner=False)
def _pair(level, gens, seed, selection, k):
    return E.live_pair(level, gens, seed, selection, k)


@st.cache_data(show_spinner=False)
def _solvability(level):
    return E.solvability(level)


def _evals(x):
    return C.fmt_med(x)


def _cells(n):
    """Zellenzahl mit richtiger Einzahl (1 Zelle, 0 Zellen, 16 Zellen)."""
    return f"{n:.0f} Zelle" if round(n) == 1 else f"{n:.0f} Zellen"


def _pct(x):
    """Anteil als Prozent mit Leerzeichen (0.464 -> "46 %")."""
    return f"{100 * x:.0f} %"


def _mean(cell, key):
    return sum(cell[key]) / len(cell[key])


def _runs(case, sel):
    return E.study_cell(PRE, case, sel)["runs"]


def _succ(case, sel):
    return E.success_count(_runs(case, sel))


st.title("🧭 Novelty plus Ziel – wie viel Ziel verträgt die Suche?")
st.markdown(
    """
Reine Novelty Search ignoriert das Ziel, und das ist ihre Stärke in der Falle und ihre Schwäche überall sonst: wer das Ziel kennt, sollte es nutzen. Dieses Stück **gibt das Ziel zurück**, auf vier Arten: als **gewichtete Summe** aus Novelty
und Fitness, als **Pareto-Auswahl** mit zwei Zielen, als **Novelty mit lokalem Wettbewerb** (NSLC: Novelty plus „wer ist besser als seine Nachbarn“) und als **Novelty mit Minimalkriterium** (nur wer ein Mindestmaß erfüllt, pflanzt sich fort).
Es misst, **wie viel Ziel die Suche verträgt, bevor die Falle zurückkehrt**, und was davon auf der Schlangenlinie (Stufe 4) hilft.
"""
)
st.caption(
    "Stück 4 der Linie „Novelty Search und Quality-Diversity“; Labyrinth, Roboter, Suche und Verhalten (Endposition, ohne Archiv) wie in [Stück 2](https://sebastianhanisch-novelty-search-demo.streamlit.app/) "
    "und [Stück 3](https://sebastianhanisch-behavior-space-demo.streamlit.app/)."
)

with st.expander("Die vier Arten, das Ziel zurückzugeben", expanded=True):
    st.markdown(
        """
- **Gewichtete Summe:** `w · Fitness + (1 − w) · Novelty`, beide in jeder Generation auf 0 bis 1 skaliert. `w = 0` ist reine Novelty, `w = 1` reine Fitness; dazwischen wählt der Regler *Fitness-Gewicht* (0.1 bis 0.9).
- **Pareto:** zwei Ziele (Novelty, Fitness), beide maximieren; nicht dominierte Individuen zuerst, innerhalb einer Front die weniger gedrängten (NSGA-II).
- **NSLC:** zwei Ziele (Novelty, lokaler Wettbewerb). Lokaler Wettbewerb = Anteil der k nächsten Nachbarn im Verhaltensraum, die eine schlechtere Fitness haben: wer in seiner Gegend der Beste ist, wird belohnt, auch wenn die Gegend weit vom Ziel liegt.
- **Minimalkriterium:** Novelty, aber nur für Individuen, die ein Kriterium erfüllen (mindestens 3 Zellen vom Start; höchstens 19 beziehungsweise 12 Zellen vom Ziel). Wer es nicht erfüllt, pflanzt sich nicht fort.
- **Fitness** ist hier immer der negative Abstand der Endposition zum Ziel, also der täuschende Wegweiser aus Stück 1. Die **Falle** ist das lokale Minimum der Luftlinie: „nahe der Falle“ heißt höchstens 3 Zellen von ihrem Mittelpunkt.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
preset_cols = st.columns(len(C.PRESET_ORDER))
for i, name in enumerate(C.PRESET_ORDER):
    with preset_cols[i]:
        st.button(name, key=f"preset_{name}", width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP[name] or None)

st.caption(
    "🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, "
    "um ein Szenario zu teilen."
)

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    level = st.select_slider("Labyrinth (Stufe)", options=C.LEVELS, key="level_select", format_func=lambda v: C.LEVEL_LABELS[v], help="0 freies Feld bis 4 Schlangenlinie; je höher, desto länger der Umweg.")
    selection = st.selectbox("Auswahlverfahren", options=C.LIVE_SELECTIONS, key="selection_select", format_func=lambda v: C.SELECTION_LABELS[v],
                             help="Wie Novelty und Ziel zusammenkommen. Daneben läuft immer der Bezug (reine Novelty, wie in Stück 2) mit gleichem Seed und gleicher Anfangspopulation.")
    k = st.slider("Nachbarn k für die Novelty", *bounds("k_slider"), step=C.K_STEP, key="k_slider", help="Novelty = mittlerer Abstand zu den k nächsten Endpositionen; bei NSLC auch die Nachbarschaft für den lokalen Wettbewerb.")
    gens = st.slider("Generationen im Live-Lauf", *bounds("gens_slider"), step=C.GENS_STEP, key="gens_slider", help=f"Je Generation {C.POP} Individuen, also {C.POP} Evaluationen.")
    seed = st.number_input("Zufalls-Seed", min_value=bounds("seed_input")[0], max_value=bounds("seed_input")[1], step=1, key="seed_input", help="Bestimmt die Anfangspopulation und alle Zufallsentscheidungen; beide Suchen starten gleich.")
    st.button("🎲 Neuen Lauf würfeln", on_click=randomize_seed)

level, selection, k, gens, seed = int(level), str(selection), int(k), int(gens), int(seed)
sync_query_params({"level_select": level, "selection_select": selection, "k_slider": k, "gens_slider": gens, "seed_input": seed})

grid = M.make_maze(level)
lm = E.level_metrics(level)
free = lm["free"]

st.markdown("---")
st.markdown("## 🔗 Wie täuschend ist diese Stufe?")
st.caption(f"{C.LEVEL_LABELS[level]}. Die Maße stammen aus Stück 1: sie sagen, wie sehr die Fitness „Abstand zum Ziel“ in die Irre führt.")
m1 = st.columns(4)
m1[0].metric("Kürzester Weg", f"{lm['path']} Zellen", delta=f"{lm['detour']:.2f}-facher Umweg", delta_color="off")
m1[1].metric("Fitness-Distanz-Korrelation", f"{lm['fdc']:.2f}", help="1 = die Luftlinie führt zum Ziel; je kleiner, desto täuschender.")
m1[2].metric("Fallen (lokale Minima)", f"{lm['minima']}", help="Zellen, in denen die Luftlinie zum Ziel kleiner ist als in allen freien Nachbarzellen.")
m1[3].metric("Freie Zellen", f"{free}", help="So viele Zellen kann eine Endposition belegen.")
dist_ok, steps_used = _solvability(level)
st.caption(f"Lösbar? Ja: eine aus dem kürzesten Pfad abgeleitete Steuerfolge fährt ins Ziel (Abstand am Ende {dist_ok:.2f} Zellen, {steps_used} von {C.STEPS} Schritten benutzt).")

st.markdown("---")
st.subheader("📐 Ein Lauf: reine Novelty gegen gewähltes Verfahren")
with st.spinner(f"Lasse beide Suchen {gens} Generationen laufen …"):
    (base, base_track), (sel, sel_track) = _pair(level, gens, seed, selection, k)
base_ok, sel_ok = base.first_solved is not None, sel.first_solved is not None
c_base, c_sel = st.columns(2)
with c_base:
    st.markdown(f"**{C.SELECTION_LABELS[C.BASELINE]}**")
    b1 = st.columns(2)
    b1[0].metric("Bezug: Ergebnis", "Ziel erreicht" if base_ok else "Ziel verfehlt")
    b1[1].metric("Bezug: Evaluationen bis zum Erfolg", C.fmt_int(base.first_solved) if base_ok else "–")
    b2 = st.columns(2)
    b2[0].metric("Bezug: besuchte Zellen", f"{base.visited_cells[-1]} von {free}")
    b2[1].metric("Bezug: kleinster Restweg", _cells(base.best_progress[-1]))
    st.plotly_chart(build_run_figure(grid, base.end_positions[-1], base_track, base.visited_grid, base.last_scores), width="stretch", key=f"run_base_{level}_{gens}_{seed}_{k}")
with c_sel:
    st.markdown(f"**{C.SELECTION_LABELS[selection]}** ({C.GROUP_LABELS[C.SELECTION_GROUP[selection]]}, k = {k})")
    s1 = st.columns(2)
    s1[0].metric("Verfahren: Ergebnis", "Ziel erreicht" if sel_ok else "Ziel verfehlt")
    s1[1].metric("Verfahren: Evaluationen bis zum Erfolg", C.fmt_int(sel.first_solved) if sel_ok else "–")
    s2 = st.columns(2)
    s2[0].metric("Verfahren: besuchte Zellen", f"{sel.visited_cells[-1]} von {free}")
    s2[1].metric("Verfahren: kleinster Restweg", _cells(sel.best_progress[-1]))
    st.plotly_chart(build_run_figure(grid, sel.end_positions[-1], sel_track, sel.visited_grid, sel.last_scores), width="stretch", key=f"run_sel_{level}_{gens}_{seed}_{selection}_{k}")
st.caption("Hellblau: alle Zellen, in denen eine Fahrt je endete. Punkte: die Endpositionen der letzten Generation, gefärbt nach ihrem Auswahlwert (hell = bevorzugt). Orange: die beste Fahrt.")
col_c, col_d, col_e = st.columns(3)
with col_c:
    st.markdown("**Bester Abstand zum Ziel**")
    st.plotly_chart(build_progress_chart(base.best_distance, sel.best_distance, selection, lm["trap"]), width="stretch", key=f"progress_{level}_{gens}_{seed}_{selection}_{k}")
with col_d:
    st.markdown("**Besuchte Zellen**")
    st.plotly_chart(build_coverage_chart(base.visited_cells, sel.visited_cells, selection, free), width="stretch", key=f"coverage_{level}_{gens}_{seed}_{selection}_{k}")
with col_e:
    st.markdown("**Population nahe der Falle**")
    st.plotly_chart(build_trap_chart(base.trap_share, sel.trap_share, selection), width="stretch", key=f"trap_{level}_{gens}_{seed}_{selection}_{k}")
if lm["minima"] == 0:
    st.caption("Diese Stufe hat keine Falle: der Anteil nahe der Falle ist überall 0.")
if sel_ok and not base_ok:
    st.info(f"Mit „{C.SELECTION_LABELS[selection]}“ erreicht die Suche das Ziel nach {C.fmt_int(sel.first_solved)} Evaluationen, mit reiner Novelty in {gens} Generationen noch nicht (bester Abstand {min(base.best_distance):.2f} Zellen). "
            "Das ist ein Lauf; ob es typisch ist, zeigt die Studie unten.")
elif base_ok and not sel_ok:
    st.info(f"Mit reiner Novelty erreicht die Suche das Ziel nach {C.fmt_int(base.first_solved)} Evaluationen, mit „{C.SELECTION_LABELS[selection]}“ in {gens} Generationen noch nicht (bester Abstand {min(sel.best_distance):.2f} Zellen). "
            "Das ist ein Lauf; ob es typisch ist, zeigt die Studie unten.")
elif base_ok and sel_ok:
    st.info(f"Beide Suchen erreichen das Ziel: reine Novelty nach {C.fmt_int(base.first_solved)}, „{C.SELECTION_LABELS[selection]}“ nach {C.fmt_int(sel.first_solved)} Evaluationen. Ein einzelner Lauf zeigt Streuung; die Studie unten vergleicht {N} Läufe.")
else:
    st.info(f"Keine der beiden Suchen erreicht das Ziel in {gens} Generationen (bester Abstand reine Novelty {min(base.best_distance):.2f}, Verfahren {min(sel.best_distance):.2f} Zellen; kleinster Restweg "
            f"{base.best_progress[-1]:.0f} gegen {sel.best_progress[-1]:.0f} Zellen). Mehr Generationen oder ein anderer Seed können das ändern; die Studie unten fasst {N} Läufe zusammen.")

st.markdown("---")
st.subheader("🔬 Wie viel Ziel verträgt die Suche? Die Studie über 20 Läufe")
st.markdown(
    f"Stufe 3, je Verfahren {N} Läufe mit unterschiedlichem Seed, jeweils bis zu {C.fmt_int(PRE['budget'])} Evaluationen ({C.STUDY_GENS} Generationen), Verhalten = Endposition, k = {C.K_DEFAULT}, kein Archiv. "
    f"**Erfolg** heißt: irgendein Individuum endet höchstens {C.SUCCESS_RADIUS:g} Zellen vom Ziel. Der Median zählt nur die erfolgreichen Läufe."
)
st.plotly_chart(build_median_chart(PRE, "l3"), width="stretch", key="median_chart_l3")
st.markdown("**Das Gewicht der Fitness in der Summe**")
st.plotly_chart(build_weight_chart(PRE, "l3"), width="stretch", key="weight_chart_l3")
rows = []
for s in C.SELECTIONS:
    c3, c4 = E.study_cell(PRE, "l3", s), E.study_cell(PRE, "l4", s)
    rows.append(f"| {C.SELECTION_LABELS[s]} | {C.GROUP_LABELS[C.SELECTION_GROUP[s]]} | {E.success_count(c3['runs'])} von {N} | {_evals(E.median_solved(c3['runs']))} | {_mean(c3, 'cells'):.2f} | {_mean(c3, 'progress'):.2f} | "
                f"{E.success_count(c4['runs'])} von {N} | {_mean(c4, 'cells'):.2f} | {_mean(c4, 'progress'):.2f} |")
st.markdown("| Verfahren | Art | Stufe 3: Erfolg | Median der Evaluationen | besuchte Zellen (von 465) | Restweg | Stufe 4: Erfolg | besuchte Zellen (von 427) | Restweg |\n|---|---|---|---|---|---|---|---|---|\n" + "\n".join(rows))
ws = {s: _succ("l3", s) for s in C.WEIGHT_SELECTIONS}
mwp = [E.mannwhitney_p(_runs("l3", "novelty"), _runs("l3", s)) for s in C.SELECTIONS if s != "novelty" and None not in _runs("l3", s)]
st.info(
    f"**Das Gewicht:** mit Fitness-Gewicht 0.1 / 0.25 / 0.5 löst die Summe {ws['w10']} / {ws['w25']} / {ws['w50']} von {N} Läufen, mit 0.75 nur {ws['w75']}, mit 0.9 nur {ws['w90']} (reine Fitness: {ws['fitness']}, reine Novelty: {ws['novelty']}). "
    f"Die Median-Werte der erfolgreichen Läufe liegen bei Gewicht 0.1 bis 0.5 zwischen {_evals(min(E.median_solved(_runs('l3', s)) for s in ('w10', 'w25', 'w50')))} und {_evals(max(E.median_solved(_runs('l3', s)) for s in ('w10', 'w25', 'w50')))} "
    f"gegen {_evals(E.median_solved(_runs('l3', 'novelty')))} bei reiner Novelty, aber dieser Unterschied ist nicht belegt (Mann-Whitney-Test über die {N} Laufzeiten: p = {min(mwp):.2f} bis {max(mwp):.2f} für alle {len(mwp)} Verfahren, die alle {N} Läufe lösen). "
    f"**Das Ziel darf also mitreden, aber nicht führen:** es hilft nicht belegbar, und ab Gewicht 0.75 kehrt die Falle zurück. "
    f"Pareto ({_succ('l3', 'pareto')} von {N}), NSLC ({_succ('l3', 'nslc')} von {N}) und die drei Minimalkriterien ({_succ('l3', 'mc_start3')}, {_succ('l3', 'mc_goal19')} und {_succ('l3', 'mc_goal12')} von {N}) lösen die Falle ebenfalls. "
    f"**Stufe 4 mit 140 Schritten** löst keines der Verfahren ({sum(_succ('l4', s) for s in C.SELECTIONS)} Erfolge in {len(C.SELECTIONS) * N} Läufen)."
)

st.markdown("---")
st.subheader("🔬 Hilft das Ziel auf Stufe 4? Die Steuerfolge mit 300 Schritten")
st.markdown(
    f"Stufe 4 mit **300 statt 140 Schritten** (in Stück 3 die Länge, bei der Novelty die Schlangenlinie erstmals löst), je Verfahren **{NL} Läufe**, damit Unterschiede nicht im Rauschen verschwinden. Die Balken zeigen die Erfolgsquote mit "
    "95-%-Intervall (Wilson); die Zahl daneben sind die erfolgreichen Läufe."
)
st.plotly_chart(build_long_chart(PRE), width="stretch", key="long_chart")
nov_l = _runs("l4long", "novelty")
fp = {s: E.fisher_p(nov_l, _runs("l4long", s)) for s in C.SELECTIONS if s != "novelty"}
worse = [s for s in fp if fp[s] < 0.05 and _succ("l4long", s) < _succ("l4long", "novelty")]
better = [s for s in fp if fp[s] < 0.05 and _succ("l4long", s) > _succ("l4long", "novelty")]
st.info(
    f"Reine Novelty löst {_succ('l4long', 'novelty')} von {NL} Läufen, reine Fitness {_succ('l4long', 'fitness')}. Die Summen mit Gewicht 0.1 / 0.25 / 0.5 liegen mit {_succ('l4long', 'w10')} / {_succ('l4long', 'w25')} / {_succ('l4long', 'w50')} in derselben Größenordnung "
    f"(Fisher-Test gegen Novelty: p = {fp['w10']:.2f} / {fp['w25']:.2f} / {fp['w50']:.2f}), ebenso Pareto ({_succ('l4long', 'pareto')}, p = {fp['pareto']:.2f}), NSLC ({_succ('l4long', 'nslc')}, p = {fp['nslc']:.2f}) und das Minimalkriterium über den Start "
    f"({_succ('l4long', 'mc_start3')}, p = {fp['mc_start3']:.2f}). **Kein Verfahren ist belegt besser als reine Novelty** ({'keines mit p < 0.05 bei mehr Erfolgen' if not better else ', '.join(better)}). Belegt schlechter sind: "
    f"{', '.join(C.SELECTION_LABELS[s] + ' (' + str(_succ('l4long', s)) + ', p = ' + format(fp[s], '.3f') + ')' for s in worse)}. Diese p-Werte sind ohne Korrektur für die {len(fp)} Vergleiche gerechnet; das Minimalkriterium über das Ziel "
    "(12 Zellen) ist damit ein Grenzfall."
)

st.markdown("---")
st.subheader("🔬 Wo sammelt sich die Population? Die Falle")
st.markdown(
    f"Gemessen wird, welcher Anteil der Population in jeder Generation höchstens 3 Zellen von der Falle entfernt endet (Mittel über {N} Läufe, Stufe 3, 140 Schritte). Je höher die Kurve, desto mehr Individuen sammeln sich vor der Wand. "
    "Die Vermutung dazu: eine Auswahl, in der die Fitness mit zählt, hält Individuen an der Falle fest."
)
st.plotly_chart(build_trap_curves(PRE, "l3"), width="stretch", key="trap_curves_l3")
tr = {s: E.study_cell(PRE, "l3", s)["mean_trap"] for s in ("fitness", "novelty", "w50", "w75", "w90", "pareto", "nslc")}
late = {s: sum(v[100:]) / len(v[100:]) for s, v in tr.items()}
st.info(
    f"Im Mittel über die Generationen 101 bis {C.STUDY_GENS} stehen bei der Fitness {_pct(late['fitness'])} der Population nahe der Falle, bei der Summe mit Gewicht 0.9 {_pct(late['w90'])}, mit 0.75 {_pct(late['w75'])}, mit 0.5 {_pct(late['w50'])}, "
    f"bei Pareto {_pct(late['pareto'])}, bei NSLC {_pct(late['nslc'])} und bei reiner Novelty {_pct(late['novelty'])}. Der Anteil wächst also mit dem Gewicht der Fitness und erklärt, warum die Summe ab 0.75 kippt. **Die Pareto-Auswahl hält die Population "
    "dagegen nicht an der Falle fest** (Anteil wie bei reiner Novelty): die Vermutung, die Fitness im zweiten Ziel halte Individuen vor der Wand, ist für Pareto und NSLC nicht bestätigt."
)

st.markdown("---")
st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Das Ziel ist ein brauchbarer Wegweiser** | Die Fitness hier ist der täuschende Abstand aus Stück 1. Jedes Gewicht, das sie führen lässt, führt in die Falle (ab 0.75 kehrt sie zurück). Ein besseres Ziel (etwa der wahre Weg) wurde nicht gemessen. | kein Folgestück |
| **Ein Skalierungsverfahren genügt** | Die Summe skaliert beide Größen je Generation auf 0 bis 1. Andere Skalierungen (Ränge, feste Maße) verschieben die Gewichte; nicht gemessen. | kein Folgestück |
| **Die Minimalkriterien sind richtig gewählt** | Die drei Kriterien und ihre Schwellen sind von Hand gewählt; andere Schwellen können andere Zahlen liefern. | kein Folgestück |
| **Eine Suche mit fester Steuerfolge** | Kein Verfahren löst Stufe 4 mit 140 Schritten; mit 300 Schritten löst ein Teil der Läufe (höchstens 15 von 100), und kein Verfahren ist belegt besser als reine Novelty: der Engpass bleibt der Suchraum (Stück 3). | **Go-Explore** und **POET** (Folgestücke) |
| **Eine einzige Lösung genügt** | Alle Verfahren liefern den Weg zum Ziel, nicht eine Menge verschiedener guter Wege. | **MAP-Elites** (Folgestück) |
"""
)
st.caption(
    "Verwandt im Portfolio: [deception-maze-demo](https://sebastianhanisch-deception-maze-demo.streamlit.app/) (Stück 1: das täuschende Labyrinth), [novelty-search-demo](https://sebastianhanisch-novelty-search-demo.streamlit.app/) "
    "(Stück 2: Novelty Search), [behavior-space-demo](https://sebastianhanisch-behavior-space-demo.streamlit.app/) (Stück 3: Verhaltensräume), [nsga2-demo](https://sebastianhanisch-nsga2-demo.streamlit.app/) (NSGA-II mit zwei Zielen)."
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Größen.** Verhalten $b(g)$ = Endposition; Fitness $f(g) = -\lVert b(g) - z\rVert_2$; Novelty $\rho_k(i) = \frac{1}{k}\sum_{j \in N_k(i)} \lVert b_i - b_j\rVert_2$ mit den $k$ nächsten anderen Individuen $N_k(i)$ der Population.

**Gewichtete Summe.** $s_i = w\,\tilde f_i + (1-w)\,\tilde\rho_i$ mit $\tilde x_i = (x_i - \min_j x_j)/(\max_j x_j - \min_j x_j)$ je Generation (0, falls alle gleich sind).

**Pareto.** Individuum $i$ dominiert $j$, wenn $\rho_i \ge \rho_j$ und $f_i \ge f_j$ mit mindestens einer strengen Ungleichung. Front $0$ = nicht dominierte Individuen, Front $r$ = nicht dominiert nach Entfernen der Fronten $0,\dots,r-1$. Crowding $c_i$ je Front
und Ziel: Randpunkte $+1$, innere Punkte $+(x_{\text{nächster}} - x_{\text{voriger}})/\text{Spannweite}$. Auswahlwert $-\text{Front}(i) + \tfrac12\, c_i / \max_j c_j$.

**NSLC.** Lokaler Wettbewerb $q_i = \frac{1}{k}\,\bigl|\{ j \in N_k(i) : f_j < f_i \}\bigr|$; dann Pareto-Auswahl auf $(\rho_i, q_i)$.

**Minimalkriterium.** $s_i = \rho_i$, falls $\lVert b_i - s_0\rVert \ge \tau$ (Start $s_0$) beziehungsweise $-f_i \le \tau$, sonst $-1$; erfüllt niemand das Kriterium, gilt $s_i = \rho_i$ für alle.

**Auswahl.** Turnier der Größe 3, Elite 2, Mutation Gauß 0.25 auf 15 % der Gene, kein Archiv. Erfolg: $\lVert b(g) - z\rVert_2 < 2$.

Implementiert in `nobj_selection.py` (alle Auswahlwerte), `nobj_search.py` (Lauf, Anteil nahe der Falle), `nobj_maze.py` und `nobj_robot.py` (wie Stück 1).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html))."
)
