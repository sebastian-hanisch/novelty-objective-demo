# Novelty plus Ziel – wie viel Ziel verträgt die Suche? (Streamlit-Demo)

---

Interaktive Demo zu **Novelty Search mit Ziel**. **Viertes Stück der Konzepte-Linie „Novelty Search und Quality-Diversity“** im Portfolio von
[Sebastian Hanisch](https://sebastianhanisch.net) (Operations Research und Machine Learning): ein Verfahren, ein wachsendes Beispiel, jedes Folgestück hebt genau eine Annahme auf.

Derselbe Hof-Roboter, dasselbe täuschende Labyrinth und dieselbe Suche wie in [`deception-maze-demo`](https://github.com/sebastian-hanisch/deception-maze-demo) (Stück 1), [`novelty-search-demo`](https://github.com/sebastian-hanisch/novelty-search-demo)
(Stück 2) und [`behavior-space-demo`](https://github.com/sebastian-hanisch/behavior-space-demo) (Stück 3). Reine Novelty Search ignoriert das Ziel; dieses Stück **gibt es auf vier Arten zurück**: als **gewichtete Summe** aus Novelty und Fitness, als
**Pareto-Auswahl** (zwei Ziele), als **Novelty mit lokalem Wettbewerb** (NSLC) und als **Novelty mit Minimalkriterium**. Es misst, **wie viel Ziel die Suche verträgt, bevor die Falle zurückkehrt**, und ob ein Verfahren Stufe 4 besser löst als reine Novelty.

## Kernfrage

Hilft es, der Novelty-Auswahl das Ziel zurückzugeben, und ab wann schadet es?

## Modell und Methodik

- **Labyrinth, Roboter, Suche** (`nobj_maze.py`, `nobj_robot.py`, `nobj_search.py`): unverändert aus Stück 1 bis 3. 24 × 24 Zellen, Start (3.5, 3.5), Ziel (3.5, 20.5); Stufe 3 eine Querwand mit Öffnung rechts (Falle 9 Zellen vor dem Ziel), Stufe 4 drei Wände.
  Feste Steuerfolge aus 140 Befehlen (im Langlauf 300), kein Sensor; Verhalten = Endposition, **kein Archiv**. Genetischer Algorithmus mit 100 Individuen, Turnier der Größe 3, die zwei Besten bleiben, Mutation Gauß σ = 0.25 auf 15 % der Gene, keine Kreuzung.
  **Erfolg:** eine Endposition höchstens 2 Zellen vom Ziel.
- **Zwölf Auswahlverfahren** (`nobj_selection.py`, größer ist besser):
  - *Fitness* (−Abstand zum Ziel, Bezug aus Stück 1) und *Novelty* (mittlerer Abstand zu den k = 15 nächsten Endpositionen der Population, Bezug aus Stück 2).
  - *Gewichtete Summe* `w · Fitness' + (1 − w) · Novelty'` mit w = 0.1 / 0.25 / 0.5 / 0.75 / 0.9; beide Größen je Generation auf 0 bis 1 skaliert (Minimum und Maximum der Population).
  - *Pareto* (NSGA-II): zwei Ziele (Novelty, Fitness), nicht dominierte Fronten, innerhalb einer Front das normierte Crowding (halbes Gewicht, ändert nie die Front-Reihenfolge).
  - *NSLC*: zwei Ziele (Novelty, lokaler Wettbewerb = Anteil der k nächsten Nachbarn mit schlechterer Fitness), dann Pareto wie oben.
  - *Minimalkriterium*: Novelty, aber nur für Individuen, die das Kriterium erfüllen (mindestens 3 Zellen vom Start; höchstens 19 oder 12 Zellen vom Ziel); wer es verfehlt, bekommt den Wert −1, außer wenn niemand es erfüllt.
- **Messgrößen:** Evaluationen bis zum ersten Erfolg; besuchte Zellen; **wahrer Restweg** (kürzester Weg von der besten besuchten Zelle zum Ziel); **Anteil der Population nahe der Falle** (Endposition höchstens 3 Zellen vom Mittelpunkt eines lokalen Minimums der Luftlinie).
- **Statistik:** Erfolgsquoten mit 95-%-Wilson-Intervall; Vergleich mit reiner Novelty über den exakten Test nach Fisher (Erfolgsquoten) und den Mann-Whitney-Test (Laufzeiten, nur wo alle Läufe erfolgreich waren).
- **Gegenproben:** (1) jede Auswahlgröße von Hand an Mini-Mengen (Skalierung, Fronten mit gleichen Punkten, Crowding auf vier Punkten einer Geraden: 2, 4/3, 4/3, 2, lokaler Wettbewerb, Summe mit w = 0.25, Minimalkriterien samt Rückfall auf reine Novelty);
  (2) **Orakel:** Fronten gegen ein Abschälen in zwei Schleifen (ein anderer Algorithmus), Crowding gegen eine skalare Neuberechnung, lokaler Wettbewerb gegen scikit-learns `NearestNeighbors` und eine skalare Definition (auch mit doppelten Punkten), Novelty gegen
  SciPys `cKDTree`, besuchte Zellen und Falle-Anteil gegen eine Zählung über Tupel; (3) das Ziel wandert: nur Verfahren mit Fitness oder Zielabstand ändern ihre Läufe; (4) reine Novelty reproduziert die Läufe aus Stück 2 (Seeds 0 bis 5 auf Stufe 3: 1 246, 1 737, 515, 631, 2 035, 1 641
  Evaluationen); (5) Studienläufe werden mit demselben Seed frisch reproduziert, auch mit 300 Schritten.
- **Vorgerechnete Studie** (`generate_precomputed.py` → `precomputed_sweep.json`, rund zehn Minuten parallel): alle zwölf Verfahren auf Stufe 3 (140 Schritte, 20 Läufe), Stufe 4 (140 Schritte, 20 Läufe) und Stufe 4 mit 300 Schritten (**100 Läufe**, weil bei 20 Läufen
  die Unterschiede zwischen 0 und 4 Erfolgen im Rauschen verschwinden); je 400 Generationen (40 000 Evaluationen).

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen stehen in `tests/test_claims.py`. Der Median zählt nur die erfolgreichen Läufe; „nahe der Falle“ ist der Mittelwert über die Generationen 101 bis 400 auf Stufe 3.

| Verfahren | Art | Stufe 3: Erfolg von 20 | Median | besuchte Zellen (von 465) | Restweg | Population nahe der Falle | Stufe 4, 140 Schritte: Erfolg von 20 | Restweg | Stufe 4, 300 Schritte: Erfolg von 100 | p gegen reine Novelty (Fisher) |
|---|---|---|---|---|---|---|---|---|---|---|
| Fitness (Bezug aus Stück 1) | eine Größe | 11 | 1 267 | 343.90 | 5.55 | 52 % | 0 | 34.75 | 0 | 0.001 |
| Novelty allein (Bezug) | eine Größe | 20 | 988.5 | 465.00 | 0.00 | 7 % | 0 | 26.25 | 11 | – |
| Summe, Fitness-Gewicht 0.1 | gewichtete Summe | 20 | 709 | 465.00 | 0.00 | 7 % | 0 | 25.95 | 10 | 1.000 |
| Summe, Fitness-Gewicht 0.25 | gewichtete Summe | 20 | 570 | 465.00 | 0.00 | 8 % | 0 | 26.50 | 15 | 0.529 |
| Summe, Fitness-Gewicht 0.5 | gewichtete Summe | 20 | 827 | 464.80 | 0.00 | 10 % | 0 | 25.75 | 14 | 0.670 |
| Summe, Fitness-Gewicht 0.75 | gewichtete Summe | 16 | 1 231 | 398.35 | 3.45 | 28 % | 0 | 29.15 | 2 | 0.018 |
| Summe, Fitness-Gewicht 0.9 | gewichtete Summe | 12 | 931.5 | 357.10 | 5.50 | 46 % | 0 | 33.90 | 1 | 0.005 |
| Pareto (Novelty × Fitness) | Pareto-Auswahl | 20 | 925 | 465.00 | 0.00 | 8 % | 0 | 26.85 | 5 | 0.191 |
| NSLC (Novelty × lokaler Wettbewerb) | Pareto-Auswahl | 20 | 925.5 | 465.00 | 0.00 | 8 % | 0 | 27.10 | 6 | 0.311 |
| Minimalkriterium: ≥ 3 Zellen vom Start | Minimalkriterium | 20 | 1 064 | 465.00 | 0.00 | 7 % | 0 | 26.10 | 9 | 0.814 |
| Minimalkriterium: ≤ 19 Zellen vom Ziel | Minimalkriterium | 20 | 794.5 | 463.65 | 0.00 | 9 % | 0 | 25.70 | 4 | 0.105 |
| Minimalkriterium: ≤ 12 Zellen vom Ziel | Minimalkriterium | 20 | 1 114 | 457.55 | 0.00 | 14 % | 0 | 24.55 | 3 | 0.049 |

| Frage | Befund |
|---|---|
| Löst das Ziel in der Summe die Falle schneller? | **Nicht belegt.** Mit Fitness-Gewicht 0.1 / 0.25 / 0.5 lösen alle Läufe Stufe 3 (20 von 20), Median 709 / 570 / 827 gegen 988.5 bei reiner Novelty, aber der Mann-Whitney-Test über die Laufzeiten gibt p = 0.23 bis 0.89 für alle 8 Verfahren mit 20 von 20 (kleinstes p bei Summe 0.25 mit 0.23, größtes bei Minimalkriterium Start 3 mit 0.89). Der Unterschied liegt im Rauschen von 20 Läufen. |
| Ab wann schadet das Ziel? | **Ab Fitness-Gewicht 0.75.** Summe 0.75: 16 von 20 (Fisher gegen reine Novelty p = 0.106), Summe 0.9: **12 von 20** (p = 0.003), reine Fitness 11 von 20 (p = 0.001). Besuchte Zellen im Mittel 398.35 (0.75) und 357.1 (0.9) statt 465; schlimmster Lauf 7 977 und 7 826 Evaluationen gegen 2 356 bei Summe 0.5. |
| Warum kippt es? | Der Anteil der Population nahe der Falle wächst mit dem Gewicht: reine Novelty 7 %, Summe 0.5: 10 %, 0.75: **28 %**, 0.9: **46 %**, reine Fitness 52 %. |
| Halten Pareto und NSLC die Population an der Falle fest? | **Nein.** Pareto 8 % (8.1 %) und NSLC 8 %, reine Novelty 7 % (7.0 %). Beide lösen Stufe 3 in 20 von 20 Läufen (Median 925 und 925.5). |
| Und die Minimalkriterien? | Alle drei lösen Stufe 3 in 20 von 20 Läufen (Median 1 064 / 794.5 / 1 114; schlimmster Lauf bei „höchstens 12 Zellen vom Ziel“: 11 915, besuchte Zellen im Mittel 457.55). |
| Löst irgendein Verfahren Stufe 4 mit 140 Schritten? | **Keines:** 0 von 240 Läufen. Kleinster wahrer Restweg im Mittel 24.55 (Minimalkriterium Ziel 12) bis 33.9 (Summe 0.9) für die elf Verfahren mit Novelty, 34.75 für die Fitness. |
| Und mit 300 Schritten? (100 Läufe) | Erfolge von 100: Fitness **0**, reine Novelty **11** (Median 27 378; 95-%-Intervall 6.3 % bis 18.6 %), Summe 0.1 / 0.25 / 0.5 / 0.75 / 0.9: 10 / 15 / 14 / **2** / **1**, Pareto 5, NSLC 6, Minimalkriterium Start 3 / Ziel 19 / Ziel 12: 9 / 4 / 3. **Kein Verfahren ist belegt besser als reine Novelty** (kein p < 0.05 bei mehr Erfolgen; Summe 0.25: p = 0.53, Summe 0.5: p = 0.67). |
| Welche sind belegt schlechter? | Fitness (p = 0.001), Summe 0.9 (p = 0.005), Summe 0.75 (p = 0.018), Minimalkriterium Ziel 12 (p = 0.049, Grenzfall). Die p-Werte sind ohne Korrektur für die 11 Vergleiche gerechnet; bei Bonferroni (Schwelle 0.0045) bliebe nur die Fitness unter der Schwelle. |
| Wie nah kommt die Suche mit 300 Schritten? | Mittlerer kleinster wahrer Restweg: Fitness 17.4, reine Novelty 7.97, Summe 0.25: 7.34, Summe 0.9: 16.43. |

## Befunde und Korrekturen gegenüber der Vorab-Messreihe

- **Pareto fällt nicht heraus; die Vorab-Messung war ein Artefakt.** In der Vorab-Messung (eine erste, eigene Fassung des Crowding) löste Pareto mit 300 Schritten nur 2 von 100 Läufen („deutlich schlechter als Novelty“), und ich vermutete, die erste Front halte die
  Population an der Falle fest. Mit dem gebauten, getesteten Auswahlcode löst Pareto **5 von 100** (Fisher gegen reine Novelty p = 0.19, nicht belegt), und der Falle-Anteil beträgt 8 % wie bei reiner Novelty. **Die Vermutung ist nicht bestätigt.** Die Vorab-Fassung hatte
  Randpunkte beim Crowding anders behandelt (überschrieben statt addiert); Fronten und Crowding sind jetzt von Hand und gegen unabhängige Referenzen geprüft.
- **„Ein wenig Ziel hilft“ ist nicht belegt.** Die Vorab-Medians (570 bis 827 gegen 988.5) sahen nach einem kleinen Gewinn aus; über 20 Läufe ist der Unterschied nicht unterscheidbar vom Rauschen (p = 0.23 bis 0.89).
- **Der Schaden des Ziels ist dagegen belegt und hat eine Ursache.** Ab Gewicht 0.75 sinkt die Erfolgsquote auf Stufe 3 (16 und 12 von 20), und der Anteil nahe der Falle steigt von 7 % über 28 % auf 46 %. Das ist eine Kante: bei Gewicht 0.5 (10 %) merkt man noch nichts.
- **Mehr Beobachtungen entzerren das Bild.** Bei 20 Läufen auf Stufe 4 mit 300 Schritten sah es nach Unterschieden zwischen 0 und 4 Erfolgen aus (Summe 0.5: 4 von 20); mit 100 Läufen ist es 14 von 100 gegen 11 von 100 für reine Novelty, kein Unterschied.
- **Stufe 4 bleibt eine Frage des Suchraums.** Mit 140 Schritten löst kein Verfahren auch nur einen Lauf; mit 300 Schritten löst ein Teil der Läufe, und die Rangfolge der Verfahren ändert daran nichts (Stück 3).

## Ehrliche Grenzen

- **Die Fitness ist der täuschende Abstand.** Jedes Gewicht, das sie führen lässt, führt in die Falle. Ein besseres Ziel (etwa der wahre Weg zum Ziel) wurde in dieser Reihe nicht als Fitness gemessen.
- **Eine Skalierung, eine Wahl.** Die Summe skaliert beide Größen je Generation auf 0 bis 1. Ränge oder feste Maße würden die Gewichte verschieben; **nicht gemessen**.
- **Minimalkriterien und Schwellen von Hand gewählt** (3 Zellen, 19 und 12 Zellen); andere Schwellen können andere Zahlen liefern.
- **Crowding-Gewicht fest.** Das normierte Crowding geht mit höchstens 0.5 in den Auswahlwert ein; andere Gewichte oder die Standardregel (Randpunkte unendlich) sind nicht gemessen.
- **20 Läufe auf Stufe 3 und 4 mit 140 Schritten.** Unterschiede zwischen 18 und 20 von 20 oder zwischen Medianen innerhalb von 1.5-fach liegen im Rauschen. Der Langlauf mit 100 Läufen ist genauer, aber ein Intervall von 6 % bis 19 % bleibt breit.
- **Mehrfache Vergleiche.** Die 11 Fisher-Tests auf Stufe 4 sind nicht korrigiert; ein p von 0.049 ist ein Grenzfall.
- **Ein einfacher GA ohne Kreuzung;** Kreuzung und andere Operatoren würden die Zahlen ändern (nicht gemessen).

## Verwandte Demos im Portfolio

- [`deception-maze-demo`](https://github.com/sebastian-hanisch/deception-maze-demo): Stück 1, das täuschende Labyrinth und die zielgetriebene Suche.
- [`novelty-search-demo`](https://github.com/sebastian-hanisch/novelty-search-demo): Stück 2, Novelty Search mit und ohne Archiv.
- [`behavior-space-demo`](https://github.com/sebastian-hanisch/behavior-space-demo): Stück 3, zwölf Verhaltensräume.
- [`nsga2-demo`](https://github.com/sebastian-hanisch/nsga2-demo): NSGA-II mit zwei Zielen, die Auswahlregel hinter dem Pareto-Verfahren.

## Bewusst nicht umgesetzt

Jede dieser Annahmen hebt ein Folgestück der Linie auf:

| Annahme | Folgestück |
|---|---|
| Das Ziel ist ein brauchbarer Wegweiser | kein Folgestück |
| Ein Skalierungsverfahren genügt | kein Folgestück |
| Die Minimalkriterien sind richtig gewählt | kein Folgestück |
| Eine Suche mit fester Steuerfolge (Stufe 4 hängt am Suchraum) | Go-Explore und POET (Folgestücke) |
| Eine einzige Lösung genügt | MAP-Elites (Folgestück) |

## Tests

255 Tests, rund anderthalb Minuten: Labyrinth und Roboter (aus Stück 1), alle Auswahlgrößen von Hand an Mini-Mengen, Fronten, Crowding, lokaler Wettbewerb und Novelty gegen unabhängige Referenzen, Suche (Reproduzierbarkeit je Verfahren,
Zählung der Evaluationen, besuchte Zellen, Restweg, Falle-Anteil, wer das Ziel sieht), Auswertung und Vollständigkeit der vorgerechneten Datei (inklusive Neurechnung einzelner Läufe und eines Langlaufs, Wilson-, Fisher- und Mann-Whitney-Hilfen), Presets und Permalink,
Diagramme (gesperrte Achsen, Fehlerbalken), AppTest-Rauchtests mit festem Würfel-Seed, der Smoke-Test der Portfolio-Vorlage, ein Quelltext-Test gegen Satz-Komma-Fehler und `test_claims.py` für jede Zahl dieser README.

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Oberfläche |
| `nobj_maze.py`, `nobj_robot.py` | Labyrinth, Breitensuche, Täuschungsmaße; Kinematik, Population, Pfad-Regler (wie Stück 1) |
| `nobj_selection.py` | alle Auswahlwerte: Skalierung, Fronten, Crowding, Pareto, lokaler Wettbewerb, Minimalkriterium |
| `nobj_search.py` | Genetischer Algorithmus mit wählbarem Auswahlverfahren, besuchte Zellen, Restweg, Falle-Anteil |
| `nobj_evaluation.py` | Maße je Stufe, Lösbarkeit, Live-Paar, Zugriff auf die Studie, Wilson-, Fisher-, Mann-Whitney-Hilfen |
| `generate_precomputed.py` | rechnet die Studie vor → `precomputed_sweep.json` |
| `nobj_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `nobj_presets.py`, `nobj_constants.py` | Presets, Permalink, Grenzen |
| `tests/` | siehe oben |

## Literatur

- Lehman, J., Stanley, K. O. (2011): Abandoning objectives: evolution through the search for novelty alone. *Evolutionary Computation* 19(2), 189–223.
- Lehman, J., Stanley, K. O. (2010): Revising the evolutionary computation abstraction: minimal criteria novelty search. *Proceedings of the 12th Annual Genetic and Evolutionary Computation Conference (GECCO '10)*, 1031–1038.
- Lehman, J., Stanley, K. O. (2011): Evolving a diversity of virtual creatures through novelty search and local competition. *Proceedings of the Genetic and Evolutionary Computation Conference (GECCO '11)*, 211–218.
- Deb, K., Pratap, A., Agarwal, S., Meyarivan, T. (2002): A fast and elitist multiobjective genetic algorithm: NSGA-II. *IEEE Transactions on Evolutionary Computation* 6(2), 182–197.

## Lokal ausführen

```
pip install -r requirements.txt
streamlit run app.py
```

Tests: `pip install -r requirements-dev.txt` und `python -m pytest tests/ -v`. Studie neu rechnen: `python generate_precomputed.py` (rund zehn Minuten).

Gebaut mit Streamlit und Plotly (die Simulation rechnet NumPy).
