# Primal-dualer Schritt im Kegel – Jordan-Produkt, Skalierung und ein Zertifikat – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-kegel-primal-dual-demo.streamlit.app/)** -->

Stück 6 von 16 der **Konzepte-Reihe „Konvexe und konische Optimierung“** für die Website „Sebastian Hanisch – Operations Research und Machine Learning“. Wie jede Konzepte-Demo dreht sich alles um **ein** Verfahren mit wachsendem Beispiel: ein Löser für Kegelprogramme, dessen Kegel von Stück zu Stück wächst (Orthant, Lorentz-Kegel, PSD-Kegel, später weitere). Vorgänger sind [konvexitaet-zertifikat-demo](https://github.com/sebastian-hanisch/konvexitaet-zertifikat-demo) (Stück 1), [kegelform-demo](https://github.com/sebastian-hanisch/kegelform-demo) (Stück 2: der Löser als Rechenwerk), [kruemmungsregeln-demo](https://github.com/sebastian-hanisch/kruemmungsregeln-demo) (Stück 3: Krümmungsregeln übersetzen ein Ausdrucksprogramm in ein Kegelprogramm), [kegel-dualitaet-demo](https://github.com/sebastian-hanisch/kegel-dualitaet-demo) (Stück 4: sein Dual) und [kegel-barriere-demo](https://github.com/sebastian-hanisch/kegel-barriere-demo) (Stück 5: die Barrierenmethode). Nachfolger ist der Exponentialkegel (Stück 7). Die Linien-Spezifikation und die Einbindung in die Website folgen, wenn die Reihe vollständig ist.

## Kernfrage

**Was leistet der primal-duale Schritt im Kegel?** Die Barriere aus Stück 5 nimmt an, dass auf dem zentralen Pfad `ν/t` die Lücke ist. Der primal-duale Löser rechnet den Dualpunkt mit: er löst `min cᵀx` mit `s = b − A x ∈ K` und `Aᵀz = −c`, `z ∈ K` zugleich, und die Lücke `cᵀx + bᵀz` ist ein Zertifikat aus einem echten Dualpunkt. Dafür muss die LP-Formel `s_i z_i = μ` in die Kegel übertragen werden, und das geht nicht wörtlich: im Orthant ist das Produkt komponentenweise, im Lorentz- und PSD-Kegel braucht es das **Jordan-Produkt** (für Matrizen `(UV + VU)/2`, weil `UV` nicht symmetrisch ist). Die **Nesterov–Todd-Skalierung** macht den Schritt invariant unter den Automorphismen des Kegels. Die Demo misst, **was jeder der drei Schritte leistet** (wörtlich, Jordan, skaliert), was die Skalierung bringt und was nicht, wie viele Iterationen das gegenüber der Barriere kostet und warum ein echter Dualpunkt den falschen Wert aus Stück 3 nicht zulässt.

Das Beispiel bleibt die kleinste umschließende Form einer Punktwolke in den sechs Formen aus Stück 2 (Würfel, Raute, Kugel als Lorentz- und PSD-Kegel, Ellipse achsenparallel und allgemein; die Ellipsen um den Ursprung mit Ziel kleinste Summe der quadrierten Halbachsen).

## Modell

- **Kern** (`kegel_kern.py`, wortgleich mit Stück 2, 3, 4 und 5, unverändert): Kegel, Barrieren, Barrierenmethode (hier nur als Vergleich und zur Kontrolle der Werte), Problemtyp `ConeProblem`.
- **Kegelalgebra** (`kpd_cones.py`): Einselement, Jordan-Produkt (Orthant komponentenweise; Lorentz `(uᵀv, u₀v̄ + v₀ū)`; PSD `(UV + VU)/2`), Schrittweite bis zum Rand (Lorentz: kleinste Nullstelle einer quadratischen Gleichung; PSD: kleinster Eigenwert), Nesterov–Todd-Skalierung `W` mit `W z = W⁻¹ s = λ` je Kegel. Der Lorentz-Anteil benutzt die geschlossene Jordan-Wurzel statt einer Eigenzerlegung (die liefert nahe am Rand negative Eigenwerte).
- **Löser** (`kpd_solver.py`): Mehrotra-Prädiktor-Korrektor mit einem dichten KKT-System `[[0, Aᵀ], [A, −H]]` je Iteration; Start aus den kleinsten Quadraten mit Verschiebung in den Kegel (kein streng zulässiger Startpunkt nötig); `σ = (1 − α_aff)³`, Schritt 99 % des Wegs zum Rand; Ausgang als Status (`gelöst`, `Schrittgrenze erreicht`, `steht still`, `verlässt den Kegel`, `Matrix singulär`, `Rechengrenze`) statt Absturz. Drei Richtungen: **wörtlich** (die LP-Formel auf allen Koordinaten), **Jordan** (`L(z)⁻¹L(s)`, ohne Skalierung), **NT** (skaliert, `W²` im System).
- **Beispiele** (`kpd_examples.py`): der Ausreißer aus Stück 3 als literale Instanz, die Fälle ohne Slater-Punkt aus Stück 4, ein unzulässiges LP, Zufallsprogramme (streng zulässig im Primal und im Dual durch Konstruktion `b = A x₀ + s₀`, `c = −Aᵀ z₀`) und Kegelautomorphismen (PSD `X ↦ T X Tᵀ`, Lorentz `P(w) = 2 w wᵀ − J`) mit mitgenommenem Start.
- **Rückumformung** wie in Stück 2: das Ergebnis wird in der Ausgangsaufgabe und im Kegelprogramm geprüft. **Kein Eingriff in den Kern:** alles Neue steht in `kpd_*.py`; BLAS-Threads laufen mit einem Thread (`threadpoolctl`), weil die Systeme klein sind (das dichte KKT-System hat die Größe Zeilen + Variablen).

## Methodik

Fünf Experimente mit live berechnetem Verdikt: (1) der LP-Schritt, Kegel für Kegel (die sechs Formen mit den drei Richtungen, dazu Zufallsprogramme je Kegelmischung); (2) das Matrixprodukt ist nicht symmetrisch (von Hand mit 2×2-Matrizen, dann entlang des NT-Laufs); (3) Skalierung und Invarianz (Identitäten an gezogenen Punkten, dann ein Zufallsprogramm mit Kegelautomorphismus der Kondition κ umgeschrieben, Regler κ); (4) Iterationen gegen ν und Dimension, gegen die Newton-Schritte der Barriere; (5) das Zertifikat und der Start (der Ausreißer aus Stück 3, die Fälle ohne Slater-Punkt, das LP ohne zulässigen Punkt mit dem Farkas-Strahl).

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen stehen in `tests/test_claims.py` (wo dort nur ein Band geprüft wird, wegen der Rundung auf Linux, steht die gemessene Zahl trotzdem hier); Sweeps laufen über feste Wolken `SWEEP_SEEDS` (drei) und feste Zufallsprogramme. Die Standardwolke (20 Punkte, Streckung 4, 45 Grad, Seed 35) ist ausdrücklich eine einzelne.

| Frage | Befund |
|---|---|
| Standardwolke, Kugel (Lorentz-Form) | **NT 11, Jordan 8 Iterationen**, der wörtliche Schritt steht nach 6 still (Schrittlänge 0). Die Barriere aus Stück 5 braucht 34 Newton-Schritte in 13 Runden. NT-Zertifikat `cᵀx + bᵀz` = 2,0·10⁻⁸, Abweichung vom Referenzwert 3,4·10⁻¹⁰ |
| Die sechs Formen, drei Schritte (Standardwolke) | Würfel 8 und Raute 9 Iterationen in **allen drei** Schritten (im Orthant ist es derselbe Schritt). Kugel (Lorentz) NT 11 / Jordan 8; Kugel (PSD) NT 10 / Jordan 8; allgemeine Ellipse NT 23 / Jordan 20 (Barriere 57); **achsenparallele Ellipse: NT endet an der Rechengrenze** (24 Iterationen, Abweichung 1,8·10⁻⁵); Jordan meldet sie auf dem Entwicklungsrechner nach 22 Iterationen als optimal (Abweichung vom Referenzwert 4,8·10⁻⁷, nicht 10⁻⁸), auf dem GitHub-Runner endet auch er mit singulärem KKT-System (CPU-abhängig, kein Test verlangt hier ein Ergebnis). Der wörtlich übertragene Schritt scheitert in **allen vier** Formen mit Lorentz- oder PSD-Kegel (zweimal steht er still, zweimal wird das KKT-System singulär) |
| Zufallsprogramme (je 12, streng zulässig durch Konstruktion) | Wörtlicher Schritt: Orthant 12 von 12, Lorentz-Kegel, PSD-Kegel und gemischt **0 von 12**. Jordan **12 von 12** mit Median 8 / 6 / 7 / 8 Iterationen (Orthant / Lorentz / PSD / gemischt); NT 12 von 12 mit Median 8 / 9 / 8,5 / 10,5. Der Jordan-Schritt braucht in den Mischungen mit Lorentz- oder PSD-Kegel **weniger** Iterationen als der skalierte (im Orthant gleich viele) |
| Nichtkommutativität (PSD-Block, Standardwolke) | Von Hand: `S = diag(2, 1)`, `Z = [[1, 1], [1, 2]]` gibt `S Z = [[2, 2], [1, 2]]` und `Z S = [[2, 1], [2, 2]]`; der antisymmetrische Anteil hat die Norm 0,707 bei `‖S Z‖` = 3,606. Entlang des NT-Laufs der PSD-Kugel wächst `‖SZ − ZS‖/(‖S‖‖Z‖)` von 8·10⁻¹⁷ am Start über 3,4·10⁻³ (Iteration 3) auf **0,14** (Iteration 6), danach fällt der Wert wieder (am Ende 0,078); gemessen am ersten PSD-Block; bei der allgemeinen Ellipse bis 0,091 |
| Identitäten der NT-Skalierung (200 Punkte je Kegel: Orthant 4, Lorentz 3 und 6, PSD 2, 3 und 5) | `Q z = s`, `W z = λ`, `W⁻¹ s = λ`, `W` symmetrisch: größte Abweichung **1,0·10⁻¹⁴**; kleinster Eigenwert von `W` 0,11 (positiv definit) |
| Invarianz (zwei PSD-Blöcke 3×3, 10 Programme je Stufe, Start mitgenommen) | Median der NT-Iterationen **8,5 / 8,5 / 8,5 / 16** bei κ = 1 / 10² / 10⁴ / 10⁶: für κ bis 10⁴ unabhängig von κ; bei κ = 10⁶ gelingt nur 1 von 10 Läufen, mit 16 Iterationen. **Außerdem:** gelöst NT 10 / 10 / **4** / 1 von 10, Jordan 10 / 9 / **9** / 2. Die Iterationszahl bleibt bis κ = 10⁴ gleich, die Erfolgsquote nicht |
| Iterationen gegen ν (Kugel, n = 4 bis 80, ν = 8 bis 160, Zeilen 12 bis 240; Median über 3 feste Wolken) | NT **9 bis 12**, Jordan 6 bis 9 (Jordan steigt mit ν leicht), Barriere **29 bis 34 Newton-Schritte** in 12 bis 13 Runden: die Barriere braucht mehr als doppelt so viele Iterationen wie der primal-duale Schritt |
| Iterationen gegen die Dimension (allgemeine Ellipse, d = 2 bis 6, PSD-Block 4 bis 12, Variablen 6 bis 42, ν = 14 bis 34) | NT 12 / 16 / 17 / 17, Jordan 10 / 11 / 12 / 13, Barriere 42 / 46 / 57 / 58. Eine Iteration löst ein dichtes KKT-System der Größe Zeilen + Variablen (Kugel der Standardwolke: 60 + 3 = 63; d = 6: 100 + 42 = 142): der Vergleich zählt Iterationen, keine Zeit |
| Der Ausreißer aus Stück 3 (Optimum 1,0·10⁻⁶) | Barriere ohne Absicherung: Wert **190,386** (190,368 auf dem GitHub-Runner: Rundung, CPU-abhängig) bei Lückenschranke 1,7·10⁻¹⁰ (`far_from_path = True`). **Primal-dual (NT): optimal in 18 Iterationen, Primal 1,00004·10⁻⁶ und Dual 9,9992·10⁻⁷ klammern das Optimum**, Lücke `cᵀx + bᵀz` = 1,2·10⁻¹⁰, Residuen unter 10⁻⁹ |
| Ohne Slater-Punkt (Stück 4) | Lückenbeispiel ε = 0,1 / 10⁻³ / 10⁻⁵: Primal = Dual = **−1** in 7 / 17 / 28 Iterationen, `‖x‖` wächst 11 / 2,3·10³ / 1,2·10⁵. Nicht angenommenes Infimum: Primal 4,9·10⁻⁸ (der Status meldet „gelöst“), aber `‖x‖` = 2,2·10⁷ (39 Iterationen). Schwach unzulässig `[[x,1],[1,0]] ⪰ 0`: nach 60 Iterationen kein Abschluss, `‖x‖` über 10⁹ (gemessen 1,5·10¹¹) |
| LP ohne zulässigen Punkt (`x ≥ 1`, `x ≤ 0`) | Abbruch nach 8 Iterationen (Band 5 bis 12), der Dualpunkt läuft davon (`‖z‖` über 10¹⁰, hier 10²⁹; die genaue Zahl hängt vom Prozessor ab); normiert ist `ẑ` = (1, 1)/√2 ein **Farkas-Strahl** mit `Aᵀẑ` = 0 und `bᵀẑ` = −0,707 < 0 |

## Befunde und Korrekturen gegenüber dem Plan

- **Die Hypothese der Gliederung ist nur zur Hälfte bestätigt.** Die Gliederung vermutete: „der naive LP-Schritt verliert im PSD-Kegel die Symmetrie, die Skalierung rettet ihn“. Gemessen: wörtlich übertragen scheitert der Schritt sofort, **aber** mit dem Jordan-Produkt läuft er auf diesen Zufallsprogrammen (Gauß-Daten, vier Mischungen) sogar schneller als mit Skalierung (Median 6 bis 8 gegen 8,5 bis 10,5 Iterationen in den drei Kegelmischungen). Die Skalierung bringt die Invarianz unter Kegelautomorphismen und die Theorie, nicht kürzere oder zuverlässigere Läufe. Die Demo erzählt es so, und der Text sagt nirgends, der Jordan-Schritt „breche“.
- **Die Skalierung macht den Löser nicht robuster.** Bei κ = 10⁴ ist die Erfolgsquote mit NT schlechter als mit Jordan (4 gegen 9 von 10); bei κ = 10⁶ (1 gegen 2) und κ = 100 (10 gegen 9) ist kein deutlicher Unterschied zu sehen. Der Plan nannte aus 40 Programmen 21 gegen 38 von 40 (52 % gegen 95 %); die App rechnet mit 10 Programmen je Stufe (40 % gegen 90 %). Die Richtung stimmt, die Zahl schwankt mit der Stichprobe: Sie steht hier als Zahl dieser 10 Programme, nicht als Rate.
- **Kleinere Stichproben als in der Phase 0.** Plan: 100 Zufallsprogramme je Mischung, Kugel bis n = 320, 5 Wolken je Punktzahl. App: 12 Programme je Mischung, n bis 80, 3 feste Wolken, weil das dichte KKT-System in der App live gerechnet wird (n = 320 braucht auf dem Entwicklungsrechner einige Sekunden bis eine Minute je Lauf). Die Phase-0-Zahlen für n = 320 (NT 13 Iterationen) stehen nur im Plan, nicht in einem Test dieses Repos.
- **Plan: „NT: 599 von 600 optimal (ein Lauf verlässt den Kegel)“.** In den 12 Programmen je Mischung der App ist es 12 von 12; der Test verlangt mindestens 11 (Rundung, CPU).
- **Ein „gelöst“ ist kein Existenzbeweis.** Beim nicht angenommenen Infimum meldet der Löser den Status „gelöst“ (die Lücke fällt unter die Toleranz), obwohl `x` gegen unendlich läuft. Das ist keine Fehlmeldung des Löser-Codes, sondern die Grenze eines Zertifikats ohne Slater-Punkt: Primal und Dual stimmen, ein Minimierer existiert nicht. Die Tabelle zeigt deshalb `‖x‖` neben dem Status.
- **Die achsenparallele Ellipse** (`t ≥ 1/p` mit `t ≈ 80`, Dualwerte bis 3·10³) bleibt die eine Form, an der der Löser an der Rechengenauigkeit endet; wo genau und mit welcher Richtung, hängt von der CPU ab: auf dem Entwicklungsrechner endet NT (Abweichung 1,8·10⁻⁵) und Jordan löst (22 Iterationen), auf dem GitHub-Runner endet auch Jordan (singuläres KKT-System). Die Tests verlangen für diese Form nur, dass der wörtliche Schritt scheitert und dass jeder gemeldete Erfolg stimmt. Ruiz-Äquilibrierung der Daten und ein Rückfall auf reine Zentrierung bei kleiner Schrittlänge wurden in der Vormessung geprüft und **nicht eingebaut**: die Äquilibrierung brachte nichts, der Rückfall machte die allgemeine Ellipse schlechter (42 statt 25 Iterationen). Diese Messung steht in den Vormessungs-Skripten der Planung (`docs/planung/konvexe-optimierung-planung/vormessung/`, außerhalb dieses Repos), nicht in einem Test hier.
- **Quellen:** Nesterov und Todd (Mathematics of Operations Research 22 (1), 1997, S. 1–42; Seitenangabe aus der Datenbank des IAOR, nicht aus dem Verlagsexemplar), Alizadeh, Haeberly und Overton (SIAM Journal on Optimization 8 (3), 1998, S. 746–768) und Vandenberghe (Dokumentation von CVXOPT, ohne Jahr) sind per Suche geprüft. Die Beweise der Quellen (Eindeutigkeit der Skalierung, Konvergenzaussagen) habe ich nicht nachgelesen und beweise sie hier nicht.
- **Port:** 9304 (`launch.json` und Netstat gelesen, frei).

## Ehrliche Grenzen

- **Die Iterationszahlen sind ein Messbefund dieser Implementierung** (dichtes KKT-System, Mehrotra-Schritt, Start nach CVXOPT, Toleranz 10⁻⁸), keine Eigenschaft der Verfahren. Dass der primal-duale Schritt weniger als die Hälfte der Barrieren-Schritte braucht, gilt für diese Wolken und diese Parameter; die Theorie nennt Schranken mit `√ν` für beide.
- **Iterationen sind keine Zeit.** Eine Iteration löst ein dichtes System der Größe Zeilen + Variablen, ein Newton-Schritt der Barriere eines der Variablen. Zeiten hängen vom Rechner ab und werden weder gemessen noch getestet.
- **Der Löser ist ein Prototyp, nicht Clarabel.** Er löst die gut gestellten Formen; bei großer Verzerrung κ und bei der achsenparallelen Ellipse endet er an der Rechengrenze. Das ist ein Befund und kein Mangel, der hier „repariert“ wird (keine Äquilibrierung, kein Rückfall). cvxpy mit Clarabel löst dieselben Programme (Orakel).
- **Keine homogene Einbettung.** Bei Programmen ohne Slater-Punkt und bei unzulässigen Programmen läuft ein Iterat davon; die Demo zeigt, was man dann sieht (der Dualpunkt läuft auf einen Strahl, normiert ist er das Farkas-Zertifikat), aber sie meldet die Unzulässigkeit nicht selbst. Ein Einbettungsverfahren würde es sauber melden; es ist nicht gebaut.
- **Größe:** die App begrenzt die Punktzahl auf 40 (Sweeps bis 80); n = 320 braucht auf dem Entwicklungsrechner einige Sekunden bis eine Minute je Lauf.
- **„Skalierung“** meint die Skalierung des Newton-Systems mit `W`, nicht die Äquilibrierung der Daten.
- **Der Exponentialkegel** ist nicht selbstdual; diese Skalierung gilt für ihn nicht (Stück 7).

## Verifikation

- **Kegelalgebra** (`tests/test_cones.py`): Jordan-Produkt von Hand (Orthant, Lorentz `u = (2; 1, 0)`, `v = (3; 0, 1)` gibt `(6; 3, 2)`, PSD `(UV + VU)/2`) und die Gesetze (Einselement, kommutativ, Matrixform); das Matrixprodukt zweier positiv definiter Matrizen ist nicht symmetrisch, das Jordan-Produkt schon; Schrittweite bis zum Rand von Hand und als Eigenschaft (landet genau auf dem Rand); NT-Skalierung von Hand (Orthant, diagonale PSD-Blöcke) und die Identitäten `Q z = s`, `W z = λ`, `W⁻¹ s = λ`, `W` symmetrisch positiv definit an 100 Zufallspunkten je Kegel, auch nahe am Rand des Lorentz-Kegels; Verträglichkeit mit einem Automorphismus.
- **Löser** (`tests/test_solver.py`): die drei Mini-Programme von Hand (Optimum 1; 5; 1) mit ihren Dualpunkten (`z = 1`; `z = (1; −3/5, −4/5)`; `Z = ½ [[1, −1], [−1, 1]]`), das Zertifikat als Differenz der beiden Zahlen, Verlauf und Iterate im Inneren der Kegel, der Start liest `x₀` nicht, die drei Schritte sind im Orthant derselbe und scheitern wörtlich in den Kegelformen, Jordan und NT gegen die Barriere, flache Iterationszahl in der Größe, der Ausreißer gelöst und geklammert, Statusfälle (`max_iter`, unbekannte Skalierung), der Farkas-Strahl von Hand.
- **Beispiele** (`tests/test_examples.py`): der Ausreißer gleich dem aus den Kern-Tests, Matrizen des Lückenbeispiels von Hand, Zufallsprogramme in Primal und Dual streng zulässig, Kegelautomorphismen (Kondition, Start mitgenommen, Optimum bleibt, `sᵀz` bleibt), Invarianz der NT-Iterationszahl.
- **Orakel** (`tests/test_oracle_kegel_primal_dual.py`): Optimalwert aller sechs Formen und von 16 Zufallsprogrammen gegen cvxpy mit Clarabel (Primal und Dual); Dualzulässigkeit `Aᵀz = −c`; Skalierung des Ziels und der Punkte; das Optimum des Ausreißers aus der Handrechnung `(10⁻⁴/0,1)² = 10⁻⁶`.
- **Randfälle** (`tests/test_edge_cases.py`): zwei Punkte von Hand, ein Punkt, d = 1 und 3 in jeder Form, doppelte Punkte, sehr kleine und große Koordinaten, Startpunkt mit falscher Größe oder außerhalb des Kegels (nie „gelöst“), Nullzeile, Eckwerte der Regler, `max_step` bei Nullrichtung.
- **Verdikte** (`tests/test_verdicts.py`, Werte von Hand gesetzt), **Claims** (`tests/test_claims.py`: jede Zahl dieser README), **Oberfläche** (AppTest: jeder Button, jeder Regler an Minimum und Maximum, alle Regler zugleich, jede Form, jede Verzerrung κ, PDF), Presets (die Zahlen der Hilfetexte), Permalink, Layout, Abbildungen, Footer, Laufzeit-Importe.
- **Umfang:** 182 Tests (3 davon, die cvxpy-Orakel gegen Clarabel, überspringen sich ohne cvxpy; mit cvxpy laufen alle 7 Orakel-Tests grün), rund 1,5 Minuten (2026-11-03, Windows).
- **Nicht gelaufen:** kein Mutationstest (seit 2026-10-28 nicht mehr Teil des Ablaufs). cvxpy und scipy sind Gegenproben im dev-Satz; ohne sie überspringen sich die Orakel-Tests. Linux-CI: noch nicht gelaufen (Repo noch nicht veröffentlicht).

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App (Skelett nach DEMO-PLAYBOOK §2) |
| `kegel_kern.py` | Gemeinsamer Kern der Reihe, wortgleich mit Stück 2, 3, 4 und 5 |
| `kpd_cones.py` | Jordan-Algebra, Schrittweite, Nesterov–Todd-Skalierung je Kegel |
| `kpd_solver.py` | Primal-dualer Löser (Mehrotra, dichtes KKT-System), drei Richtungen, Farkas-Strahl |
| `kpd_examples.py` | Ausreißer, Fälle ohne Slater-Punkt, Zufallsprogramme, Kegelautomorphismen |
| `kpd_forms.py` | Die sechs Formen als Kegelprogramm samt Rückumformung (aus Stück 5 übernommen) |
| `kpd_constants.py` | Regler-Grenzen, feste Annahmen, Farben, Presets |
| `kpd_evaluation.py` | Analyse, Experimente, Verdikte, Formatierung |
| `kpd_layout.py` | Fester Kopf (Stylesheet) |
| `kpd_pdf_export.py` | PDF-Export (fpdf2) |
| `kpd_presets.py` | Permalink, Presets (Schnellstart, aktives Preset hervorgehoben), Zufalls-Seed-Button |
| `kpd_scenario.py` | Punktwolken (gestreckt, gedreht), Handinstanz |
| `kpd_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `requirements-dev.txt` | Laufzeit plus pytest, scipy und cvxpy (Gegenproben) |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt) |
| `tests/` | Kegelalgebra, Löser, Beispiele, Orakel, Randfälle, Verdikte, Claims, Presets, Layout, Abbildungen, AppTest, Footer, Laufzeit-Importe |

## Bewusst nicht umgesetzt

- **Homogene selbstduale Einbettung und Phase I:** nicht gebaut; die Demo zeigt das Verhalten ohne sie (Experiment 5).
- **Äquilibrierung der Daten und Rückfall auf Zentrierung:** gemessen, bringt nichts bzw. schadet; nicht im Löser.
- **Dünne Zerlegung des KKT-Systems:** verlöre die Lesbarkeit; die Größe der App ist deshalb begrenzt.
- **Exponential- und Potenzkegel:** Stück 7. **Freier Programmeditor:** die Eingabe sind die Formen aus Stück 5 mit ihren Reglern.
- **Konvergenzbeweise** des primal-dualen Verfahrens: nur als Hintergrund genannt.
- **Mutationstest:** nicht Teil des Ablaufs.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

Streamlit Cloud: Python-Version in den App-Einstellungen prüfen (lokal Python 3.14, CI Python 3.12); der Code verwendet nur Merkmale, die in Python 3.12 laufen. <!-- lint-ok: dezimalpunkt -->

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy, pandas, fpdf2 und threadpoolctl.

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Konvexe Optimierung: vom Beweis zur Verfahrenswahl](https://sebastianhanisch.net/konzepte-konvexe-optimierung.html).
