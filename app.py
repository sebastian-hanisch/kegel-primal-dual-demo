"""Primal-dualer Schritt im Kegel – Jordan-Produkt, Skalierung und ein Zertifikat - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Stück 6 der Reihe „Konvexe und konische Optimierung“: Der Barrierenkern aus Stück 5 zentriert und nimmt an, dass `ν/t` die Lücke ist. Ein primal-dualer Löser läuft mit einem echten Dualpunkt mit:
seine Lücke cᵀx + bᵀz ist ein Zertifikat. Die Demo zeigt, warum die LP-Formel s_i z_i = μ im Lorentz- und im PSD-Kegel das Jordan-Produkt braucht, was die Nesterov–Todd-Skalierung dazu beiträgt
(Invarianz, nicht kürzere Läufe), wie viele Iterationen der Löser braucht und was er bei Programmen ohne Slater-Punkt und beim Fall aus Stück 3 meldet.
Aufbau und Regeln: DEMO-PLAYBOOK.md (§2).

Lauffähig mit: streamlit run app.py
"""

import pandas as pd
import streamlit as st

import kpd_constants as C
import kpd_forms as A
import kpd_solver as S
from kpd_evaluation import (
    Settings,
    analyse,
    certificate_experiment,
    commutator_rows,
    de,
    dim_rows,
    hand_commutator,
    invariance_rows,
    nu_rows,
    num,
    random_rows,
    scaling_identities,
    sci,
    slater_rows,
    status_short,
    status_text,
    variant_rows,
    verdict_certificate,
    verdict_commutator,
    verdict_identities,
    verdict_invariance,
    verdict_nu,
    verdict_random,
    verdict_run,
    verdict_slater,
    verdict_variants,
)
from kpd_layout import dunkles_design, kopf_css
from kpd_pdf_export import build_pdf
from kpd_presets import bounds, init_session_state_defaults, load_permalink_settings, randomize_seed, sidebar_presets, sync_query_params
from kpd_visualization import build_certificate, build_commutator, build_convergence, build_invariance, build_nu, build_variants

st.set_page_config(page_title="Primal-dualer Schritt im Kegel – Sebastian Hanisch", layout="wide")


def _cloud_key(settings):
    """Ein Experiment, das nur von der Wolke abhängt, rechnet nicht neu, wenn Form oder κ wechseln."""
    return Settings(C.DEFAULT_FORM, settings.n, settings.stretch, settings.angle, settings.seed, C.DEFAULT_KAPPA)


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return analyse(settings)


@st.cache_data(show_spinner=False)
def _variants(key):
    return variant_rows(key)


@st.cache_data(show_spinner=False)
def _random():
    return random_rows(C.RANDOM_PROGRAMS)


@st.cache_data(show_spinner=False)
def _commutator(settings):
    return commutator_rows(Settings(settings.form, settings.n, settings.stretch, settings.angle, settings.seed, C.DEFAULT_KAPPA))


@st.cache_data(show_spinner=False)
def _identities():
    return scaling_identities()


@st.cache_data(show_spinner=False)
def _invariance():
    return invariance_rows()


@st.cache_data(show_spinner=False)
def _sweeps():
    return nu_rows(), dim_rows()


@st.cache_data(show_spinner=False)
def _certificate():
    return certificate_experiment()


@st.cache_data(show_spinner=False)
def _slater():
    return slater_rows()


def verdict(pair):
    kind, text = pair
    getattr(st, kind)(text)


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in kpd_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🧮 Primal-dualer Schritt im Kegel – Jordan-Produkt, Skalierung und ein Zertifikat")
st.markdown(
    """
**Die Barriere aus Stück 5 nimmt an, dass `ν/t` die Lücke ist; der primal-duale Löser rechnet den Dualpunkt mit.** Er löst `min cᵀx` mit `s = b − A x ∈ K` und `Aᵀz = −c`, `z ∈ K` zugleich; die Lücke `cᵀx + bᵀz` ist ein Zertifikat aus einem echten Dualpunkt.
Dafür muss die LP-Formel `s_i z_i = μ` in die Kegel übertragen werden, und das geht nicht wörtlich: im Orthant ist das Produkt komponentenweise, im Lorentz- und im PSD-Kegel braucht es das **Jordan-Produkt**
(`(UV + VU)/2` für Matrizen, deren Produkt nicht symmetrisch ist). Die **Nesterov–Todd-Skalierung** macht den Schritt invariant unter den Automorphismen des Kegels. Die Demo misst, was jeder dieser drei Schritte leistet.
Stück 6 der Reihe „Konvexe und konische Optimierung“.

Weiter unten: **fünf Experimente** (**🔬**), **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert der Schritt", expanded=True):
    st.markdown(
        """
1. **Pfad statt Zentrierung:** Statt für festes `t` zu zentrieren, folgt der Löser Primal und Dual zugleich; je Iteration ein lineares System (KKT) für `Δx`, `Δz`, `Δs`.
2. **Komplementarität:** Ziel ist `s∘z = μe` mit wachsendem Abstand zum Rand (`μ → 0`). Im Orthant ist `∘` komponentenweise; im Lorentz-Kegel `(uᵀv, u₀v̄ + v₀ū)`; im PSD-Kegel `(UV + VU)/2`.
3. **Drei Richtungen:** (a) die LP-Formel wörtlich auf allen Koordinaten, (b) das Jordan-Produkt des Kegels (ohne Skalierung), (c) die Nesterov–Todd-Skalierung `W` mit `W z = W⁻¹ s`: die Komplementarität wird im skalierten Raum linearisiert.
4. **Vorhersage und Korrektur (Mehrotra):** erst eine Richtung zum Ziel `μ = 0`, daraus `σ = (1 − α)³` und ein Korrekturterm, dann der eigentliche Schritt mit 99 % des Wegs zum Rand.
5. **Start ohne Zulässigkeit:** `x` aus den kleinsten Quadraten, `s` und `z` in den Kegel verschoben; es braucht keinen streng zulässigen Startpunkt.
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Form**")
    form = st.selectbox("Gesuchte Form", A.FORMS, key="form_select", format_func=A.FORM_NAMES.get,
                        help="Für welche der sechs Formen das Hauptdiagramm und Experiment 2 gelten; die Tabelle in Experiment 1 vergleicht immer alle sechs.")
    st.markdown("**Punktwolke**")
    n = st.slider("Punkte n", *bounds("n_slider"), key="n_slider", help="Wie viele Punkte die Wolke hat.")
    stretch = st.slider("Streckung", *bounds("stretch_slider"), step=0.1, key="stretch_slider", help="Wie viel breiter die Wolke entlang ihrer Längsachse ist als quer dazu.")
    angle = st.slider("Drehung (Grad)", *bounds("angle_slider"), key="angle_slider", help="Winkel der Längsachse gegen die x-Achse.")
    seed = st.number_input("Zufalls-Seed der Wolke", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Wolke generieren", width="stretch", on_click=randomize_seed)
    st.markdown("**Experiment 3**")
    kappa = st.select_slider("Verzerrung κ des Kegelautomorphismus", options=C.KAPPAS, key="kappa_select", format_func=lambda k: sci(float(k), 0) if k > 1 else "1",
                             help="Wie stark das Zufallsprogramm in Experiment 3 mit einem Automorphismus des Kegels verzerrt wird (Kondition κ).")

sync_query_params({"form_select": form, "n_slider": int(n), "stretch_slider": float(stretch), "angle_slider": int(angle), "seed_input": int(seed), "kappa_select": int(kappa)})

settings = Settings(form, int(n), float(stretch), int(angle), int(seed), int(kappa))
a = _analysis(settings)
nt, jo, ve = a.runs["nt"], a.runs["jordan"], a.runs["vector"]

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
m1, m2, m3 = st.columns(3)
m4, m5 = st.columns(2)
m1.metric("NT-Schritt (Iterationen)", f"{nt.iterations}", help=f"Ausgang: {status_text(nt.status)}.")
m2.metric("Jordan-Schritt (Iterationen)", f"{jo.iterations}", help=f"Ausgang: {status_text(jo.status)}.")
m3.metric("LP-Formel wörtlich", status_short(ve.status), help=f"Die LP-Formel s_i z_i = μ auf allen Koordinaten, ohne Kegelstruktur. Ausgang: {status_text(ve.status)} nach {ve.iterations} Iterationen.")
m4.metric("Barriere (Stück 5): Newton-Schritte", f"{a.barrier.newton_steps}", help=f"Die Barrierenmethode aus Stück 5 auf derselben Wolke, in {a.barrier.rounds} Runden.")
m5.metric("Zertifikat cᵀx + bᵀz", sci(abs(nt.certificate_gap)), help="Der Abstand von Primal und Dual des NT-Laufs, aus einem echten Dualpunkt.")
st.plotly_chart(build_convergence(a.runs), width="stretch", key="main_chart")
st.caption("Lücke sᵀz je Iteration der drei Richtungen. Ein rotes Kreuz markiert einen Lauf, der nicht gelöst endet. Im Orthant (Würfel, Raute) sind alle drei Richtungen dieselbe.")

if st.button("📄 PDF der aktuellen Wolke erstellen", key="pdf_make"):
    st.session_state["pdf_made"] = (settings, build_pdf(a))
made = st.session_state.get("pdf_made")
if made and made[0] == settings:
    st.download_button("⬇️ PDF herunterladen", data=made[1], file_name="kegel-primal-dual.pdf", mime="application/pdf", key="pdf_download")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Was leistet der primal-duale Schritt im Kegel?")
verdict(verdict_run(a))

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Der LP-Schritt, Kegel für Kegel")
st.caption("Alle sechs Formen derselben Wolke mit den drei Richtungen; darunter Zufallsprogramme (streng zulässig durch Konstruktion).")
vrows = _variants(_cloud_key(settings))
st.plotly_chart(build_variants(vrows), width="stretch", key="variants_chart")
st.dataframe(pd.DataFrame([{"Form": A.FORM_NAMES[r["form"]], "Richtung": C.SCALING_NAMES[r["scaling"]], "Ausgang": status_text(r["status"]), "Iterationen": r["iterations"],
                            "Abweichung vom Referenzwert": sci(r["error"]), "relativ (Maß der Toleranz)": sci(r["relative_error"]), "Lücke cᵀx + bᵀz": sci(abs(r["certificate_gap"]))} for r in vrows]), hide_index=True, width="stretch")
verdict(verdict_variants(vrows))
rrows = _random()
st.dataframe(pd.DataFrame([{"Kegelmischung": r["mix"], "Richtung": C.SCALING_NAMES[r["scaling"]], "gelöst": f"{r['solved']} von {r['count']}", "Iterationen (Median)": de(r["median"], 1) if r["median"] is not None else "–",
                            "größte Zahl": str(r["max"]) if r["max"] is not None else "–"} for r in rrows]), hide_index=True, width="stretch")
verdict(verdict_random(rrows))

st.markdown("---")

st.subheader("🔬 Experiment 2: Das Matrixprodukt ist nicht symmetrisch")
st.caption("Im PSD-Kegel ist `S Z` im Allgemeinen nicht symmetrisch und `Z S ≠ S Z`; das Jordan-Produkt nimmt den symmetrischen Anteil.")
hand = hand_commutator()
h1, h2, h3 = st.columns(3)
h1.markdown(f"**S Z**\n\n`{hand['SZ'].tolist()}`")
h2.markdown(f"**Z S**\n\n`{hand['ZS'].tolist()}`")
h3.markdown(f"**S ∘ Z = (SZ + ZS)/2**\n\n`{hand['sym'].tolist()}`")
comm = _commutator(settings)
st.plotly_chart(build_commutator(comm), width="stretch", key="comm_chart")
verdict(verdict_commutator(comm, hand))

st.markdown("---")

st.subheader("🔬 Experiment 3: Skalierung und Invarianz")
st.caption("Erst die Identitäten der Nesterov–Todd-Skalierung an gezogenen Punkten, dann ein Zufallsprogramm mit zwei PSD-Blöcken, mit einem Kegelautomorphismus der Kondition κ umgeschrieben (der Start wird mitgenommen).")
irows = _identities()
st.dataframe(pd.DataFrame([{"Kegel": r["cone"], "Q z − s": sci(r["Qz_s"]), "W z − λ": sci(r["Wz_lam"]), "W⁻¹ s − λ": sci(r["Winv_s_lam"]), "Asymmetrie von W": sci(r["asym"]), "kleinster Eigenwert von W": de(r["min_eig"], 3)}
                           for r in irows]), hide_index=True, width="stretch")
verdict(verdict_identities(irows))
inv = _invariance()
st.plotly_chart(build_invariance(inv), width="stretch", key="inv_chart")
verdict(verdict_invariance(inv, int(kappa)))

st.markdown("---")

st.subheader("🔬 Experiment 4: Iterationen gegen ν und Dimension")
st.caption(f"Kugel (Lorentz-Form) auf {C.N_SWEEP[0]} bis {C.N_SWEEP[-1]} Punkten und allgemeine Ellipse in d = {C.D_SWEEP[0]} bis {C.D_SWEEP[-1]} Dimensionen (Median über {len(C.SWEEP_SEEDS)} feste Wolken), gegen die Newton-Schritte der Barriere.")
nrows, drows = _sweeps()
st.plotly_chart(build_nu(nrows, drows), width="stretch", key="nu_chart")
st.dataframe(pd.DataFrame([{"Kugel: Punkte": r["n"], "ν": r["nu"], "Zeilen": r["rows"], "NT": r["nt"], "Jordan": r["jordan"], "Barriere: Newton": r["newton"], "Barriere: Runden": r["rounds"]} for r in nrows]),
             hide_index=True, width="stretch")
st.dataframe(pd.DataFrame([{"Ellipse: d": r["d"], "Punkte": r["points"], "Variablen": r["variables"], "PSD-Größe": r["psd_size"], "ν": r["nu"], "NT": r["nt"], "Jordan": r["jordan"], "Barriere: Newton": r["newton"]}
                           for r in drows]), hide_index=True, width="stretch")
verdict(verdict_nu(nrows, drows))

st.markdown("---")

st.subheader("🔬 Experiment 5: Das Zertifikat und der Start")
st.caption("Der Fall aus Stück 3, in dem die Barriere ohne Absicherung einen falschen Wert mit winziger Lücke meldet, die Programme ohne Slater-Punkt aus Stück 4 und ein unzulässiges LP.")
cert = _certificate()
st.plotly_chart(build_certificate(cert), width="stretch", key="cert_chart")
st.caption("Nach dem ersten vollen Schritt (Schrittlänge 1) liegen beide Residuen auf Rundungsniveau; Werte darunter zeichnet die Achse bei 10⁻¹⁶ (Untergrenze, kein gemessener Wert).")
verdict(verdict_certificate(cert))
srows = _slater()
st.dataframe(pd.DataFrame([{"Programm": r["label"], "Ausgang": status_text(r["result"].status), "Iterationen": r["result"].iterations, "Primal": num(r["result"].value), "Dual": num(r["result"].dual_value),
                            "‖x‖": sci(r["x_norm"]), "‖z‖": sci(r["z_norm"]), "Bemerkung": r["note"]} for r in srows]), hide_index=True, width="stretch")
verdict(verdict_slater(srows))

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Der Jordan-Schritt „bricht“ im PSD-Kegel** | Nicht auf diesen Programmen: er löst alle Zufallsprogramme und braucht in den Mischungen mit Lorentz- oder PSD-Kegel weniger Iterationen als der skalierte Schritt (im Orthant gleich viele). Ihm fehlen Invarianz und Theorie; bei stark verzerrten Programmen gelingen aber auch seine Läufe nur zum Teil (Experiment 3: 2 von 10 bei κ = 10⁶). | – |
| **Die Skalierung macht den Löser robuster** | Sie hält die Iterationszahl bis κ = 10⁴ gleich, nicht die Erfolgsquote: bei κ = 10⁴ scheiterten in der Stichprobe von 10 Programmen 6 NT-Läufe gegen 1 mit Jordan (bei κ = 10⁶ gelingen nur 1 gegen 2 von 10, bei κ = 100 NT 10 gegen Jordan 9; Experiment 3). | – |
| **Die achsenparallele Ellipse** | `t ≥ 1/p` mit `t ≈ 80` und Dualwerten bis 3·10³ endet an der Rechengenauigkeit; Primal und Dual stimmen absolut auf 10⁻⁵ (NT, nicht abgeschlossen) bis 10⁻⁷ (Jordan, gelöst); die Toleranz 10⁻⁸ gilt relativ zu max(1, Betrag des Werts), und bei einem Wert um 134 sind 4,8·10⁻⁷ relativ 3,6·10⁻⁹. Ruiz-Äquilibrierung und ein Rückfall auf Zentrierung helfen nicht (gemessen, nicht eingebaut). | – |
| **Ein Zertifikat liegt vor, wenn der Lauf endet** | Bei Programmen ohne Slater-Punkt und unzulässigen Programmen läuft ein Iterat davon; das Lückenbeispiel hat den Wert −1, das schwach unzulässige Programm keinen Abschluss und kein Zertifikat. Eine homogene Einbettung würde das sauber melden (nicht gebaut). | – |
| **Eine Iteration kostet wenig** | Ein dichtes KKT-System der Größe Zeilen + Variablen je Iteration; der Vergleich mit der Barriere zählt Iterationen, keine Zeit. | – |
| **Der Exponentialkegel** | Er ist nicht selbstdual; die Skalierung gilt nicht. | Stück 7 |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Kegelprogramm.** Primal $\min c^\top x$ mit $s = b - Ax \in K$; Dual $\max -b^\top z$ mit $A^\top z = -c$, $z \in K$. Für zulässige Punkte ist $c^\top x + b^\top z = z^\top s \ge 0$ (Stück 4); die Lücke ist null genau bei Komplementarität $s \circ z = 0$.

**Jordan-Produkt und Einselement.** Orthant: $u \circ v = (u_i v_i)_i$, $e = \mathbf{1}$. Lorentz-Kegel: $u \circ v = (u^\top v,\ u_0 \bar v + v_0 \bar u)$, $e = (1, 0, \dots, 0)$. PSD-Kegel: $U \circ V = \tfrac12 (UV + VU)$, $e = I$.
Zentraler Pfad: $s \circ z = \mu e$; dort ist $s^\top z = \mu\, e^\top e$ ($e^\top e$ = Zeilen im Orthant, 1 je Lorentz-Kegel, $k$ im $k \times k$-PSD-Kegel).

**Der Schritt ohne Skalierung (AHO-Richtung).** Linearisierung $L(z)\Delta s + L(s)\Delta z = \sigma\mu e - s \circ z$ mit $L(u)v = u \circ v$; zusammen mit $A\Delta x + \Delta s = -r_p$, $A^\top \Delta z = -r_d$ ein KKT-System
$\begin{pmatrix} 0 & A^\top \\ A & -L(z)^{-1}L(s) \end{pmatrix}$. Die Matrix $L(z)^{-1}L(s)$ ist im Allgemeinen nicht symmetrisch.

**NT-Skalierung.** Es gibt genau ein symmetrisch positiv definites $W$ mit $W z = W^{-1} s =: \lambda$ und $W K = K$. Orthant: $W = \operatorname{diag}\sqrt{s/z}$. PSD: $W = S^{1/2}(S^{1/2} Z S^{1/2})^{-1/2} S^{1/2}$ (also $WZW = S$), wirkend als $X \mapsto W^{1/2} X W^{1/2}$.
Lorentz: $w = (\hat s + J\hat z)/\sqrt{2(1 + \hat s^\top \hat z)}$ mit $\hat s = s/\sqrt{s^\top J s}$, $J = \operatorname{diag}(1, -I)$, und $W^2 = \beta^2 (2 w w^\top - J)$, $\beta^2 = \sqrt{s^\top J s / z^\top J z}$.
Im skalierten Raum lautet die Gleichung $\lambda \circ (W\Delta z + W^{-1}\Delta s) = \sigma\mu e - \lambda \circ \lambda - \Delta\tilde s_{\mathrm{aff}} \circ \Delta\tilde z_{\mathrm{aff}}$ mit $\Delta\tilde s = W^{-1}\Delta s$, $\Delta\tilde z = W \Delta z$ (Mehrotra-Korrektor); das KKT-System ist
$\begin{pmatrix} 0 & A^\top \\ A & -W^2 \end{pmatrix}$, symmetrisch. Die Richtung ist in exakter Rechnung invariant unter den Automorphismen des Kegels (ein Automorphismus $\mathcal{T}$ ersetzt $A, b$ durch $\mathcal{T}A, \mathcal{T}b$ und $z$ durch $\mathcal{T}^{-\top}z$).

**Schrittweite.** $\alpha_{\max}$ aus der Bedingung $\lambda + \alpha\,\Delta\tilde s \in K$ und $\lambda + \alpha\,\Delta\tilde z \in K$ (Orthant: Minimum über negative Komponenten; Lorentz: kleinste Nullstelle einer quadratischen Gleichung; PSD: kleinster Eigenwert von $\Lambda^{-1/2}\Delta\Lambda\,\Lambda^{-1/2}$); Schritt $0{,}99\,\alpha_{\max}$, höchstens 1; $\sigma = (1 - \alpha_{\mathrm{aff}})^3$.

**Farkas-Strahl.** Ist $Ax + s = b$, $s \in K$ unzulässig, gibt es $z \in K$ mit $A^\top z = 0$ und $b^\top z < 0$; ein davonlaufender Dualpunkt, normiert, nähert diesen Strahl.

**Literatur.** Y. E. Nesterov und M. J. Todd: *Self-Scaled Barriers and Interior-Point Methods for Convex Programming*, Mathematics of Operations Research 22 (1), 1997, S. 1–42 (die Skalierung, selbstskalierte Barrieren).
F. Alizadeh, J.-P. A. Haeberly und M. L. Overton: *Primal-Dual Interior-Point Methods for Semidefinite Programming: Convergence Rates, Stability and Numerical Results*, SIAM Journal on Optimization 8 (3), 1998, S. 746–768 (die Richtung ohne Skalierung).
L. Vandenberghe: *The CVXOPT linear and quadratic cone program solvers*, Bericht zur Dokumentation von CVXOPT (cvxopt.org/documentation): Formeln der Skalierung, des Korrektors und des Starts, denen die Einstellungen dieser Demo folgen.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Konvexe Optimierung: vom Beweis zur Verfahrenswahl](https://sebastianhanisch.net/konzepte-konvexe-optimierung.html)."
)
