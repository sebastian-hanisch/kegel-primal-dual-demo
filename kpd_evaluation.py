"""Kennzahlen, Experimente und Verdikte. Jedes Verdikt ist eine reine Funktion (Art, Text); die App ruft st.<Art>(Text)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import median

import numpy as np

import kegel_kern as K
import kpd_cones as CO
import kpd_constants as C
import kpd_examples as E
import kpd_forms as A
import kpd_solver as S
from kpd_scenario import Cloud, make_cloud


@dataclass(frozen=True)
class Settings:
    form: str = C.DEFAULT_FORM
    n: int = C.DEFAULT_N
    stretch: float = C.DEFAULT_STRETCH
    angle: int = C.DEFAULT_ANGLE
    seed: int = C.DEFAULT_SEED
    kappa: int = C.DEFAULT_KAPPA


@dataclass(frozen=True, eq=False)
class Analysis:
    settings: Settings
    cloud: Cloud
    problem: K.ConeProblem
    reference: float            # Barrierenmethode mit Lücke 1e-10: Vergleichswert für die Abweichung
    barrier: K.BarrierResult    # Barrierenmethode bis 1e-9 (Stück 5)
    runs: dict                  # Skalierung -> PDResult der gewählten Form

    @property
    def nt(self) -> S.PDResult:
        return self.runs["nt"]


# ----------------------------------------------------------------------------- Formatierung

def de(x: float, digits: int = 2) -> str:
    """Zahl mit deutschem Dezimalkomma und typografischem Minuszeichen."""
    return f"{x:.{digits}f}".replace(".", ",").replace("-", "−")


_SUPER = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def sci(x: float, digits: int = 1) -> str:
    """Zahl in Zehnerpotenz-Schreibweise, z. B. 2,0·10⁻¹⁰."""
    if x == 0.0 or not math.isfinite(x):
        return de(x, digits) if math.isfinite(x) else "unendlich"
    e = int(math.floor(math.log10(abs(x))))
    m = x / 10.0 ** e
    if round(abs(m), digits) >= 10.0:
        m, e = m / 10.0, e + 1
    return f"{de(m, digits)}·10{str(e).translate(_SUPER)}"


def num(x: float) -> str:
    """Wert für Texte: Dezimalkomma, kleine und große Beträge in Zehnerpotenz."""
    return sci(x, 2) if x != 0.0 and (abs(x) < 1e-3 or abs(x) >= 1e5) else de(x, 4 if abs(x) < 10 else 3)


CONE_NAMES = {"l": "Orthant", "q": "Lorentz", "s": "PSD"}


def status_text(status: str) -> str:
    return C.STATUS_NAMES.get(status, status)


def status_short(status: str) -> str:
    return C.STATUS_SHORT.get(status, status)


# ----------------------------------------------------------------------------- Analyse der eingestellten Wolke

def run_all(problem: K.ConeProblem, record: bool = False) -> dict:
    return {sc: S.solve_pd(problem, tol=C.TOL, max_iter=C.MAX_ITER, scaling=sc, record=record and sc == "nt") for sc in S.SCALINGS}


def analyse(s: Settings) -> Analysis:
    cloud = make_cloud(s.n, s.stretch, s.angle, s.seed)
    prob = A.build_form(s.form, cloud.points)
    return Analysis(s, cloud, prob, K.solve_barrier(prob, tol=1e-10).value, K.solve_barrier(prob, tol=1e-9), run_all(prob))


def deviation(a: Analysis, result: S.PDResult) -> float:
    return abs(result.value - a.reference)


def relative(error: float, reference: float) -> float:
    """Abweichung im Maß des Abbruchkriteriums: |Abweichung| / max(1, |Referenzwert|). Der Löser bricht relativ ab (Lücke sᵀz ≤ tol · max(1, |cᵀx|)),
    bei einem Wert von 134 sind 4,8·10⁻⁷ also 3,6·10⁻⁹ und unter der Toleranz 10⁻⁸."""
    return error / max(1.0, abs(reference))


# ----------------------------------------------------------------------------- Experiment 1: der LP-Schritt, Kegel für Kegel

def variant_rows(s: Settings) -> list:
    """Alle sechs Formen derselben Wolke mit den drei Richtungen: Ausgang, Iterationen, Abweichung vom Referenzwert."""
    cloud = make_cloud(s.n, s.stretch, s.angle, s.seed)
    rows = []
    for form in A.FORMS:
        prob = A.build_form(form, cloud.points)
        ref = K.solve_barrier(prob, tol=1e-10).value
        for sc in S.SCALINGS:
            r = S.solve_pd(prob, tol=C.TOL, max_iter=C.MAX_ITER, scaling=sc)
            rows.append({"form": form, "scaling": sc, "status": r.status, "iterations": r.iterations, "error": abs(r.value - ref), "relative_error": relative(abs(r.value - ref), ref), "certificate_gap": r.certificate_gap,
                         "cones": "+".join(sorted({k for k, _ in prob.cones}))})
    return rows


def random_rows(count: int = 40, mixes=E.MIXES, seed0: int = C.DEFAULT_SOLVER_SEED) -> list:
    """Zufallsprogramme je Kegelmischung: Anteil gelöster Läufe und Iterationen (Median) je Richtung."""
    rows = []
    for name, (cones, nvar) in mixes.items():
        for sc in S.SCALINGS:
            res = [S.solve_pd(E.random_program(cones, nvar, seed0 + k), tol=C.TOL, max_iter=C.MAX_ITER, scaling=sc) for k in range(count)]
            ok = [r.iterations for r in res if r.status == S.OPTIMAL]
            rows.append({"mix": name, "scaling": sc, "solved": len(ok), "count": count, "median": float(median(ok)) if ok else None, "max": max(ok) if ok else None})
    return rows


# ----------------------------------------------------------------------------- Experiment 2: das Matrixprodukt ist nicht symmetrisch

def hand_commutator() -> dict:
    """Von Hand: S = diag(2, 1), Z = [[1, 1], [1, 2]] (beide positiv definit): SZ = [[2, 2], [1, 2]], ZS = [[2, 1], [2, 2]]; der symmetrische Anteil von SZ ist [[2, 1,5], [1,5, 2]]."""
    Sm, Z = np.diag([2.0, 1.0]), np.array([[1.0, 1.0], [1.0, 2.0]])
    P = Sm @ Z
    return {"S": Sm, "Z": Z, "SZ": P, "ZS": Z @ Sm, "sym": (P + P.T) / 2.0, "antisym_norm": float(np.linalg.norm((P - P.T) / 2.0)), "SZ_norm": float(np.linalg.norm(P)),
            "ratio": float(np.linalg.norm(Sm @ Z - Z @ Sm) / (np.linalg.norm(Sm) * np.linalg.norm(Z)))}


def commutator_rows(s: Settings) -> dict:
    """‖SZ − ZS‖ / (‖S‖ ‖Z‖) je Iteration des NT-Laufs im ersten PSD-Block; hat die gewählte Form keinen, gilt ersatzweise die PSD-Kugel derselben Wolke."""
    form = s.form
    prob = A.build_form(form, make_cloud(s.n, s.stretch, s.angle, s.seed).points)
    psd = [b for b in CO.blocks(prob.cones) if b[0] == "s"]
    if not psd:
        form = "ball_psd"
        prob = A.build_form(form, make_cloud(s.n, s.stretch, s.angle, s.seed).points)
        psd = [b for b in CO.blocks(prob.cones) if b[0] == "s"]
    kind, m, off, d = psd[0]
    r = S.solve_pd(prob, tol=C.TOL, max_iter=C.MAX_ITER, scaling="nt", record=True)
    rows = []
    for it, (sv, zv) in enumerate(r.iterates):
        Sm, Z = K.smat(sv[off:off + d], m), K.smat(zv[off:off + d], m)
        rows.append({"it": it, "ratio": float(np.linalg.norm(Sm @ Z - Z @ Sm) / (np.linalg.norm(Sm) * np.linalg.norm(Z))), "gap": r.history[it]["gap"]})
    return {"form": form, "substitute": form != s.form, "rows": rows}


# ----------------------------------------------------------------------------- Experiment 3: Skalierung und Invarianz

def scaling_identities(points: int = C.IDENTITY_POINTS, seed: int = 1) -> list:
    """Die Identitäten der NT-Skalierung an gezogenen Punkten: größte Abweichung je Kegel (Q z = s, W z = λ, W⁻¹ s = λ, W symmetrisch) und kleinster Eigenwert von W."""
    rng = np.random.default_rng(seed)
    rows = []
    for kind, m in (("l", 4), ("q", 3), ("q", 6), ("s", 2), ("s", 3), ("s", 5)):
        e1 = e2 = e3 = e4 = 0.0
        smallest = math.inf
        for _ in range(points):
            s, z = E.interior(kind, m, rng), E.interior(kind, m, rng)
            Q, W, lam = CO.nt_scaling(kind, m, s, z)
            e1 = max(e1, float(np.linalg.norm(Q @ z - s) / np.linalg.norm(s)))
            e2 = max(e2, float(np.linalg.norm(W @ z - lam) / np.linalg.norm(lam)))
            e3 = max(e3, float(np.linalg.norm(np.linalg.solve(W, s) - lam) / np.linalg.norm(lam)))
            e4 = max(e4, float(np.linalg.norm(W - W.T) / np.linalg.norm(W)))
            smallest = min(smallest, float(np.linalg.eigvalsh((W + W.T) / 2.0).min()))
        rows.append({"cone": f"{CONE_NAMES[kind]} {m}", "Qz_s": e1, "Wz_lam": e2, "Winv_s_lam": e3, "asym": e4, "min_eig": smallest})
    return rows


def invariance_rows(kappas=C.KAPPAS, count: int = C.INVARIANCE_PROGRAMS, seed0: int = 2000) -> list:
    """Zufallsprogramme mit zwei PSD-Blöcken 3×3, mit einem Kegelautomorphismus der Kondition κ umgeschrieben (Start mitgenommen): Läufe, die gelingen, und deren Iterationen."""
    cones, nvar = (("s", 3),) * 2, 6
    rows = []
    for kappa in kappas:
        for sc in ("nt", "jordan"):
            res = []
            for k in range(count):
                base = E.random_program(cones, nvar, seed0 + k)
                prob, start = E.automorphism(base, kappa, k) if kappa > 1 else (base, None)
                res.append(S.solve_pd(prob, tol=C.TOL, max_iter=C.MAX_ITER, scaling=sc, start=start))
            ok = [r.iterations for r in res if r.status == S.OPTIMAL]
            rows.append({"kappa": kappa, "scaling": sc, "solved": len(ok), "count": count, "median": float(median(ok)) if ok else None, "max": max(ok) if ok else None})
    return rows


# ----------------------------------------------------------------------------- Experiment 4: Iterationen gegen ν und Dimension

def nu_rows(ns=C.N_SWEEP, seeds=C.SWEEP_SEEDS) -> list:
    """Kugel (Lorentz-Form) auf wachsender Punktzahl: ν = 2n, Zeilen 3n, Variablen 3."""
    rows = []
    for n in ns:
        its = {"nt": [], "jordan": []}
        stat = {"nt": [], "jordan": []}
        newton, rounds = [], []
        for seed in seeds:
            prob = A.build_form("ball_soc", make_cloud(n, 3.0, 30, seed).points)
            for sc in ("nt", "jordan"):
                r = S.solve_pd(prob, tol=C.TOL, max_iter=C.MAX_ITER, scaling=sc)
                its[sc].append(r.iterations)
                stat[sc].append(r.status)
            b = K.solve_barrier(prob, tol=1e-9)
            newton.append(b.newton_steps)
            rounds.append(b.rounds)
        rows.append({"n": n, "nu": K.barrier_parameter(prob.cones), "rows": prob.A.shape[0], "variables": prob.A.shape[1], "nt": float(median(its["nt"])), "jordan": float(median(its["jordan"])),
                     "solved_nt": sum(x == S.OPTIMAL for x in stat["nt"]), "solved_jordan": sum(x == S.OPTIMAL for x in stat["jordan"]), "seeds": len(seeds),
                     "newton": float(median(newton)), "rounds": float(median(rounds))})
    return rows


def dim_rows(ds=C.D_SWEEP, seeds=C.SWEEP_SEEDS) -> list:
    """Allgemeine Ellipse in wachsender Dimension d (3d + 4 Punkte): PSD-Block 2d."""
    rows = []
    for d in ds:
        its = {"nt": [], "jordan": []}
        stat = {"nt": [], "jordan": []}
        newton = []
        for seed in seeds:
            pts = np.random.default_rng(seed).normal(0.0, 1.0, (3 * d + 4, d))
            pts -= pts.mean(axis=0)
            prob = A.build_form("ellipse_psd", pts)
            for sc in ("nt", "jordan"):
                r = S.solve_pd(prob, tol=C.TOL, max_iter=150, scaling=sc)
                its[sc].append(r.iterations)
                stat[sc].append(r.status)
            newton.append(K.solve_barrier(prob, tol=1e-9).newton_steps)
        rows.append({"d": d, "points": 3 * d + 4, "variables": prob.A.shape[1], "rows": prob.A.shape[0], "psd_size": 2 * d, "nu": K.barrier_parameter(prob.cones), "nt": float(median(its["nt"])),
                     "jordan": float(median(its["jordan"])), "solved_nt": sum(x == S.OPTIMAL for x in stat["nt"]), "solved_jordan": sum(x == S.OPTIMAL for x in stat["jordan"]), "seeds": len(seeds),
                     "newton": float(median(newton))})
    return rows


# ----------------------------------------------------------------------------- Experiment 5: Zertifikat und Start

def certificate_experiment() -> dict:
    """Der Fall aus Stück 3, in dem der Barrierenkern ohne Absicherung 190,4 bei einer Lücke von 1,7·10⁻¹⁰ meldet (Optimum 1,0·10⁻⁶)."""
    prob = E.outlier_problem()
    bar = K.solve_barrier(prob, tol=1e-9)
    pd = S.solve_pd(prob, tol=1e-9, max_iter=C.MAX_ITER, scaling="nt")
    return {"barrier": bar, "pd": pd, "optimum": E.OUTLIER_OPTIMUM}


def slater_rows() -> list:
    """Die Fälle ohne Slater-Punkt aus Stück 4 und das unzulässige LP, mit dem primal-dualen Löser (NT)."""
    out = []

    def add(label, prob, note, max_iter=C.MAX_ITER):
        r = S.solve_pd(prob, tol=C.TOL, max_iter=max_iter, scaling="nt")
        out.append({"label": label, "result": r, "note": note, "x_norm": float(np.linalg.norm(r.x)), "z_norm": float(np.linalg.norm(r.z))})
        return r

    for eps in C.SLATER_EPS:
        add(f"Lückenbeispiel ε = {eps:g}", E.gapex_problem(eps), "Wert −1 für jedes ε > 0")
    add("[[x₁,1],[1,x₂]] ⪰ 0, Ziel x₁", E.nonattained_problem((1.0, 0.0)), "Infimum 0 wird nie angenommen: x₂ läuft davon")
    add("[[x₁,1],[1,x₂]] ⪰ 0, Ziel x₁ + x₂", E.nonattained_problem((1.0, 1.0)), "Optimum 2 bei (1, 1)")
    add("[[x,1],[1,0]] ⪰ 0 (schwach unzulässig)", E.weak_problem(), "kein Zertifikat: |x| wächst, die Residuen fallen nicht", max_iter=60)
    lp = E.lp_infeasible_problem()
    r = add("LP x ≥ 1 und x ≤ 0", lp, "unzulässig: z läuft auf den Farkas-Strahl", max_iter=60)
    zh, atz, btz = S.farkas_ray(lp, r)
    out[-1].update({"ray": zh, "ray_ATz": atz, "ray_bTz": btz})
    return out


# ----------------------------------------------------------------------------- Verdikte (reine Funktionen: (Art, Text))

def _row(rows, form, scaling):
    return next(r for r in rows if r["form"] == form and r["scaling"] == scaling)


def verdict_run(a: Analysis) -> tuple:
    nt, jo, ve = a.runs["nt"], a.runs["jordan"], a.runs["vector"]
    name = A.FORM_NAMES[a.settings.form]
    text = (f"Für die Form **{name}** braucht der NT-Schritt {nt.iterations} Iterationen ({status_text(nt.status)}), der Jordan-Schritt {jo.iterations} ({status_text(jo.status)}), "
            f"der wörtlich übertragene LP-Schritt endet nach {ve.iterations} ({status_text(ve.status)}). Die Barriere aus Stück 5 braucht {a.barrier.newton_steps} Newton-Schritte in {a.barrier.rounds} Runden.")
    if nt.status == S.OPTIMAL:
        text += (f" Das Zertifikat des NT-Laufs: Primal {num(nt.value)}, Dual {num(nt.dual_value)}, Lücke cᵀx + bᵀz = {sci(abs(nt.certificate_gap))}; "
                 f"Abweichung vom Referenzwert der Barriere {sci(deviation(a, nt))}, relativ {sci(relative(deviation(a, nt), a.reference))} (die Toleranz {sci(C.TOL, 0)} gilt relativ zu max(1, |Wert|)).")
        return "success", text
    return "warning", text + f" Der NT-Lauf ist nicht abgeschlossen (Lücke cᵀx + bᵀz = {sci(abs(nt.certificate_gap))}): das ist die Rechengrenze dieser Form, kein Wert, dem man trauen sollte."


def verdict_variants(rows: list) -> tuple:
    lp = [r for r in rows if r["form"] in ("cube", "diamond")]
    cone = [r for r in rows if r["form"] not in ("cube", "diamond")]
    same = all(r["iterations"] == _row(rows, r["form"], "nt")["iterations"] for r in lp)
    vec_fail = [r for r in cone if r["scaling"] == "vector" and r["status"] != S.OPTIMAL]
    jo_ok = [r for r in cone if r["scaling"] == "jordan" and r["status"] == S.OPTIMAL]
    nt_ok = [r for r in cone if r["scaling"] == "nt" and r["status"] == S.OPTIMAL]
    return "info", (f"Im Orthant (Würfel und Raute) sind alle drei Schritte derselbe: {'dieselben Iterationen' if same else 'verschiedene Iterationen'}. In den vier Formen mit Lorentz- oder PSD-Kegel "
                    f"scheitert der wörtlich übertragene LP-Schritt in **{len(vec_fail)} von {len(cone) // 3}**; der Jordan-Schritt löst {len(jo_ok)}, der NT-Schritt {len(nt_ok)} davon.")


def verdict_random(rows: list) -> tuple:
    parts = []
    for name in dict.fromkeys(r["mix"] for r in rows):
        by = {r["scaling"]: r for r in rows if r["mix"] == name}
        parts.append(f"{name}: wörtlich {by['vector']['solved']}/{by['vector']['count']}, Jordan {by['jordan']['solved']}/{by['jordan']['count']} (Median {de(by['jordan']['median'], 1) if by['jordan']['median'] is not None else '–'}), "
                     f"NT {by['nt']['solved']}/{by['nt']['count']} (Median {de(by['nt']['median'], 1) if by['nt']['median'] is not None else '–'})")
    return "info", "Zufallsprogramme (streng zulässig durch Konstruktion), gelöste Läufe von je " + str(rows[0]["count"]) + ": " + "; ".join(parts) + ". Der Jordan-Schritt braucht in den Mischungen mit Lorentz- oder PSD-Kegel **weniger** Iterationen als der skalierte (im Orthant gleich viele)."


def verdict_commutator(res: dict, hand: dict) -> tuple:
    peak = max(res["rows"], key=lambda r: r["ratio"])
    lead = "Die gewählte Form hat keinen PSD-Kegel; entlang des NT-Laufs der PSD-Kugel derselben Wolke" if res["substitute"] else f"Entlang des NT-Laufs der Form {A.FORM_NAMES[res['form']]}"
    return "info", (f"Von Hand: für S = diag(2, 1) und Z = [[1, 1], [1, 2]] ist S Z = [[2, 2], [1, 2]], aber Z S = [[2, 1], [2, 2]]; der antisymmetrische Anteil hat die Norm {de(hand['antisym_norm'], 3)} "
                    f"bei ‖S Z‖ = {de(hand['SZ_norm'], 3)}. {lead} wächst ‖S Z − Z S‖ / (‖S‖ ‖Z‖) von {sci(res['rows'][0]['ratio'])} am Start auf höchstens "
                    f"**{de(peak['ratio'], 2)}** (Iteration {peak['it']}, am ersten PSD-Block gemessen; gegen Ende fällt der Wert wieder): der Teil des Produkts, den eine Symmetrisierung wegwirft, ist nicht klein.")


def verdict_identities(rows: list) -> tuple:
    worst = max(max(r["Qz_s"], r["Wz_lam"], r["Winv_s_lam"], r["asym"]) for r in rows)
    return "success", (f"An {C.IDENTITY_POINTS} gezogenen Punkten je Kegel stimmen die Identitäten der NT-Skalierung (Q z = s, W z = λ, W⁻¹ s = λ, W symmetrisch) bis auf **{sci(worst)}**; "
                       f"der kleinste Eigenwert von W ist {de(min(r['min_eig'] for r in rows), 2)} (also positiv definit).")


def verdict_invariance(rows: list, kappa: int) -> tuple:
    def cell(sc, k):
        return next(r for r in rows if r["scaling"] == sc and r["kappa"] == k)

    nt1, ntk = cell("nt", 1), cell("nt", kappa)
    jo1, jok = cell("jordan", 1), cell("jordan", kappa)
    fmt = lambda c: f"{c['solved']}/{c['count']} gelöst" + (f", Median {de(c['median'], 1)} Iterationen" if c["median"] is not None else "")
    return "info", (f"Zwei PSD-Blöcke 3×3, mit einem Automorphismus der Kondition κ = {sci(float(kappa), 0)} umgeschrieben (Start mitgenommen): NT bei κ = 1: {fmt(nt1)}, bei κ = {sci(float(kappa), 0)}: {fmt(ntk)}; "
                    f"Jordan bei κ = 1: {fmt(jo1)}, bei κ = {sci(float(kappa), 0)}: {fmt(jok)}. Die Iterationszahl des skalierten Schritts hängt für gelungene Läufe bis κ = 10⁴ nicht an κ (bei κ = 10⁶ stammt der Median aus sehr wenigen Läufen); wie viele Läufe numerisch gelingen, tut es.")


def verdict_nu(rows: list, drows: list) -> tuple:
    a, b = rows[0], rows[-1]
    da, db = drows[0], drows[-1]
    return "info", (f"Kugel mit n = {a['n']} bis {b['n']} Punkten (ν von {a['nu']} auf {b['nu']}, Zeilen von {a['rows']} auf {b['rows']}): der NT-Schritt braucht {int(min(r['nt'] for r in rows))} bis "
                    f"{int(max(r['nt'] for r in rows))} Iterationen (Median je Wolkensatz), die Barriere {int(min(r['newton'] for r in rows))} bis {int(max(r['newton'] for r in rows))} Newton-Schritte in "
                    f"{int(min(r['rounds'] for r in rows))} bis {int(max(r['rounds'] for r in rows))} Runden. Allgemeine Ellipse von d = {da['d']} bis d = {db['d']}: NT {int(da['nt'])} auf {int(db['nt'])} Iterationen, "
                    f"Barriere {int(da['newton'])} auf {int(db['newton'])} Newton-Schritte. Eine Iteration löst ein dichtes KKT-System der Größe Zeilen + Variablen: der Vergleich zählt Iterationen, keine Zeit.")


def verdict_certificate(res: dict) -> tuple:
    bar, pd = res["barrier"], res["pd"]
    lo, hi = sorted([pd.value, pd.dual_value])
    return "success", (f"Die Barriere ohne Absicherung meldet den Wert {num(bar.value)} mit Lückenschranke {sci(bar.gap_bound)} (`far_from_path = {bar.far_from_path}`); das Optimum ist {sci(res['optimum'])}. "
                       f"Der primal-duale Löser braucht {pd.iterations} Iterationen und meldet Primal {sci(pd.value, 5)} und Dual {sci(pd.dual_value, 5)}: **beide klammern das Optimum** "
                       f"({sci(lo, 5)} ≤ {sci(res['optimum'], 5)} ≤ {sci(hi, 5)}), die Lücke cᵀx + bᵀz ist {sci(abs(pd.certificate_gap))}. Das Zertifikat kommt aus einem echten Dualpunkt, nicht aus der Annahme eines zentrierten Punktes.")


def verdict_slater(rows: list) -> tuple:
    gap = [r for r in rows if r["label"].startswith("Lücken")]
    vals = [r["result"].value for r in gap]
    lp = next(r for r in rows if r["label"].startswith("LP"))
    na = next(r for r in rows if r["label"].endswith("Ziel x₁"))
    weak = next(r for r in rows if "schwach" in r["label"])
    return "info", (f"Lückenbeispiel: Primal und Dual sind für jedes ε > 0 {de(min(vals), 4)} bis {de(max(vals), 4)} (Wert −1, {gap[0]['result'].iterations} bis {gap[-1]['result'].iterations} Iterationen). "
                    f"Nicht angenommenes Infimum: Primal {sci(na['result'].value)}, aber ‖x‖ = {sci(na['x_norm'])}. Schwach unzulässig: nach {weak['result'].iterations} Iterationen ‖x‖ = {sci(weak['x_norm'])}, kein Abschluss. "
                    f"Unzulässiges LP: der Dualpunkt läuft davon (‖z‖ = {sci(lp['z_norm'])}); normiert ist ẑ ein Farkas-Strahl mit ‖Aᵀẑ‖ = {sci(lp['ray_ATz'])} und bᵀẑ = {de(lp['ray_bTz'], 3)} < 0.")
