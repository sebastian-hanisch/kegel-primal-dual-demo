"""Jede Zahl aus README.md und den App-Texten wird hier über dieselben Auswertungsfunktionen nachgerechnet (DEMO-PLAYBOOK §4).

Alle Sweeps laufen über feste Wolken (SWEEP_SEEDS) oder feste Zufallsprogramme; die Standardwolke der Primäransicht (Seed 35) ist ausdrücklich eine einzelne.
Bänder statt Punktwerte, wo die Rundung die Zahl bewegt: die Auswahl der BLAS-Routinen auf dem Runner verschiebt Läufe im Rundungsbereich (Lehre vom 2026-11-01);
dort nennt die README beide Zahlen (Entwicklungsrechner und Band)."""

import math

import numpy as np
import pytest

import kegel_kern as K
import kpd_constants as C
import kpd_evaluation as ev
import kpd_examples as E
import kpd_forms as F
import kpd_solver as S


@pytest.fixture(scope="module")
def std():
    return ev.analyse(ev.Settings())


@pytest.fixture(scope="module")
def variants():
    rows = ev.variant_rows(ev.Settings())
    return {(r["form"], r["scaling"]): r for r in rows}


@pytest.fixture(scope="module")
def random_rows():
    rows = ev.random_rows(C.RANDOM_PROGRAMS)
    return {(r["mix"], r["scaling"]): r for r in rows}


def test_standard_cloud_run(std):
    """README: NT 11, Jordan 8 Iterationen, der wörtliche Schritt steht nach 6 still; Barriere 34 Newton-Schritte in 13 Runden; Zertifikat des NT-Laufs unter 10⁻⁷."""
    nt, jo, ve = std.runs["nt"], std.runs["jordan"], std.runs["vector"]
    assert nt.status == jo.status == S.OPTIMAL and 9 <= nt.iterations <= 13 and 6 <= jo.iterations <= 10 and ve.status == S.STALLED and ve.iterations <= 10
    assert 28 <= std.barrier.newton_steps <= 40 and 12 <= std.barrier.rounds <= 14
    assert abs(nt.certificate_gap) < 1e-7 and ev.deviation(std, nt) < 1e-8 and ev.deviation(std, jo) < 1e-8
    assert std.problem.A.shape == (60, 3) and K.barrier_parameter(std.problem.cones) == 40


def test_the_six_forms_with_the_three_steps(variants):
    """README: Würfel 8 und Raute 9 Iterationen in allen drei Schritten; Kugel Lorentz NT 11 / Jordan 8; Kugel PSD NT 10 / Jordan 8; allgemeine Ellipse NT 23 / Jordan 20; achsenparallele Ellipse: NT endet an der Rechengrenze (Abweichung 1,8·10⁻⁵), Jordan löst sie in 22; der wörtliche Schritt scheitert in allen vier Kegelformen."""
    for form, expected in (("cube", 8), ("diamond", 9)):
        its = {variants[(form, sc)]["iterations"] for sc in S.SCALINGS}
        assert len(its) == 1 and abs(next(iter(its)) - expected) <= 2 and all(variants[(form, sc)]["status"] == S.OPTIMAL for sc in S.SCALINGS), form
    for form in ("ball_soc", "ball_psd", "ellipse_axis", "ellipse_psd"):
        assert variants[(form, "vector")]["status"] != S.OPTIMAL, form
    for form in ("ball_soc", "ball_psd", "ellipse_psd"):
        assert variants[(form, "jordan")]["status"] == S.OPTIMAL, form
    assert variants[("ball_soc", "vector")]["status"] == S.STALLED
    assert variants[("ball_psd", "vector")]["status"] == S.LINALG and variants[("ellipse_psd", "vector")]["status"] == S.LINALG
    nt = {f: variants[(f, "nt")] for f in ("ball_soc", "ball_psd", "ellipse_psd")}
    assert all(r["status"] == S.OPTIMAL for r in nt.values())
    assert 9 <= nt["ball_soc"]["iterations"] <= 13 and 8 <= nt["ball_psd"]["iterations"] <= 12 and 18 <= nt["ellipse_psd"]["iterations"] <= 28
    assert 6 <= variants[("ball_soc", "jordan")]["iterations"] <= 10 and 6 <= variants[("ball_psd", "jordan")]["iterations"] <= 10 and 16 <= variants[("ellipse_psd", "jordan")]["iterations"] <= 24
    for sc in ("nt", "jordan"):                                                                        # achsenparallele Ellipse: Rechengrenze, CPU-abhängig (Entwicklungsrechner: NT endet, Jordan löst; GitHub-Runner: auch Jordan endet)
        hard = variants[("ellipse_axis", sc)]
        assert hard["status"] != S.OPTIMAL or hard["error"] < 1e-5, sc                                 # ein gemeldeter Erfolg stimmt


def test_the_barrier_needs_more_steps_than_the_primal_dual_iterations_on_every_form(std):
    """README: Barriere 34 (Kugel), 57 (allgemeine Ellipse), 49 (achsenparallele Ellipse) Newton-Schritte gegen 8 bis 23 Iterationen."""
    cloud = std.cloud.points
    for form, lo, hi in (("ellipse_psd", 45, 70), ("ellipse_axis", 40, 60)):
        b = K.solve_barrier(F.build_form(form, cloud), tol=1e-9)
        assert lo <= b.newton_steps <= hi, form
        assert b.newton_steps > S.solve_pd(F.build_form(form, cloud), tol=C.TOL, scaling="jordan").iterations


def test_random_programs(random_rows):
    """README: je 12 Zufallsprogramme je Mischung. Wörtlicher Schritt: Orthant 12/12, Lorentz, PSD und gemischt 0/12. Jordan 12/12 mit Median 8, 6, 7, 8; NT 12/12 mit Median 8, 9, 8,5, 10,5 (Bänder)."""
    mixes = list(E.MIXES)
    assert [random_rows[(m, "vector")]["solved"] for m in mixes] == [12, 0, 0, 0] and all(random_rows[(m, "vector")]["count"] == 12 for m in mixes)
    assert all(random_rows[(m, "jordan")]["solved"] == 12 for m in mixes) and all(random_rows[(m, "nt")]["solved"] >= 11 for m in mixes)
    bands_j = {"Orthant: 12 Zeilen": (7, 9), "Lorentz: drei Kegel der Größe 4": (5, 7.5), "PSD: zwei 3×3": (6, 8.5), "gemischt: Orthant 4, Lorentz 4, PSD 3": (7, 9.5)}
    bands_n = {"Orthant: 12 Zeilen": (7, 9), "Lorentz: drei Kegel der Größe 4": (8, 10.5), "PSD: zwei 3×3": (7.5, 10), "gemischt: Orthant 4, Lorentz 4, PSD 3": (9, 12)}
    for m in mixes:
        assert bands_j[m][0] <= random_rows[(m, "jordan")]["median"] <= bands_j[m][1] and bands_n[m][0] <= random_rows[(m, "nt")]["median"] <= bands_n[m][1], m
    for m in mixes[1:]:
        assert random_rows[(m, "jordan")]["median"] < random_rows[(m, "nt")]["median"], m                      # „weniger Iterationen als der skalierte Schritt“
    assert random_rows[(mixes[0], "jordan")]["median"] == random_rows[(mixes[0], "nt")]["median"] == random_rows[(mixes[0], "vector")]["median"]      # im Orthant derselbe Schritt


def test_the_commutator(std):
    """README: Von Hand ‖antisymmetrischer Anteil‖ = 0,707 bei ‖S Z‖ = 3,606; entlang des NT-Laufs der PSD-Kugel von 10⁻¹⁶ am Start bis 0,14, der allgemeinen Ellipse bis 0,09."""
    h = ev.hand_commutator()
    assert h["antisym_norm"] == pytest.approx(math.sqrt(0.5)) and h["SZ_norm"] == pytest.approx(math.sqrt(13.0))
    ball = ev.commutator_rows(ev.Settings())
    assert ball["form"] == "ball_psd" and ball["substitute"] and ball["rows"][0]["ratio"] < 1e-12 and 0.10 <= max(r["ratio"] for r in ball["rows"]) <= 0.20
    peak = max(ball["rows"], key=lambda r: r["ratio"])
    assert 5 <= peak["it"] <= 7 and 1e-3 <= ball["rows"][3]["ratio"] <= 1e-2                               # „3,4·10⁻³ in Iteration 3, 0,14 in Iteration 6“
    ell = ev.commutator_rows(ev.Settings(form="ellipse_psd"))
    assert not ell["substitute"] and 0.06 <= max(r["ratio"] for r in ell["rows"]) <= 0.13


def test_scaling_identities():
    """README: Q z = s, W z = λ, W⁻¹ s = λ und die Symmetrie von W stimmen an 200 Punkten je Kegel bis auf 10⁻¹⁴ (Band 10⁻¹²); der kleinste Eigenwert von W ist 0,11."""
    rows = ev.scaling_identities()
    assert [r["cone"] for r in rows] == ["Orthant 4", "Lorentz 3", "Lorentz 6", "PSD 2", "PSD 3", "PSD 5"]
    assert max(max(r["Qz_s"], r["Wz_lam"], r["Winv_s_lam"], r["asym"]) for r in rows) < 1e-12
    assert 0.05 < min(r["min_eig"] for r in rows) < 0.2


@pytest.fixture(scope="module")
def invariance():
    return {(r["kappa"], r["scaling"]): r for r in ev.invariance_rows()}


def test_invariance(invariance):
    """README: zwei PSD-Blöcke 3×3, 10 Programme je Stufe: NT löst 10, 10, 4, 1 bei κ = 1, 10², 10⁴, 10⁶ mit Median 8,5 (16 bei 10⁶); Jordan löst 10, 9, 9, 2 mit Median 7."""
    nt, jo = {k: invariance[(k, "nt")] for k in C.KAPPAS}, {k: invariance[(k, "jordan")] for k in C.KAPPAS}
    assert nt[1]["solved"] == jo[1]["solved"] == 10 and 7.5 <= nt[1]["median"] <= 9.5 and 6 <= jo[1]["median"] <= 8
    assert nt[100]["solved"] >= 8 and 2 <= nt[10000]["solved"] <= 7 and nt[1000000]["solved"] <= 4 and jo[10000]["solved"] >= 8 and jo[1000000]["solved"] <= 5
    assert jo[10000]["solved"] > nt[10000]["solved"]                                                             # „NT ist nicht robuster“
    assert all(abs(nt[k]["median"] - nt[1]["median"]) <= 2 for k in (100, 10000))                                 # die Iterationszahl hängt nicht an κ
    assert all(r["count"] == C.INVARIANCE_PROGRAMS == 10 for r in invariance.values())


def test_iterations_against_nu_and_dimension():
    """README: Kugel n = 4 bis 80 (ν = 8 bis 160, Zeilen 12 bis 240): NT 9 bis 12, Jordan 6 bis 9, Barriere 29 bis 34 Newton-Schritte in 12 bis 13 Runden (Median über 3 feste Wolken). Allgemeine Ellipse d = 2 bis 6: NT 12 auf 17, Jordan 10 auf 13, Barriere 42 auf 58."""
    rows = ev.nu_rows()
    assert [r["n"] for r in rows] == list(C.N_SWEEP) and [r["nu"] for r in rows] == [2 * n for n in C.N_SWEEP] and [r["rows"] for r in rows] == [3 * n for n in C.N_SWEEP]
    assert all(r["solved_nt"] == r["solved_jordan"] == r["seeds"] == 3 for r in rows)
    assert all(8 <= r["nt"] <= 14 and 5 <= r["jordan"] <= 10 and 26 <= r["newton"] <= 38 and 12 <= r["rounds"] <= 14 for r in rows)
    assert max(r["nt"] for r in rows) - min(r["nt"] for r in rows) <= 5 and max(r["newton"] for r in rows) - min(r["newton"] for r in rows) <= 8
    assert min(r["newton"] / r["nt"] for r in rows) > 2.0                                                          # die Barriere braucht mehr als doppelt so viele Schritte
    drows = ev.dim_rows()
    assert [(r["d"], r["variables"], r["psd_size"], r["nu"]) for r in drows] == [(2, 6, 4, 14), (3, 12, 6, 19), (4, 20, 8, 24), (6, 42, 12, 34)]
    assert all(r["solved_nt"] == r["solved_jordan"] == 3 for r in drows)
    assert 10 <= drows[0]["nt"] <= 14 and 14 <= drows[-1]["nt"] <= 20 and 8 <= drows[0]["jordan"] <= 12 and 11 <= drows[-1]["jordan"] <= 16
    assert 38 <= drows[0]["newton"] <= 48 and 52 <= drows[-1]["newton"] <= 65 and all(r["nt"] < r["newton"] for r in drows)


def test_the_cost_of_one_iteration_is_the_kkt_size():
    """README: ein dichtes KKT-System der Größe Zeilen + Variablen (Standardwolke Kugel: 60 + 3 = 63; allgemeine Ellipse d = 6: 100 + 42 = 142)."""
    prob = F.build_form("ball_soc", ev.analyse(ev.Settings()).cloud.points)
    assert prob.A.shape[0] + prob.A.shape[1] == 63
    drows = ev.dim_rows(ds=(6,), seeds=(100000,))
    assert drows[0]["rows"] + drows[0]["variables"] == 142


def test_the_outlier_certificate():
    """README: Barriere ohne Absicherung 190,386 bei Lückenschranke 1,7·10⁻¹⁰ (far_from_path); PD 18 Iterationen (Band 14 bis 22), Primal 1,00004·10⁻⁶, Dual 9,9992·10⁻⁷, Lücke 1,2·10⁻¹⁰."""
    res = ev.certificate_experiment()
    bar, pd = res["barrier"], res["pd"]
    assert bar.far_from_path and 190.3 < bar.value < 190.5 and bar.gap_bound < 1e-9                          # 190,386 hier, 190,368 auf dem GitHub-Runner
    assert pd.status == S.OPTIMAL and 14 <= pd.iterations <= 22 and pd.dual_value <= E.OUTLIER_OPTIMUM <= pd.value
    assert pd.value == pytest.approx(1.00004e-6, rel=1e-3) and pd.dual_value == pytest.approx(9.9992e-7, rel=1e-3) and pd.certificate_gap == pytest.approx(1.2e-10, rel=0.3)
    assert max(h["pres"] for h in pd.history[-1:]) < 1e-9 and pd.history[-1]["dres"] < 1e-9
    full = next(i for i, h in enumerate(pd.history) if h["alpha"] is not None and h["alpha"] > 0.999)                  # App-Text: nach dem ersten vollen Schritt liegen beide Residuen auf Rundungsniveau
    assert full < len(pd.history) - 2 and all(h["pres"] < 1e-10 and h["dres"] < 1e-10 for h in pd.history[full + 1:])


def test_the_cases_without_a_slater_point():
    """README: Lückenbeispiel ε = 0,1, 10⁻³, 10⁻⁵: Wert −1 in 7, 17 und 28 Iterationen (Band ±4/±6), ‖x‖ wächst 11, 2,3·10³, 1,2·10⁵; nicht angenommenes Infimum: Primal 5·10⁻⁸ bei ‖x‖ = 2,2·10⁷;
    schwach unzulässig: kein Abschluss nach 60 Iterationen, ‖x‖ bis 10¹¹; LP ohne zulässigen Punkt: Abbruch nach 8 Iterationen, ẑ = (1, 1)/√2 mit Aᵀẑ = 0 und bᵀẑ = −0,707."""
    rows = {r["label"]: r for r in ev.slater_rows()}
    gap = [rows[f"Lückenbeispiel ε = {e:g}"] for e in C.SLATER_EPS]
    assert all(r["result"].status == S.OPTIMAL and abs(r["result"].value + 1.0) < 1e-6 and abs(r["result"].dual_value + 1.0) < 1e-6 for r in gap)
    assert all(lo <= r["result"].iterations <= hi for r, (lo, hi) in zip(gap, ((5, 10), (12, 23), (22, 36))))
    assert gap[0]["x_norm"] < gap[1]["x_norm"] < gap[2]["x_norm"] and 5 < gap[0]["x_norm"] < 20 and 1e3 < gap[1]["x_norm"] < 5e3 and 5e4 < gap[2]["x_norm"] < 3e5
    na = rows["[[x₁,1],[1,x₂]] ⪰ 0, Ziel x₁"]
    assert na["result"].value < 1e-6 and 1e6 < na["x_norm"] < 1e8 and na["result"].status == S.OPTIMAL and 30 <= na["result"].iterations <= 50      # der Status „gelöst“ trotz davonlaufendem x
    assert rows["[[x₁,1],[1,x₂]] ⪰ 0, Ziel x₁ + x₂"]["result"].value == pytest.approx(2.0, abs=1e-6)
    weak = rows["[[x,1],[1,0]] ⪰ 0 (schwach unzulässig)"]
    assert weak["result"].status == S.MAX_ITER and weak["result"].iterations == 60 and weak["x_norm"] > 1e9
    lp = rows["LP x ≥ 1 und x ≤ 0"]
    assert lp["result"].status != S.OPTIMAL and 5 <= lp["result"].iterations <= 12 and lp["z_norm"] > 1e10 and lp["ray"] == pytest.approx([1 / math.sqrt(2.0)] * 2, abs=1e-3) and abs(lp["ray_ATz"]) < 1e-6 and lp["ray_bTz"] == pytest.approx(-1 / math.sqrt(2.0), abs=1e-3)


def test_the_kernel_is_unchanged_and_byte_identical_in_the_series():
    """README: kegel_kern.py ist wortgleich mit Stück 2, 4, 5 und 3: Prüfsumme wie im Schwester-Repo, sofern es nebenan liegt."""
    from pathlib import Path
    import hashlib

    here = Path(__file__).resolve().parent.parent / "kegel_kern.py"
    sibling = here.parent.parent / "kegel-barriere-demo" / "kegel_kern.py"
    if not sibling.exists():
        pytest.skip("Schwester-Repo liegt nicht nebenan")
    assert hashlib.sha256(here.read_bytes()).hexdigest() == hashlib.sha256(sibling.read_bytes()).hexdigest()
