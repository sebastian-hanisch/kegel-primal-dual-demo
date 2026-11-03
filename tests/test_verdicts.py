"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Zahl im Text und jede Schwelle der Art steht hier."""

from types import SimpleNamespace

import numpy as np

import kpd_constants as C
import kpd_evaluation as ev
import kpd_solver as S


def test_formatting_helpers():
    assert ev.de(2.3456, 3) == "2,346" and ev.de(-1.0, 0) == "−1" and ev.sci(1.7e-10) == "1,7·10⁻¹⁰" and ev.sci(2.5e5) == "2,5·10⁵" and ev.sci(float("inf")) == "unendlich"
    assert ev.sci(9.99e-3, 1) == "1,0·10⁻²" and ev.sci(0.0) == "0,0" and ev.num(0.5) == "0,5000" and ev.num(12.3456) == "12,346" and ev.num(1e-6) == "1,00·10⁻⁶" and ev.num(2e5) == "2,00·10⁵"
    assert ev.status_short("stalled") == "steht still" and ev.status_short("unbekannt") == "unbekannt" and set(C.STATUS_SHORT) == set(C.STATUS_NAMES)
    assert ev.status_text("optimal") == "gelöst" and ev.status_text("stalled") == "steht still (Schrittlänge 0)" and ev.status_text("unbekannt") == "unbekannt"


def _res(status="optimal", it=10, value=2.0, dual=2.0 - 1e-9, cert=1e-9):
    return SimpleNamespace(status=status, iterations=it, value=value, dual_value=dual, certificate_gap=cert)


def _analysis(nt, jo, ve, form="ball_soc", reference=2.0, newton=34, rounds=13):
    return SimpleNamespace(settings=SimpleNamespace(form=form), runs={"nt": nt, "jordan": jo, "vector": ve}, reference=reference, barrier=SimpleNamespace(newton_steps=newton, rounds=rounds))


def test_run_verdict_success_names_iterations_certificate_and_deviation():
    a = _analysis(_res(it=11, value=2.0000000003, dual=1.9999999990, cert=1.3e-9), _res(it=8), _res("stalled", 6))
    kind, text = ev.verdict_run(a)
    assert kind == "success"
    for part in ("**Kugel (Lorentz-Kegel)**", "NT-Schritt 11 Iterationen (gelöst)", "Jordan-Schritt 8 (gelöst)", "endet nach 6 (steht still (Schrittlänge 0))", "34 Newton-Schritte in 13 Runden",
                 "Primal 2,0000, Dual 2,0000", "Lücke cᵀx + bᵀz = 1,3·10⁻⁹", "Abweichung vom Referenzwert der Barriere 3,0·10⁻¹⁰, relativ 1,5·10⁻¹⁰ (die Toleranz 1·10⁻⁸ gilt relativ zu max(1, |Wert|))"):
        assert part in text, part


def test_run_verdict_warns_when_the_nt_run_ends_at_the_rounding_limit():
    kind, text = ev.verdict_run(_analysis(_res("nan", 24, cert=3e-5), _res(it=22), _res("stalled", 7), form="ellipse_axis"))
    assert kind == "warning"
    for part in ("**Ellipse, achsenparallel**", "24 Iterationen (Rechengrenze (nicht endlich))", "Lücke cᵀx + bᵀz = 3,0·10⁻⁵", "kein Wert, dem man trauen sollte"):
        assert part in text, part
    assert "Das Zertifikat des NT-Laufs" not in text


def _variant_rows(lp_equal=True, vec_ok=False, jordan_ok=4, nt_ok=4):
    rows = []
    for form in ("cube", "diamond", "ball_soc", "ball_psd", "ellipse_axis", "ellipse_psd"):
        cone = form not in ("cube", "diamond")
        i = 0
        for sc in S.SCALINGS:
            if not cone:
                its, ok = (8 if lp_equal or sc == "nt" else 9), True
            elif sc == "vector":
                its, ok = 6, vec_ok
            else:
                i += 1
                its, ok = 10, (jordan_ok if sc == "jordan" else nt_ok) >= {"ball_soc": 1, "ball_psd": 2, "ellipse_axis": 3, "ellipse_psd": 4}[form]
            rows.append({"form": form, "scaling": sc, "status": "optimal" if ok else "stalled", "iterations": its, "error": 0.0, "certificate_gap": 0.0, "cones": "l"})
    return rows


def test_variants_verdict_counts_failures_and_successes():
    kind, text = ev.verdict_variants(_variant_rows())
    assert kind == "info"
    for part in ("dieselben Iterationen", "scheitert der wörtlich übertragene LP-Schritt in **4 von 4**", "der Jordan-Schritt löst 4, der NT-Schritt 4 davon"):
        assert part in text, part
    kind, text = ev.verdict_variants(_variant_rows(lp_equal=False, vec_ok=True, jordan_ok=3, nt_ok=2))
    assert "verschiedene Iterationen" in text and "in **0 von 4**" in text and "Jordan-Schritt löst 3, der NT-Schritt 2" in text


def _random_rows(med=(6.0, 8.5)):
    rows = []
    for mix in ("Orthant: 12 Zeilen", "PSD: zwei 3×3"):
        rows += [{"mix": mix, "scaling": "vector", "solved": 12 if mix.startswith("Orth") else 0, "count": 12, "median": 7.0 if mix.startswith("Orth") else None, "max": 9},
                 {"mix": mix, "scaling": "jordan", "solved": 12, "count": 12, "median": med[0], "max": 8},
                 {"mix": mix, "scaling": "nt", "solved": 11, "count": 12, "median": med[1], "max": 12}]
    return rows


def test_random_verdict_lists_every_mix_with_counts_and_medians():
    kind, text = ev.verdict_random(_random_rows())
    assert kind == "info" and "je 12" in text
    for part in ("Orthant: 12 Zeilen: wörtlich 12/12, Jordan 12/12 (Median 6,0), NT 11/12 (Median 8,5)", "PSD: zwei 3×3: wörtlich 0/12, Jordan 12/12 (Median 6,0), NT 11/12 (Median 8,5)", "**weniger** Iterationen"):
        assert part in text, part
    rows = _random_rows()
    rows[0]["median"] = None
    rows[3] = {**rows[3], "median": None}
    assert "wörtlich 12/12, Jordan 12/12 (Median 6,0)" in ev.verdict_random(rows)[1]


def test_commutator_verdict_names_the_hand_values_and_the_peak():
    hand = ev.hand_commutator()
    res = {"form": "ball_psd", "substitute": False, "rows": [{"it": 0, "ratio": 1e-17, "gap": 1.0}, {"it": 3, "ratio": 0.14, "gap": 0.1}, {"it": 9, "ratio": 0.02, "gap": 1e-9}]}
    kind, text = ev.verdict_commutator(res, hand)
    assert kind == "info"
    for part in ("S Z = [[2, 2], [1, 2]]", "Z S = [[2, 1], [2, 2]]", "antisymmetrische Anteil hat die Norm 0,707", "bei ‖S Z‖ = 3,606", "von 1,0·10⁻¹⁷ am Start auf höchstens **0,14** (Iteration 3, am ersten PSD-Block gemessen", "Kugel (PSD-Kegel)"):
        assert part in text, part
    assert "keinen PSD-Kegel" not in text and "Entlang des NT-Laufs der Form Kugel (PSD-Kegel) wächst" in text
    assert "Die gewählte Form hat keinen PSD-Kegel; entlang des NT-Laufs der PSD-Kugel derselben Wolke wächst" in ev.verdict_commutator({**res, "substitute": True}, hand)[1]


def test_hand_commutator_by_hand():
    h = ev.hand_commutator()
    assert np.allclose(h["SZ"], [[2, 2], [1, 2]]) and np.allclose(h["ZS"], [[2, 1], [2, 2]]) and np.allclose(h["sym"], [[2, 1.5], [1.5, 2]])
    assert np.isclose(h["antisym_norm"], np.sqrt(0.5)) and np.isclose(h["SZ_norm"], np.sqrt(13.0))                    # ‖[[0, ½], [−½, 0]]‖ = √½, ‖SZ‖² = 4 + 4 + 1 + 4
    assert np.isclose(h["ratio"], np.sqrt(2.0) / (np.sqrt(5.0) * np.sqrt(7.0)))                                     # ‖SZ − ZS‖ = ‖[[0,1],[−1,0]]‖ = √2


def test_identities_verdict_names_the_worst_error_and_the_smallest_eigenvalue():
    rows = [{"cone": "Orthant 4", "Qz_s": 1e-16, "Wz_lam": 2e-16, "Winv_s_lam": 3e-16, "asym": 0.0, "min_eig": 0.05},
            {"cone": "PSD 5", "Qz_s": 4e-15, "Wz_lam": 1e-15, "Winv_s_lam": 5e-14, "asym": 2e-16, "min_eig": 0.02}]
    kind, text = ev.verdict_identities(rows)
    assert kind == "success" and "200 gezogenen Punkten" in text and "bis auf **5,0·10⁻¹⁴**" in text and "kleinste Eigenwert von W ist 0,02" in text


def _inv_rows():
    return [{"kappa": 1, "scaling": "nt", "solved": 10, "count": 10, "median": 8.5, "max": 10}, {"kappa": 10000, "scaling": "nt", "solved": 4, "count": 10, "median": 8.5, "max": 12},
            {"kappa": 1, "scaling": "jordan", "solved": 10, "count": 10, "median": 7.0, "max": 9}, {"kappa": 10000, "scaling": "jordan", "solved": 9, "count": 10, "median": 7.0, "max": 10},
            {"kappa": 1000000, "scaling": "nt", "solved": 0, "count": 10, "median": None, "max": None}, {"kappa": 1000000, "scaling": "jordan", "solved": 2, "count": 10, "median": 16.0, "max": 20}]


def test_invariance_verdict_compares_kappa_one_with_the_chosen_kappa():
    kind, text = ev.verdict_invariance(_inv_rows(), 10000)
    assert kind == "info"
    for part in ("κ = 1·10⁴ umgeschrieben", "NT bei κ = 1: 10/10 gelöst, Median 8,5 Iterationen, bei κ = 1·10⁴: 4/10 gelöst, Median 8,5 Iterationen",
                 "Jordan bei κ = 1: 10/10 gelöst, Median 7,0 Iterationen, bei κ = 1·10⁴: 9/10 gelöst, Median 7,0 Iterationen", "bis κ = 10⁴ nicht an κ"):
        assert part in text, part
    assert "0/10 gelöst;" in ev.verdict_invariance(_inv_rows(), 1000000)[1] and "Median None" not in text


def test_nu_verdict_names_the_ranges_for_ball_and_ellipse():
    rows = [{"n": 4, "nu": 8, "rows": 12, "nt": 9.0, "newton": 29.0, "rounds": 12.0}, {"n": 80, "nu": 160, "rows": 240, "nt": 12.0, "newton": 34.0, "rounds": 13.0}]
    drows = [{"d": 2, "nt": 12.0, "newton": 42.0}, {"d": 6, "nt": 17.0, "newton": 58.0}]
    kind, text = ev.verdict_nu(rows, drows)
    assert kind == "info"
    for part in ("n = 4 bis 80 Punkten (ν von 8 auf 160, Zeilen von 12 auf 240)", "NT-Schritt braucht 9 bis 12 Iterationen", "29 bis 34 Newton-Schritte in 12 bis 13 Runden",
                 "d = 2 bis d = 6: NT 12 auf 17 Iterationen", "Barriere 42 auf 58 Newton-Schritte", "zählt Iterationen, keine Zeit"):
        assert part in text, part


def test_certificate_verdict_says_both_values_bracket_the_optimum():
    bar = SimpleNamespace(value=190.386, gap_bound=1.7e-10, far_from_path=True)
    pd = SimpleNamespace(value=1.0e-6 + 6e-11, dual_value=1.0e-6 - 7e-11, iterations=18, certificate_gap=1.2e-10)
    kind, text = ev.verdict_certificate({"barrier": bar, "pd": pd, "optimum": 1e-6})
    assert kind == "success"
    for part in ("190,386", "Lückenschranke 1,7·10⁻¹⁰", "`far_from_path = True`", "Optimum ist 1,0·10⁻⁶", "18 Iterationen", "Primal 1,00006·10⁻⁶ und Dual 9,99930·10⁻⁷", "**beide klammern das Optimum** (9,99930·10⁻⁷ ≤ 1,00000·10⁻⁶ ≤ 1,00006·10⁻⁶)", "Lücke cᵀx + bᵀz ist 1,2·10⁻¹⁰"):
        assert part in text, part


def _slater_rows():
    r = lambda it, v, ds=-1.0: SimpleNamespace(iterations=it, value=v, dual_value=ds)
    return [{"label": "Lückenbeispiel ε = 0.1", "result": r(9, -1.0), "x_norm": 1.0, "z_norm": 1.0},
            {"label": "Lückenbeispiel ε = 1e-05", "result": r(14, -0.99999), "x_norm": 1.0, "z_norm": 1.0},
            {"label": "[[x₁,1],[1,x₂]] ⪰ 0, Ziel x₁", "result": r(30, 4e-9), "x_norm": 2.5e8, "z_norm": 1.0},
            {"label": "[[x,1],[1,0]] ⪰ 0 (schwach unzulässig)", "result": r(60, 1.0), "x_norm": 3.0e7, "z_norm": 1.0},
            {"label": "LP x ≥ 1 und x ≤ 0", "result": r(60, 0.0), "x_norm": 1.0, "z_norm": 4.0e12, "ray_ATz": 2e-13, "ray_bTz": -0.707}]


def test_slater_verdict_names_each_case():
    kind, text = ev.verdict_slater(_slater_rows())
    assert kind == "info"
    for part in ("−1,0000 bis −1,0000 (Wert −1, 9 bis 14 Iterationen)", "Primal 4,0·10⁻⁹, aber ‖x‖ = 2,5·10⁸", "nach 60 Iterationen ‖x‖ = 3,0·10⁷", "‖z‖ = 4,0·10¹²", "‖Aᵀẑ‖ = 2,0·10⁻¹³ und bᵀẑ = −0,707 < 0"):
        assert part in text, part
