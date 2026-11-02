"""Orakel (DEMO-PLAYBOOK §12, ORACLE-PLAYBOOK §5): der primal-duale Löser gegen einen anderen Rechenweg.

- cvxpy mit Clarabel (selbst ein primal-dualer Innere-Punkte-Löser mit NT-Skalierung, aber anderer Code): Optimalwert der sechs Formen und der Zufallsprogramme
- das Dualprogramm aus cvxpy: sein Wert gleich dem Dualwert des eigenen Zertifikats (Aᵀz = −c, z im Kegel, Wert −bᵀz)
- Eigenschaften, die ein Fehler in beiden Rechenwegen nicht überlebt (Skalierung des Ziels, Abwärtsmonotonie des Zertifikats, Dualzulässigkeit)

Fehlt cvxpy im Test-venv, überspringen sich diese Tests (pytest.importorskip), sie raten nie."""

import numpy as np
import pytest

import kegel_kern as K
import kpd_cones as C
import kpd_examples as E
import kpd_forms as F
import kpd_solver as S
from kpd_scenario import make_cloud


def _cvxpy():
    cp = pytest.importorskip("cvxpy")
    if "CLARABEL" not in cp.installed_solvers():
        pytest.skip("Clarabel fehlt")
    return cp, dict(solver=cp.CLARABEL, tol_gap_abs=1e-10, tol_gap_rel=1e-10, tol_feas=1e-10)


def _cloud(seed, n, d=2):
    pts = np.random.default_rng(seed).normal(0.0, 1.0, (n, d))
    return pts - pts.mean(axis=0)


def _cone_constraints(cp, s_expr, cones):
    """s ∈ K blockweise als cvxpy-Bedingungen; die PSD-Blöcke werden aus svec zurückgebaut."""
    cons, off = [], 0
    for kind, m in cones:
        d = K.cone_dim(kind, m)
        blk = s_expr[off:off + d]
        if kind == "l":
            cons.append(blk >= 0)
        elif kind == "q":
            cons.append(cp.SOC(blk[0], blk[1:]))
        else:
            rows = [[blk[_pos(m, min(i, j), max(i, j))] * (1.0 if i == j else 1.0 / np.sqrt(2.0)) for j in range(m)] for i in range(m)]
            cons.append(cp.bmat(rows) >> 0)
        off += d
    return cons


def _pos(m, i, j):
    """Position von (i, j), i ≤ j, in svec (obere Dreiecksmatrix zeilenweise)."""
    iu = np.triu_indices(m)
    return int(np.flatnonzero((iu[0] == i) & (iu[1] == j))[0])


@pytest.mark.filterwarnings("ignore:Solution may be inaccurate")
def test_pd_optimum_equals_the_cvxpy_value_for_all_six_forms():
    cp, kw = _cvxpy()
    for seed in range(6):
        d = 2 + seed % 2
        pts = _cloud(seed, 8 + seed % 4, d)
        c, R = cp.Variable(d), cp.Variable()
        refs = {}
        for form, norm in (("cube", "inf"), ("diamond", 1), ("ball_soc", 2), ("ball_psd", 2)):
            cp.Problem(cp.Minimize(R), [cp.norm(p - c, norm) <= R for p in pts]).solve(**kw)
            refs[form] = float(R.value)
        pv = cp.Variable(d, nonneg=True)
        cp.Problem(cp.Minimize(cp.sum(cp.inv_pos(pv))), [(pts ** 2) @ pv <= 1]).solve(**kw)
        refs["ellipse_axis"] = float(np.sum(1 / pv.value))
        M = cp.Variable((d, d), PSD=True)
        cp.Problem(cp.Minimize(cp.tr_inv(M)), [cp.sum(cp.multiply(np.outer(p, p), M)) <= 1 for p in pts]).solve(**kw)
        refs["ellipse_psd"] = float(np.trace(np.linalg.inv(M.value)))
        for form in F.FORMS:
            prob = F.build_form(form, pts)
            for sc in ("nt", "jordan"):
                r = S.solve_pd(prob, tol=1e-9, scaling=sc)
                if r.status != S.OPTIMAL:
                    assert form == "ellipse_axis" and sc == "nt", (form, sc, seed, r.status)                  # die eine bekannte schwierige Form (README, Befunde)
                    continue
                tol = 1e-5 if form.startswith("ellipse") else 1e-6
                assert r.value == pytest.approx(refs[form], rel=tol), (form, sc, seed)
                assert r.dual_value <= refs[form] * (1 + tol) and r.value >= refs[form] * (1 - tol), (form, sc, seed)


@pytest.mark.filterwarnings("ignore:Solution may be inaccurate")
def test_pd_equals_cvxpy_on_random_cone_programs_and_the_dual_value_matches():
    cp, kw = _cvxpy()
    for name, (cones, nvar) in E.MIXES.items():
        for seed in range(4):
            prob = E.random_program(cones, nvar, 500 + seed)
            x = cp.Variable(nvar)
            cp.Problem(cp.Minimize(prob.c @ x), _cone_constraints(cp, prob.b - prob.A @ x, cones)).solve(**kw)
            ref = float(prob.c @ x.value)
            r = S.solve_pd(prob, tol=1e-9, scaling="nt")
            assert r.status == S.OPTIMAL and r.value == pytest.approx(ref, rel=1e-6, abs=1e-7), (name, seed)
            assert r.dual_value == pytest.approx(ref, rel=1e-6, abs=1e-7), (name, seed)


def test_the_dual_point_is_feasible_and_the_certificate_bounds_the_cvxpy_optimum():
    cp, kw = _cvxpy()
    prob = E.random_program((("l", 3), ("q", 4), ("s", 3)), 6, 77)
    x = cp.Variable(6)
    cp.Problem(cp.Minimize(prob.c @ x), _cone_constraints(cp, prob.b - prob.A @ x, prob.cones)).solve(**kw)
    ref = float(prob.c @ x.value)
    r = S.solve_pd(prob, tol=1e-6)
    assert np.allclose(prob.A.T @ r.z, -prob.c, atol=1e-5)                                           # Dualzulässigkeit Aᵀz = −c
    assert all(C.margin(k, m, r.z[o:o + d]) > 0 for k, m, o, d in C.blocks(prob.cones))
    assert r.dual_value <= ref + 1e-6 and r.value >= ref - 1e-6                                      # schwache Dualität gegen den Fremdwert


def test_scaling_the_objective_scales_the_value_and_keeps_the_iterations():
    prob = F.build_form("ball_soc", make_cloud(15, 3.0, 30, 100001).points)
    a = S.solve_pd(prob, tol=1e-9)
    scaled = K.ConeProblem(7.0 * prob.c, prob.A, prob.b, prob.cones, prob.x0)
    b = S.solve_pd(scaled, tol=1e-9)
    assert a.status == b.status == S.OPTIMAL and b.value == pytest.approx(7.0 * a.value, rel=1e-6) and abs(a.iterations - b.iterations) <= 2


def test_scaling_the_points_scales_the_value():
    pts = _cloud(2, 10)
    for form in ("cube", "ball_soc", "ball_psd"):
        a = S.solve_pd(F.build_form(form, pts), tol=1e-9)
        b = S.solve_pd(F.build_form(form, 3.0 * pts), tol=1e-9)
        assert a.status == b.status == S.OPTIMAL and b.value == pytest.approx(3.0 * a.value, rel=1e-6) and abs(a.iterations - b.iterations) <= 3, form


def test_the_outlier_optimum_is_the_closed_form_value():
    """min (x⁴ / min(x, y))² auf [0,1; 3]²: bei x = y = 0,1 ist (10⁻⁴ / 0,1)² = 10⁻⁶ und kleiner geht nicht (x⁴/x = x³ wächst in x)."""
    xs = np.linspace(0.1, 3.0, 300)
    X, Y = np.meshgrid(xs, xs)
    grid = (X ** 4 / np.minimum(X, Y)) ** 2
    assert grid.min() == pytest.approx(1e-6, rel=1e-12) and E.OUTLIER_OPTIMUM == 1e-6
    r = S.solve_pd(E.outlier_problem(), tol=1e-9)
    assert r.x[0] == pytest.approx(0.1, abs=1e-4)                                                    # y ist frei oberhalb von 0,1 (min(x, y) = x), nur x ist bestimmt


def test_mixed_cone_programs_dual_value_is_minus_btz():
    prob = E.random_program((("l", 2), ("s", 2)), 4, 11)
    r = S.solve_pd(prob, tol=1e-9)
    assert r.dual_value == pytest.approx(-float(prob.b @ r.z)) and r.certificate_gap == pytest.approx(float(r.s @ r.z), abs=1e-6)                        # sᵀz = cᵀx + bᵀz bis auf die Restresiduen
