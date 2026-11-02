"""Randfälle, an denen Kernroutinen erfahrungsgemäß brechen (DEMO-PLAYBOOK §12)."""

import numpy as np
import pytest

import kegel_kern as K
import kpd_cones as C
import kpd_evaluation as ev
import kpd_examples as E
import kpd_forms as F
import kpd_solver as S
from kpd_scenario import make_cloud


def test_two_points_give_the_ball_of_the_hand_value():
    """Zwei Punkte (1,2) und (−1,−2): Kugel mit R = √5, Würfel mit R = 2, Raute mit R = 3."""
    pts = np.array([[1.0, 2.0], [-1.0, -2.0]])
    for form, expected in (("ball_soc", 5 ** 0.5), ("ball_psd", 5 ** 0.5), ("cube", 2.0), ("diamond", 3.0)):
        for sc in ("nt", "jordan"):
            r = S.solve_pd(F.build_form(form, pts), tol=1e-10, scaling=sc)
            assert r.status == S.OPTIMAL and r.value == pytest.approx(expected, abs=1e-7), (form, sc)


def test_a_single_point_has_the_optimum_zero_and_no_interior_to_run_into():
    for form in ("cube", "ball_soc"):
        r = S.solve_pd(F.build_form(form, np.array([[3.0, -2.0]])), tol=1e-9)
        assert r.status == S.OPTIMAL and abs(r.value) < 1e-7, form


def test_one_and_three_dimensions_run_through_every_form():
    p1 = np.array([[-2.0], [1.0], [3.0]])
    p3 = np.random.default_rng(5).normal(0, 1, (9, 3))
    for pts in (p1 - p1.mean(), p3 - p3.mean(axis=0)):
        for form in F.FORMS:
            ref = K.solve_barrier(F.build_form(form, pts), tol=1e-10).value
            r = S.solve_pd(F.build_form(form, pts), tol=1e-9, scaling="jordan")
            assert r.status == S.OPTIMAL and r.value == pytest.approx(ref, rel=1e-5, abs=1e-7), (form, pts.shape)


def test_duplicated_points_change_nothing():
    base = make_cloud(10, 3.0, 30, 100000).points
    twice = np.vstack([base, base])
    for form in ("cube", "ball_soc", "ball_psd"):
        a = S.solve_pd(F.build_form(form, base), tol=1e-9)
        b = S.solve_pd(F.build_form(form, twice), tol=1e-9)
        assert a.status == b.status == S.OPTIMAL and b.value == pytest.approx(a.value, rel=1e-6), form


def test_very_small_and_very_large_coordinates_are_solved_alike_up_to_the_unit():
    pts = make_cloud(12, 3.0, 20, 100002).points
    ref = S.solve_pd(F.build_form("ball_soc", pts), tol=1e-9)
    for factor in (1e-3, 1e3):
        r = S.solve_pd(F.build_form("ball_soc", factor * pts), tol=1e-9)
        assert r.status == S.OPTIMAL and r.value == pytest.approx(factor * ref.value, rel=1e-5), factor


def test_a_start_with_a_wrong_scale_is_still_reached_or_flagged_never_silently_optimal():
    prob = F.build_form("ball_soc", make_cloud(10, 2.0, 0, 1).points)
    x, s, z = S.initial_point(prob)
    for factor in (1e-3, 1e4):
        r = S.solve_pd(prob, tol=1e-8, start=(x, factor * s, factor * z))
        if r.status == S.OPTIMAL:
            assert r.pres < 1e-7 and r.dres < 1e-7 and r.value == pytest.approx(S.solve_pd(prob, tol=1e-9).value, rel=1e-5)
        else:
            assert r.status in {S.MAX_ITER, S.STALLED, S.LINALG, S.NAN, S.LEFT_CONE}


def test_a_start_outside_the_cone_is_never_reported_optimal():
    prob = F.build_form("ball_soc", make_cloud(10, 2.0, 0, 1).points)
    x, s, z = S.initial_point(prob)
    bad = S.solve_pd(prob, tol=1e-8, start=(x, -s, z), max_iter=20)
    assert bad.status != S.OPTIMAL


def test_a_program_without_variables_in_a_constraint_row_still_runs():
    """Eine Zeile 0 ≤ 1 (A-Zeile null) fügt dem Orthanten einen unbeteiligten Schlupf hinzu: Optimum bleibt."""
    prob = F.build_form("cube", make_cloud(6, 2.0, 10, 3).points)
    A = np.vstack([prob.A, np.zeros(prob.A.shape[1])])
    padded = K.ConeProblem(prob.c, A, np.concatenate([prob.b, [1.0]]), (*prob.cones, ("l", 1)), prob.x0)
    assert S.solve_pd(padded, tol=1e-9).value == pytest.approx(S.solve_pd(prob, tol=1e-9).value, abs=1e-7)


def test_the_experiments_run_on_the_smallest_and_largest_settings():
    for n in (8, 40):
        for stretch, angle in ((1.0, 0), (8.0, 90)):
            s = ev.Settings(form="ball_psd", n=n, stretch=stretch, angle=angle, seed=1)
            a = ev.analyse(s)
            assert a.nt.status == S.OPTIMAL and ev.deviation(a, a.nt) < 1e-5
            assert ev.commutator_rows(s)["rows"] and ev.verdict_run(a)[0] in {"success", "warning"}


def test_commutator_falls_back_to_the_psd_ball_for_forms_without_a_psd_block():
    res = ev.commutator_rows(ev.Settings(form="cube"))
    assert res["substitute"] and res["form"] == "ball_psd"
    res = ev.commutator_rows(ev.Settings(form="ellipse_psd"))
    assert not res["substitute"] and res["form"] == "ellipse_psd"


def test_max_step_survives_zero_and_extreme_directions():
    assert C.max_step("q", 3, np.array([1.0, 0.0, 0.0]), np.zeros(3)) == float("inf")
    assert C.max_step("l", 2, np.array([1.0, 1.0]), np.zeros(2)) == float("inf")
    assert C.max_step("s", 2, C.svec(np.eye(2)), np.zeros(3)) == float("inf")
    assert C.max_step("q", 3, np.array([1.0, 0.0, 0.0]), np.array([-1e12, 0.0, 0.0])) == pytest.approx(1e-12)
    assert C.max_step("s", 2, C.svec(np.eye(2)), C.svec(np.diag([-1e9, 1.0]))) == pytest.approx(1e-9)


def test_nt_scaling_of_nearly_equal_points_is_the_identity_scaling():
    rng = np.random.default_rng(9)
    for kind, m in (("l", 3), ("q", 4), ("s", 3)):
        u = E.interior(kind, m, rng)
        Q, W, lam = C.nt_scaling(kind, m, u, u)
        assert np.allclose(W @ u, lam) and lam @ lam == pytest.approx(u @ u, rel=1e-9)               # s = z: λ² = s∘s, W bildet z auf λ ab


def test_the_solver_accepts_the_non_strict_x0_field_without_reading_it():
    prob = E.gapex_problem(1e-3)
    bad = K.ConeProblem(prob.c, prob.A, prob.b, prob.cones, np.array([1e6, -1e6]))
    assert S.solve_pd(bad, tol=1e-8).value == pytest.approx(-1.0, abs=1e-6)
