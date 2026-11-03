"""Der primal-duale Löser (kpd_solver): Mini-Programme von Hand, Zertifikat, Eigenschaften der Iteration, Fehlerfälle."""

import numpy as np
import pytest

import kegel_kern as K
import kpd_examples as E
import kpd_forms as F
import kpd_solver as S
from kpd_scenario import make_cloud


@pytest.fixture
def minis(tiny_cones):
    return tiny_cones


def test_the_three_hand_programs_reach_their_optimum_and_the_dual_brackets_it(minis):
    for name, (prob, opt) in minis.items():
        r = S.solve_pd(prob, tol=1e-10)
        assert r.status == S.OPTIMAL and r.value == pytest.approx(opt, abs=1e-8), name
        assert r.dual_value <= opt + 1e-8 and r.value >= opt - 1e-8 and r.certificate_gap == pytest.approx(r.gap, abs=1e-8) and r.certificate_gap >= -1e-9, name
        assert r.pres < 1e-10 and r.dres < 1e-10


def test_dual_solutions_by_hand(minis):
    """Orthant: z = 1; Lorentz: z = (1; −3/5, −4/5); PSD: Z = ½ [[1, −1], [−1, 1]] (svec: (½, −√2/2, ½))."""
    z = {name: S.solve_pd(prob, tol=1e-11).z for name, (prob, _) in minis.items()}
    assert z["lp"] == pytest.approx([1.0], abs=1e-6)
    assert z["soc"] == pytest.approx([1.0, -0.6, -0.8], abs=1e-6)
    assert z["sdp"] == pytest.approx([0.5, -np.sqrt(2.0) / 2.0, 0.5], abs=1e-6)


def test_certificate_is_the_difference_of_the_two_numbers_and_never_negative_up_to_rounding(minis):
    for prob, _ in minis.values():
        r = S.solve_pd(prob, tol=1e-9)
        assert r.certificate_gap == pytest.approx(r.value - r.dual_value) and all(h["pobj"] - h["dobj"] >= -1e-8 * (1.0 + abs(h["pobj"])) or h["pres"] > 1e-6 for h in r.history)


def test_the_history_records_every_iteration_and_the_gap_falls(minis):
    prob, _ = minis["soc"]
    r = S.solve_pd(prob, tol=1e-10, record=True)
    assert [h["it"] for h in r.history] == list(range(r.iterations + 1)) and len(r.iterates) == len(r.history)
    assert all(h["alpha"] is not None for h in r.history[:-1]) and r.history[-1]["alpha"] is None
    assert r.history[-1]["gap"] < 1e-3 * r.history[0]["gap"]
    s0, z0 = r.iterates[0]
    assert len(s0) == len(z0) == 3
    assert S.solve_pd(prob, tol=1e-10).iterates == ()


def test_the_iterates_stay_strictly_inside_the_cones(minis):
    for prob, _ in minis.values():
        r = S.solve_pd(prob, tol=1e-10, record=True)
        for s, z in r.iterates:
            off = 0
            for kind, m in prob.cones:
                d = K.cone_dim(kind, m)
                assert K.cone_margin(kind, s[off:off + d], m) > 0 and K.cone_margin(kind, z[off:off + d], m) > 0
                off += d


def test_the_start_does_not_use_the_strictly_feasible_x0(minis):
    prob, opt = minis["sdp"]
    bad = K.ConeProblem(prob.c, prob.A, prob.b, prob.cones, np.array([-100.0]))              # x0 verletzt den Kegel: der Löser liest es nicht
    assert S.solve_pd(bad, tol=1e-9).value == pytest.approx(opt, abs=1e-6)
    x, s, z = S.initial_point(prob)
    assert K.cone_margin("s", s, 2) > 0 and K.cone_margin("s", z, 2) > 0


def test_the_three_directions_are_the_same_step_in_the_orthant():
    cloud = make_cloud(20, 4.0, 45, 35)
    for form in ("cube", "diamond"):
        prob = F.build_form(form, cloud.points)
        runs = {sc: S.solve_pd(prob, tol=1e-9, scaling=sc) for sc in S.SCALINGS}
        assert len({r.iterations for r in runs.values()}) == 1 and all(r.status == S.OPTIMAL for r in runs.values()), form
        assert max(abs(runs["nt"].value - r.value) for r in runs.values()) < 1e-7


def test_the_literal_lp_step_fails_in_the_cone_forms():
    cloud = make_cloud(20, 4.0, 45, 35)
    for form in ("ball_soc", "ball_psd", "ellipse_psd"):
        r = S.solve_pd(F.build_form(form, cloud.points), tol=1e-8, scaling="vector")
        assert r.status != S.OPTIMAL, form
    for name, (cones, nvar) in E.MIXES.items():
        if all(k == "l" for k, _ in cones):
            continue
        solved = sum(S.solve_pd(E.random_program(cones, nvar, 1000 + k), scaling="vector").status == S.OPTIMAL for k in range(10))
        assert solved == 0, name


def test_jordan_and_nt_solve_the_balls_and_agree_with_the_barrier():
    cloud = make_cloud(20, 4.0, 45, 35)
    for form in ("ball_soc", "ball_psd", "ellipse_psd"):
        prob = F.build_form(form, cloud.points)
        ref = K.solve_barrier(prob, tol=1e-10).value
        for sc in ("nt", "jordan"):
            r = S.solve_pd(prob, tol=1e-8, scaling=sc)
            assert r.status == S.OPTIMAL and abs(r.value - ref) < 1e-6 * (1 + abs(ref)), (form, sc)


def test_iterations_are_a_small_number_and_flat_in_the_size():
    iters = []
    for n in (4, 10, 20, 40):
        prob = F.build_form("ball_soc", make_cloud(n, 3.0, 30, 100000).points)
        r = S.solve_pd(prob, tol=1e-8)
        assert r.status == S.OPTIMAL
        iters.append(r.iterations)
    assert max(iters) <= 14 and max(iters) - min(iters) <= 5


def test_the_outlier_of_the_kernel_is_solved_and_bracketed():
    r = S.solve_pd(E.outlier_problem(), tol=1e-9)
    assert r.status == S.OPTIMAL and r.value == pytest.approx(E.OUTLIER_OPTIMUM, abs=2e-9) and r.dual_value == pytest.approx(E.OUTLIER_OPTIMUM, abs=2e-9)
    assert r.dual_value <= E.OUTLIER_OPTIMUM <= r.value or abs(r.value - r.dual_value) < 1e-9
    assert r.certificate_gap < 1e-9


def test_max_iter_and_unknown_scaling():
    prob = F.build_form("ball_soc", make_cloud(10, 2.0, 0, 1).points)
    r = S.solve_pd(prob, max_iter=2)
    assert r.status == S.MAX_ITER and r.iterations == 2 and len(r.history) == 3
    with pytest.raises(ValueError, match="unbekannte Skalierung"):
        S.solve_pd(prob, scaling="egal")


def test_a_tighter_tolerance_costs_a_few_more_iterations():
    prob = F.build_form("ball_psd", make_cloud(20, 4.0, 45, 35).points)
    loose, tight = S.solve_pd(prob, tol=1e-4), S.solve_pd(prob, tol=1e-9)
    assert loose.iterations < tight.iterations <= loose.iterations + 6


def test_the_farkas_ray_of_an_infeasible_lp():
    lp = E.lp_infeasible_problem()
    r = S.solve_pd(lp, tol=1e-9, max_iter=60)
    assert r.status != S.OPTIMAL and np.linalg.norm(r.z) > 1e10
    zh, atz, btz = S.farkas_ray(lp, r)
    assert np.linalg.norm(zh) == pytest.approx(1.0) and atz < 1e-6 and btz < -0.5 and np.all(zh > 0)
    assert zh == pytest.approx(np.array([1.0, 1.0]) / np.sqrt(2.0), abs=1e-3)                        # das Zertifikat z = (1, 1) von Hand


def test_a_program_without_a_feasible_point_never_reports_optimal():
    prob = K.ConeProblem(np.array([1.0]), np.array([[1.0], [-1.0]]), np.array([-1.0, -1.0]), (("l", 2),), np.array([0.0]))      # x ≤ −1 und −x ≤ −1: unzulässig
    assert S.solve_pd(prob, max_iter=60).status != S.OPTIMAL


def test_single_thread_and_no_warnings(recwarn):
    prob = F.build_form("ellipse_axis", make_cloud(20, 4.0, 45, 35).points)
    S.solve_pd(prob, max_iter=30)
    assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)]                         # Überläufe beim Davonlaufen bleiben still


def test_the_tolerance_is_relative_and_a_reported_success_meets_it_on_every_form():
    """Das Abbruchkriterium ist relativ (Lücke sᵀz ≤ tol · max(1, |cᵀx|)): ein gemeldeter Erfolg weicht vom unabhängigen Referenzwert (Barriere, Lücke 10⁻¹²) relativ um höchstens
    10 · tol ab. Absolut sind es bei Werten um 134 (achsenparallele Ellipse) mehr als tol: 4,8·10⁻⁷ sind relativ 3,6·10⁻⁹, kein Fehler des Kriteriums. Der Status bleibt ungeprüft (CPU-abhängig)."""
    import kpd_evaluation as ev

    cloud = make_cloud(20, 4.0, 45, 35)
    for form in F.FORMS:
        prob = F.build_form(form, cloud.points)
        ref = K.solve_barrier(prob, tol=1e-12).value
        for sc in ("nt", "jordan"):
            r = S.solve_pd(prob, tol=1e-8, scaling=sc)
            if r.status != S.OPTIMAL:
                continue
            err = abs(r.value - ref)
            assert ev.relative(err, ref) <= 10e-8, (form, sc, err)
    assert ev.relative(4.8e-7, 134.265) == pytest.approx(3.575e-9, rel=1e-3) and ev.relative(3e-9, 0.5) == 3e-9       # unter 1 zählt absolut
