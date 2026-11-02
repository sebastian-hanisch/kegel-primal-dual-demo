"""Die festen Beispiele und Zufallsprogramme (kpd_examples) und die Kegelautomorphismen."""

import numpy as np
import pytest

import kegel_kern as K
import kpd_cones as C
import kpd_examples as E
import kpd_solver as S


def test_the_outlier_problem_is_the_documented_one():
    p = E.outlier_problem()
    assert p.A.shape == (15, 6) and K.barrier_parameter(p.cones) == 12 and C.degree(p.cones) == 9 and p.cones.count(("q", 3)) == 3 and p.cones.count(("l", 1)) == 6
    assert np.isfinite(K.barrier_value(p.cones, p.b - p.A @ p.x0))                              # der Start des Kerns liegt streng im Inneren
    assert float(p.c @ p.x0) == pytest.approx(487.163, abs=1e-3)
    res = K.solve_barrier(p, tol=1e-9)
    assert res.far_from_path and res.value > 100.0                                              # der Kern meldet hier einen falschen Wert und das Feld
    assert E.OUTLIER_OPTIMUM == 1e-6


def test_the_outlier_problem_equals_the_one_in_the_kernel_tests():
    import test_kern as T

    a, b = E.outlier_problem(), T._outlier_problem()
    assert np.array_equal(a.A, b.A) and np.array_equal(a.b, b.b) and np.array_equal(a.c, b.c) and a.cones == b.cones and np.array_equal(a.x0, b.x0)


def test_gap_example_matrices_by_hand():
    p = E.gapex_problem(0.5)
    assert p.A.shape == (6, 2) and p.cones == (("s", 3),)
    s = p.b - p.A @ np.array([0.0, 1.0])                                                        # Start (0, 1): diag(ε, 1, 1) positiv definit
    assert np.allclose(K.smat(s, 3), np.diag([0.5, 1.0, 1.0]))
    assert K.cone_margin("s", p.b - p.A @ np.array([0.0, 1.0]), 3) == pytest.approx(0.5)
    p0 = E.gapex_problem(0.0)                                                                   # ε = 0: derselbe Start liegt auf dem Rand, es gibt keinen streng zulässigen Punkt
    assert K.cone_margin("s", p0.b - p0.A @ np.array([0.0, 1.0]), 3) == pytest.approx(0.0, abs=1e-15)


def test_gap_example_has_the_value_minus_one_for_every_positive_eps():
    for eps in (0.1, 1e-3, 1e-5):
        r = S.solve_pd(E.gapex_problem(eps), tol=1e-8)
        assert r.status == S.OPTIMAL and r.value == pytest.approx(-1.0, abs=1e-6) and r.dual_value == pytest.approx(-1.0, abs=1e-6), eps


def test_the_nonattained_infimum_runs_away_and_its_counterpart_has_a_solution():
    r = S.solve_pd(E.nonattained_problem((1.0, 0.0)), tol=1e-8)
    assert abs(r.x[1]) > 1e6 and r.value < 1e-6
    r2 = S.solve_pd(E.nonattained_problem((1.0, 1.0)), tol=1e-9)
    assert r2.status == S.OPTIMAL and r2.x == pytest.approx([1.0, 1.0], abs=1e-6) and r2.value == pytest.approx(2.0, abs=1e-7)


def test_the_weakly_infeasible_program_never_finishes():
    r = S.solve_pd(E.weak_problem(), tol=1e-8, max_iter=60)
    assert r.status != S.OPTIMAL and abs(r.x[0]) > 1e6


def test_the_lp_without_a_feasible_point_has_the_farkas_certificate_by_hand():
    lp = E.lp_infeasible_problem()
    z = np.array([1.0, 1.0])
    assert np.allclose(lp.A.T @ z, 0.0) and float(lp.b @ z) == -1.0


def test_random_programs_are_strictly_feasible_in_primal_and_dual():
    for cones, nvar in E.MIXES.values():
        for seed in (1, 2, 3):
            p = E.random_program(cones, nvar, seed)
            assert np.isfinite(K.barrier_value(p.cones, p.b - p.A @ p.x0))                       # Primal: x₀ streng zulässig
            r = S.solve_pd(p, tol=1e-9)
            assert r.status == S.OPTIMAL and r.dres < 1e-9                                       # Dual: ein Optimum existiert, Aᵀz = −c erreichbar


def test_random_programs_are_deterministic_in_the_seed():
    a, b, c = E.random_program(E.MIXES["PSD: zwei 3×3"][0], 6, 5), E.random_program(E.MIXES["PSD: zwei 3×3"][0], 6, 5), E.random_program(E.MIXES["PSD: zwei 3×3"][0], 6, 6)
    assert np.array_equal(a.A, b.A) and not np.array_equal(a.A, c.A)


@pytest.mark.parametrize("kappa", [10.0, 100.0])
def test_an_automorphism_keeps_the_optimum_and_maps_the_start(kappa):
    base = E.random_program((("s", 3), ("q", 4)), 5, 7)
    new, (x, s, z) = E.automorphism(base, kappa, 3)
    assert new.c is base.c and new.A.shape == base.A.shape and not np.allclose(new.A, base.A)  # c bleibt, A und b werden abgebildet
    ref = S.solve_pd(base, tol=1e-9)
    res = S.solve_pd(new, tol=1e-9, start=(x, s, z))
    assert ref.status == S.OPTIMAL and res.status == S.OPTIMAL and res.value == pytest.approx(ref.value, rel=1e-5)
    # der mitgenommene Start liegt im Inneren, und die Zielgröße sᵀz bleibt (𝒯 und 𝒯⁻ᵀ heben sich auf)
    x0, s0, z0 = S.initial_point(base)
    assert s @ z == pytest.approx(s0 @ z0, rel=1e-9)
    bl = C.blocks(new.cones)
    assert all(C.margin(k, m, s[o:o + d]) > 0 and C.margin(k, m, z[o:o + d]) > 0 for k, m, o, d in bl)


def test_the_automorphism_has_the_requested_condition():
    rng = np.random.default_rng(0)
    for kappa in (1.0, 100.0, 1e4):
        assert np.linalg.cond(E._soc_map(4, kappa, rng)) == pytest.approx(kappa, rel=1e-6)
        assert np.linalg.cond(E._psd_map(3, kappa, rng)) == pytest.approx(kappa, rel=1e-6)      # X ↦ TXTᵀ hat die Eigenwerte dᵢdⱼ: Kondition (d_max/d_min)² = κ


def test_the_nt_iteration_count_does_not_depend_on_the_automorphism_when_the_start_is_mapped():
    """Die Invarianz: gleicher Start (abgebildet), gleiche Iterationen bis auf die Rundung (Band ±2) bei mäßiger Verzerrung."""
    counts = []
    for seed in range(6):
        base = E.random_program((("s", 3),) * 2, 6, 2000 + seed)
        ref = S.solve_pd(base, tol=1e-8)
        new, start = E.automorphism(base, 100.0, seed)
        res = S.solve_pd(new, tol=1e-8, start=start)
        if ref.status == S.OPTIMAL and res.status == S.OPTIMAL:
            counts.append(abs(ref.iterations - res.iterations))
    assert len(counts) >= 4 and max(counts) <= 2
