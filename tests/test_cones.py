"""Die Algebra der Kegel (kpd_cones): Jordan-Produkt, Einselement, Schrittweite und Nesterov–Todd-Skalierung, von Hand und als Eigenschaft."""

import math

import numpy as np
import pytest

import kegel_kern as K
import kpd_cones as C
import kpd_examples as E


def test_svec_and_smat_agree_with_the_kernel():
    rng = np.random.default_rng(0)
    for k in (1, 2, 3, 5):
        X = rng.normal(size=(k, k))
        X = X + X.T
        assert np.allclose(C.svec(X), K.svec(X)) and np.allclose(C.smat(K.svec(X), k), X) and np.allclose(C.smat(C.svec(X), k), K.smat(K.svec(X), k))


def test_svec_by_hand():
    assert C.svec(np.array([[1.0, 2.0], [2.0, 3.0]])).tolist() == pytest.approx([1.0, 2.0 * math.sqrt(2.0), 3.0])


def test_the_basis_is_orthonormal_and_matches_svec():
    for k in (2, 3):
        B = C.basis(k)
        assert len(B) == k * (k + 1) // 2
        gram = np.array([[np.sum(X * Y) for Y in B] for X in B])
        assert np.allclose(gram, np.eye(len(B)))
        assert all(np.allclose(C.svec(X), np.eye(len(B))[i]) for i, X in enumerate(B))


def test_identity_elements_and_degree():
    assert C.identity("l", 3).tolist() == [1.0, 1.0, 1.0] and C.identity("q", 4).tolist() == [1.0, 0.0, 0.0, 0.0]
    assert np.allclose(C.smat(C.identity("s", 3), 3), np.eye(3))
    assert C.degree((("l", 5), ("q", 3), ("q", 4), ("s", 3))) == 5 + 1 + 1 + 3
    assert C.degree((("q", 3),), vector=True) == 3                      # im Vektor-Modus sind es alle Koordinaten


@pytest.mark.parametrize("kind,m", [("l", 4), ("q", 3), ("q", 6), ("s", 2), ("s", 4)])
def test_jordan_product_laws(kind, m):
    rng = np.random.default_rng(1)
    e = C.identity(kind, m)
    for _ in range(20):
        u, v = E.interior(kind, m, rng), E.interior(kind, m, rng)
        assert np.allclose(C.product(kind, m, e, u), u)                                 # e ∘ u = u
        assert np.allclose(C.product(kind, m, u, v), C.product(kind, m, v, u))          # kommutativ
        assert np.allclose(C.product_matrix(kind, m, u) @ v, C.product(kind, m, u, v))   # Matrix und Produkt stimmen überein
        assert C.product(kind, m, u, v).shape == (K.cone_dim(kind, m),)


def test_jordan_products_by_hand():
    assert C.product("l", 3, np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])).tolist() == [4.0, 10.0, 18.0]
    # Lorentz: u = (2; 1, 0), v = (3; 0, 1): u∘v = (uᵀv, u₀ v̄ + v₀ ū) = (6; 2·(0,1) + 3·(1,0)) = (6; 3, 2)
    assert C.product("q", 3, np.array([2.0, 1.0, 0.0]), np.array([3.0, 0.0, 1.0])).tolist() == [6.0, 3.0, 2.0]
    # PSD: U = diag(2, 1), V = [[1, 1], [1, 2]]: (UV + VU)/2 = [[2, 1,5], [1,5, 2]]
    out = C.smat(C.product("s", 2, C.svec(np.diag([2.0, 1.0])), C.svec(np.array([[1.0, 1.0], [1.0, 2.0]]))), 2)
    assert np.allclose(out, [[2.0, 1.5], [1.5, 2.0]])


def test_the_matrix_product_of_two_psd_matrices_is_not_symmetric_but_the_jordan_product_is():
    Sm, Z = np.diag([2.0, 1.0]), np.array([[1.0, 1.0], [1.0, 2.0]])
    assert not np.allclose(Sm @ Z, (Sm @ Z).T) and np.allclose(Sm @ Z - Z @ Sm, [[0.0, 1.0], [-1.0, 0.0]])
    J = C.smat(C.product("s", 2, C.svec(Sm), C.svec(Z)), 2)
    assert np.allclose(J, J.T) and np.allclose(J, (Sm @ Z + Z @ Sm) / 2.0)


def test_max_step_by_hand():
    assert C.max_step("l", 3, np.array([1.0, 2.0, 3.0]), np.array([-1.0, 1.0, -6.0])) == pytest.approx(0.5)      # min(1/1, 3/6)
    assert C.max_step("l", 2, np.array([1.0, 1.0]), np.array([1.0, 2.0])) == math.inf
    # Lorentz: u = (2; 0, 0), du = (−1; 1, 0): (2 − α)² − α² ≥ 0 ⇔ α ≤ 1
    assert C.max_step("q", 3, np.array([2.0, 0.0, 0.0]), np.array([-1.0, 1.0, 0.0])) == pytest.approx(1.0)
    assert C.max_step("q", 3, np.array([2.0, 0.0, 0.0]), np.array([1.0, 0.5, 0.0])) == math.inf
    assert C.max_step("q", 3, np.array([2.0, 0.0, 0.0]), np.array([-4.0, 0.0, 0.0])) == pytest.approx(0.5)      # u₀ wird null
    # PSD: U = I, dU = diag(−2, 1): kleinster Eigenwert −2, also α = 1/2
    assert C.max_step("s", 2, C.svec(np.eye(2)), C.svec(np.diag([-2.0, 1.0]))) == pytest.approx(0.5)
    assert C.max_step("s", 2, C.svec(np.eye(2)), C.svec(np.diag([1.0, 1.0]))) == math.inf


@pytest.mark.parametrize("kind,m", [("l", 3), ("q", 4), ("s", 3)])
def test_max_step_lands_on_the_boundary(kind, m):
    rng = np.random.default_rng(2)
    for _ in range(30):
        u = E.interior(kind, m, rng)
        du = rng.normal(size=K.cone_dim(kind, m))
        a = C.max_step(kind, m, u, du)
        if math.isinf(a):
            assert C.margin(kind, m, u + 100.0 * du) > 0
            continue
        assert abs(C.margin(kind, m, u + a * du)) < 1e-9 * (1.0 + np.abs(u).max())             # genau auf dem Rand
        assert C.margin(kind, m, u + 0.99 * a * du) > 0                                         # davor im Inneren


def test_nt_scaling_of_the_orthant_by_hand():
    Q, W, lam = C.nt_scaling("l", 1, np.array([2.0]), np.array([0.5]))
    assert W[0, 0] == pytest.approx(2.0) and lam[0] == pytest.approx(1.0) and Q[0, 0] == pytest.approx(4.0)       # W = √(s/z), λ = √(sz), Q = s/z


def test_nt_scaling_commutes_with_diagonal_psd_blocks_like_the_lp_scaling():
    s, z = np.diag([4.0, 1.0]), np.diag([1.0, 4.0])
    Q, W, lam = C.nt_scaling("s", 2, C.svec(s), C.svec(z))
    assert np.allclose(C.smat(lam, 2), np.diag([2.0, 2.0]))                                       # λ = √(s z) komponentenweise
    assert np.allclose(W @ C.svec(z), lam) and np.allclose(Q @ C.svec(z), C.svec(s))


@pytest.mark.parametrize("kind,m", [("l", 4), ("q", 3), ("q", 6), ("s", 2), ("s", 3), ("s", 5)])
def test_nt_scaling_identities(kind, m):
    rng = np.random.default_rng(3)
    for _ in range(100):
        s, z = E.interior(kind, m, rng), E.interior(kind, m, rng)
        Q, W, lam = C.nt_scaling(kind, m, s, z)
        assert np.allclose(Q @ z, s, rtol=1e-9, atol=1e-11)                                         # WᵀW z = s
        assert np.allclose(W @ z, lam, rtol=1e-9, atol=1e-11) and np.allclose(np.linalg.solve(W, s), lam, rtol=1e-8, atol=1e-10)
        assert np.allclose(W, W.T, atol=1e-12) and np.linalg.eigvalsh((W + W.T) / 2).min() > 0 and np.allclose(W @ W, Q, rtol=1e-8, atol=1e-10)
        assert lam @ lam == pytest.approx(s @ z, rel=1e-8)                                          # sᵀz = ‖λ‖²


def test_nt_scaling_of_the_lorentz_cone_survives_points_close_to_the_boundary():
    rng = np.random.default_rng(4)
    for _ in range(200):
        r, r2 = rng.normal(size=2), rng.normal(size=2)
        s = np.concatenate([[np.linalg.norm(r) * (1 + rng.uniform(1e-9, 1e-2)) + 1e-12], r])
        z = np.concatenate([[np.linalg.norm(r2) * (1 + rng.uniform(1e-9, 1e-2)) + 1e-12], r2])
        Q, W, lam = C.nt_scaling("q", 3, s, z)
        assert np.allclose(Q @ z, s, rtol=1e-6, atol=1e-9) and np.allclose(W @ W, Q, rtol=1e-6, atol=1e-9)


def test_the_scaled_point_changes_only_by_the_automorphism():
    """Unter s ↦ T s Tᵀ, z ↦ T⁻ᵀ z T⁻¹ hat λ dieselben Eigenwerte (die Skalierung ist mit dem Automorphismus verträglich)."""
    rng = np.random.default_rng(5)
    s, z = E.interior("s", 3, rng), E.interior("s", 3, rng)
    T = rng.normal(size=(3, 3)) + 3 * np.eye(3)
    S2, Z2 = T @ C.smat(s, 3) @ T.T, np.linalg.inv(T).T @ C.smat(z, 3) @ np.linalg.inv(T)
    _, _, lam = C.nt_scaling("s", 3, s, z)
    _, _, lam2 = C.nt_scaling("s", 3, C.svec(S2), C.svec(Z2))
    assert np.allclose(np.linalg.eigvalsh(C.smat(lam, 3)), np.linalg.eigvalsh(C.smat(lam2, 3)), rtol=1e-8)
