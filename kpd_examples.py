"""Feste Beispiele und Zufallsprogramme (alle von Hand nachrechenbar oder aus Stück 3 und 4 übernommen).

  outlier_problem        das Programm aus Stück 3, bei dem der Barrierenkern 190,4 bei „Lücke 1,7·10⁻¹⁰“ meldet (Optimum 1,0·10⁻⁶)
  gapex_problem          Stück 4: Beispiel mit Lücke 1 ohne Slater-Punkt, für ε > 0 mit Wert −1
  nonattained_problem    Stück 4: [[x₁, 1], [1, x₂]] ⪰ 0, Infimum 0 wird nie angenommen
  weak_problem           Stück 4: [[x, 1], [1, 0]] ⪰ 0 ist schwach unzulässig (der Abstand zum Kegel geht gegen null, wird nie null)
  lp_infeasible_problem  x ≥ 1 und x ≤ 0 (Farkas-Zertifikat existiert)
  random_program         streng zulässiges Primal und Dual durch Konstruktion (b = A x₀ + s₀, c = −Aᵀ z₀)
  automorphism           dasselbe Programm, mit einem Automorphismus der Kegel umgeschrieben (PSD: X ↦ T X Tᵀ, Lorentz: Lorentz-Transformation), samt mitgenommenem Start"""

from __future__ import annotations

import math

import numpy as np

import kegel_kern as K
import kpd_cones as C
import kpd_solver as S


def outlier_problem() -> K.ConeProblem:
    """min (x⁴ / min(x, y))² auf [0,1; 3]² als Kegelprogramm (Epigraph-Übersetzung aus Stück 3): Variablen x, y, t1 ≥ x², t2 ≤ min(x, y), t3 ≥ t1²/t2, t4 ≥ t3²; Ziel t4; Optimum 1,0·10⁻⁶."""
    A = np.zeros((15, 6))
    A[0, 0], A[1, 0], A[2, 1], A[3, 1] = -1.0, 1.0, -1.0, 1.0            # Kasten: x − 0,1 ≥ 0, 3 − x ≥ 0, y − 0,1 ≥ 0, 3 − y ≥ 0
    A[4, 2] = A[5, 2] = -1.0                                              # t1 ≥ x²: (t1 + 1, t1 − 1, 2x) im Lorentz-Kegel
    A[6, 0] = -2.0
    A[7, 0], A[7, 3] = -1.0, 1.0                                          # t2 ≤ x, t2 ≤ y
    A[8, 1], A[8, 3] = -1.0, 1.0
    A[9, 3] = A[9, 4] = -1.0                                              # t3 ≥ t1²/t2: (t3 + t2, t3 − t2, 2 t1)
    A[10, 3], A[10, 4] = 1.0, -1.0
    A[11, 2] = -2.0
    A[12, 5] = A[13, 5] = -1.0                                            # t4 ≥ t3²: (t4 + 1, t4 − 1, 2 t3)
    A[14, 4] = -2.0
    b = np.array([-0.1, 3.0, -0.1, 3.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0])
    c = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    cones = (("l", 1),) * 4 + (("q", 3),) + (("l", 1),) * 2 + (("q", 3), ("q", 3))
    x0 = np.array([1.55, 1.55, 3.4025000000000003, 0.55, 22.049102272727275, 487.1629110331871])
    return K.ConeProblem(c, A, b, cones, x0)


OUTLIER_OPTIMUM = 1.0e-6


def sdp_problem(M0: np.ndarray, Ms: list, c, x0) -> K.ConeProblem:
    """Matrixform in Kegelform: s = svec(M0) − A x mit A = −[svec(M_j)] (Spalten)."""
    A = -np.stack([K.svec(M) for M in Ms], axis=1)
    return K.ConeProblem(np.array(c, float), A, K.svec(M0), (("s", M0.shape[0]),), np.array(x0, float))


def gapex_problem(eps: float) -> K.ConeProblem:
    """min x₁ mit [[ε, x₁, 0], [x₁, x₂, 0], [0, 0, x₁ + 1]] ⪰ 0: für ε > 0 streng zulässig mit Wert −1, bei ε = 0 Primal 0 und Dual −1 (Lücke 1)."""
    M0 = np.diag([eps, 0.0, 1.0])
    M1 = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    M2 = np.diag([0.0, 1.0, 0.0])
    return sdp_problem(M0, [M1, M2], [1.0, 0.0], [0.0, 1.0])


def nonattained_problem(c=(1.0, 0.0)) -> K.ConeProblem:
    """min c₁x₁ + c₂x₂ mit [[x₁, 1], [1, x₂]] ⪰ 0. Mit c = (1, 0): Infimum 0, nie erreicht; mit c = (1, 1): Optimum 2 bei (1, 1)."""
    return sdp_problem(np.array([[0.0, 1.0], [1.0, 0.0]]), [np.diag([1.0, 0.0]), np.diag([0.0, 1.0])], list(c), [2.0, 2.0])


def weak_problem() -> K.ConeProblem:
    """min x mit [[x, 1], [1, 0]] ⪰ 0: nie erfüllt, der kleinste Eigenwert (x − √(x² + 4))/2 ≈ −1/x geht gegen 0."""
    return sdp_problem(np.array([[0.0, 1.0], [1.0, 0.0]]), [np.array([[1.0, 0.0], [0.0, 0.0]])], [1.0], [0.0])


def lp_infeasible_problem() -> K.ConeProblem:
    """x ≥ 1 und x ≤ 0 als s = b − A x ≥ 0 mit A = (−1; 1), b = (−1; 0): kein x; Farkas-Zertifikat z = (1, 1): Aᵀz = 0, bᵀz = −1 < 0."""
    return K.ConeProblem(np.array([1.0]), np.array([[-1.0], [1.0]]), np.array([-1.0, 0.0]), (("l", 2),), np.array([0.0]))


# ----------------------------------------------------------------------------- Zufallsprogramme

def interior(kind: str, m: int, rng: np.random.Generator) -> np.ndarray:
    if kind == "l":
        return np.exp(rng.normal(0.0, 1.0, m))
    if kind == "q":
        r = rng.normal(size=m - 1)
        return np.concatenate([[np.linalg.norm(r) + np.exp(rng.normal())], r])
    G = rng.normal(size=(m, m))
    return K.svec(G @ G.T / m + np.exp(rng.normal()) * np.eye(m))


def random_program(cones, nvar: int, seed: int) -> K.ConeProblem:
    """Streng zulässiges Primal und Dual: x₀, s₀, z₀ ∈ int K gezogen, b = A x₀ + s₀, c = −Aᵀ z₀ (das Optimum existiert)."""
    rng = np.random.default_rng(seed)
    N = sum(K.cone_dim(k, m) for k, m in cones)
    A = rng.normal(size=(N, nvar))
    x0 = rng.normal(size=nvar)
    s0 = np.concatenate([interior(k, m, rng) for k, m in cones])
    z0 = np.concatenate([interior(k, m, rng) for k, m in cones])
    return K.ConeProblem(-A.T @ z0, A, A @ x0 + s0, tuple(cones), x0)


MIXES = {
    "Orthant: 12 Zeilen": ((("l", 12),), 5),
    "Lorentz: drei Kegel der Größe 4": ((("q", 4),) * 3, 6),
    "PSD: zwei 3×3": ((("s", 3),) * 2, 6),
    "gemischt: Orthant 4, Lorentz 4, PSD 3": ((("l", 4), ("q", 4), ("s", 3)), 7),
}


# ----------------------------------------------------------------------------- Kegelautomorphismen

def _psd_map(k: int, kappa: float, rng: np.random.Generator) -> np.ndarray:
    U, _ = np.linalg.qr(rng.normal(size=(k, k)))
    d = np.logspace(0, math.log10(kappa) / 2.0, k) if kappa > 1 else np.ones(k)
    T = U @ np.diag(d) @ U.T
    return C.op_matrix(k, lambda X: T @ X @ T.T)


def _soc_map(m: int, kappa: float, rng: np.random.Generator) -> np.ndarray:
    r = math.log(kappa) / 4.0                                             # P(w) = 2wwᵀ − J hat die Kondition e^(4r)
    e = rng.normal(size=m - 1)
    e /= np.linalg.norm(e)
    w = np.concatenate([[math.cosh(r)], math.sinh(r) * e])
    return 2.0 * np.outer(w, w) - np.diag(np.concatenate([[1.0], -np.ones(m - 1)]))


def automorphism(prob: K.ConeProblem, kappa: float, seed: int):
    """(Programm, Start): A' = 𝒯A, b' = 𝒯b, c' = c; der Start des Ausgangsprogramms wird mit 𝒯 bzw. 𝒯⁻ᵀ mitgenommen (s' = 𝒯 s, z' = 𝒯⁻ᵀ z).
    Optimalwert und Kegel bleiben gleich; ein Verfahren, das unter Automorphismen invariant ist, braucht dann dieselben Iterationen."""
    rng = np.random.default_rng(seed)
    bl = C.blocks(prob.cones)
    M = np.eye(prob.A.shape[0])
    for kind, m, off, d in bl:
        if kind == "s":
            M[off:off + d, off:off + d] = _psd_map(m, kappa, rng)
        elif kind == "q":
            M[off:off + d, off:off + d] = _soc_map(m, kappa, rng)
    x, s, z = S.initial_point(prob)
    new = K.ConeProblem(prob.c, M @ prob.A, M @ prob.b, prob.cones, prob.x0)
    return new, (x, M @ s, np.linalg.solve(M.T, z))
