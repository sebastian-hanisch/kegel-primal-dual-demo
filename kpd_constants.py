"""Konstanten: Regler-Grenzen, feste Stichproben, Farben und Presets."""

import kpd_forms as A

DEFAULT_FORM = "ball_soc"
N_MIN, N_MAX, DEFAULT_N = 8, 40, 20
STRETCH_MIN, STRETCH_MAX, DEFAULT_STRETCH = 1.2, 6.0, 4.0
ANGLE_MIN, ANGLE_MAX, DEFAULT_ANGLE = 0, 90, 45
SEED_MAX, DEFAULT_SEED = 999999, 35
KAPPAS = (1, 100, 10000, 1000000)                      # Verzerrung des Kegelautomorphismus in Experiment 3
DEFAULT_KAPPA = 100
SWEEP_SEEDS = tuple(range(100000, 100003))             # feste Wolken für die Sweeps (drei): nie ein einzelner Seed als Befund

TOL = 1e-8                                             # Genauigkeit des primal-dualen Laufs (relativ)
MAX_ITER = 100
N_SWEEP = (4, 10, 20, 40, 80)                          # Punktzahlen der Kugel (Lorentz-Form) in Experiment 4, ν = 2n
D_SWEEP = (2, 3, 4, 6)                                 # Dimensionen der allgemeinen Ellipse (PSD-Block 2d)
RANDOM_PROGRAMS = 12                                  # Zufallsprogramme je Kegelmischung in Experiment 1
INVARIANCE_PROGRAMS = 10                               # Zufallsprogramme je Stufe κ (zwei PSD-Blöcke 3×3)
IDENTITY_POINTS = 200                                  # Zufallspunkte je Kegel für die Identitäten der Skalierung
SLATER_EPS = (0.1, 1e-3, 1e-5)                         # Lückenbeispiel
DEFAULT_SOLVER_SEED = 1000                             # Seed der Zufallsprogramme

SCALING_NAMES = {"vector": "LP-Formel wörtlich (Koordinaten)", "jordan": "Jordan-Produkt (ohne Skalierung)", "nt": "Nesterov–Todd-Skalierung"}
SCALING_SHORT = {"vector": "wörtlich", "jordan": "Jordan", "nt": "NT"}
STATUS_NAMES = {"optimal": "gelöst", "max_iter": "Schrittgrenze erreicht", "stalled": "steht still (Schrittlänge 0)", "left_cone": "verlässt den Kegel", "linalg": "Matrix singulär", "nan": "Rechengrenze (nicht endlich)"}

STATUS_SHORT = {"optimal": "gelöst", "max_iter": "Schrittgrenze", "stalled": "steht still", "left_cone": "verlässt Kegel", "linalg": "singulär", "nan": "Rechengrenze"}      # kurz genug für eine Kennzahl

COLORS = {"vector": "#c0504d", "jordan": "#4c78a8", "nt": "#2F6B65", "barrier": "#888888"}
COLOR_OK = "#2F6B65"
COLOR_WARN = "#b08a2e"
COLOR_BAD = "#b34a4a"

PRESETS = {
    "Standardfall: Kugel (Lorentz-Kegel)": {"form": DEFAULT_FORM, "n": DEFAULT_N, "stretch": DEFAULT_STRETCH, "angle": DEFAULT_ANGLE, "seed": DEFAULT_SEED, "kappa": DEFAULT_KAPPA},
    "Der Würfel: ein LP": {"form": "cube", "n": DEFAULT_N, "stretch": DEFAULT_STRETCH, "angle": DEFAULT_ANGLE, "seed": DEFAULT_SEED, "kappa": DEFAULT_KAPPA},
    "Dieselbe Kugel als SDP": {"form": "ball_psd", "n": DEFAULT_N, "stretch": DEFAULT_STRETCH, "angle": DEFAULT_ANGLE, "seed": DEFAULT_SEED, "kappa": DEFAULT_KAPPA},
    "Allgemeine Ellipse (PSD)": {"form": "ellipse_psd", "n": DEFAULT_N, "stretch": DEFAULT_STRETCH, "angle": DEFAULT_ANGLE, "seed": DEFAULT_SEED, "kappa": DEFAULT_KAPPA},
    "Die schwierige Form": {"form": "ellipse_axis", "n": DEFAULT_N, "stretch": DEFAULT_STRETCH, "angle": DEFAULT_ANGLE, "seed": DEFAULT_SEED, "kappa": DEFAULT_KAPPA},
    "Stark verzerrt: κ = 10⁴": {"form": "ball_psd", "n": DEFAULT_N, "stretch": DEFAULT_STRETCH, "angle": DEFAULT_ANGLE, "seed": DEFAULT_SEED, "kappa": 10000},
}
PRESET_HELP = {
    "Standardfall: Kugel (Lorentz-Kegel)": "20 Punkte, in die Länge gezogen und um 45 Grad gedreht; die kleinste Kugel als Lorentz-Kegel: der wörtliche LP-Schritt steht still, Jordan und NT lösen sie in 8 bis 11 Iterationen.",
    "Der Würfel: ein LP": "Ein reines LP: alle drei Schritte sind derselbe, weil das Jordan-Produkt im Orthant komponentenweise ist.",
    "Dieselbe Kugel als SDP": "Dieselbe Kugel als 3×3-Matrizen im PSD-Kegel: Matrixprodukte, die nicht kommutieren; der Vektor-Schritt bricht ab, Jordan und NT lösen sie.",
    "Allgemeine Ellipse (PSD)": "Die allgemeine Ellipse mit PSD-Kegel braucht etwa doppelt so viele Iterationen wie die Kugel, bleibt aber unter den Newton-Schritten der Barriere.",
    "Die schwierige Form": "Die achsenparallele Ellipse als Lorentz-Kegel (t ≥ 1/p mit t ≈ 80, Dualwerte bis 3·10³): der NT-Lauf endet an der Rechengenauigkeit (Abweichung etwa 2·10⁻⁵), der Jordan-Schritt löst sie in 22 Iterationen.",
    "Stark verzerrt: κ = 10⁴": "Experiment 3 mit Verzerrung 10⁴: die NT-Iterationszahl bleibt gleich, aber auf dem verzerrten Programm gelingen weniger Läufe, beim Jordan-Schritt mehr.",
}
assert set(PRESETS) == set(PRESET_HELP) and all(p["form"] in A.FORMS and p["kappa"] in KAPPAS for p in PRESETS.values())
