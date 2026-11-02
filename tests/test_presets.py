"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen; Permalink verwirft ungültige Werte; die Zahlen der Hilfetexte."""

from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

import kpd_constants as C
import kpd_evaluation as ev
import kpd_forms as A
import kpd_presets as P
import kpd_solver as S

APP = str(Path(__file__).resolve().parent.parent / "app.py")
DEFAULT = "Standardfall: Kugel (Lorentz-Kegel)"


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_slider_bounds():
    for preset in C.PRESETS.values():
        assert preset["form"] in A.FORMS and preset["kappa"] in C.KAPPAS
        assert C.N_MIN <= preset["n"] <= C.N_MAX and C.STRETCH_MIN <= preset["stretch"] <= C.STRETCH_MAX
        assert C.ANGLE_MIN <= preset["angle"] <= C.ANGLE_MAX and 0 <= preset["seed"] <= C.SEED_MAX


def test_default_preset_equals_the_default_settings():
    assert ev.Settings(**C.PRESETS[DEFAULT]) == ev.Settings()


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_each_preset_uses_the_form_its_name_announces():
    forms = {name: p["form"] for name, p in C.PRESETS.items()}
    assert forms["Der Würfel: ein LP"] == "cube" and forms["Dieselbe Kugel als SDP"] == "ball_psd" and forms["Allgemeine Ellipse (PSD)"] == "ellipse_psd"
    assert forms["Die schwierige Form"] == "ellipse_axis" and forms[DEFAULT] == C.DEFAULT_FORM
    assert C.PRESETS["Stark verzerrt: κ = 10⁴"]["kappa"] == 10000 and C.PRESETS[DEFAULT]["kappa"] == C.DEFAULT_KAPPA


def test_the_help_texts_name_numbers_that_the_presets_produce():
    """Bänder statt Punktwerte, wo die Rundung von der CPU abhängt (README: Rechengrenze)."""
    std = ev.analyse(ev.Settings(**C.PRESETS[DEFAULT]))
    assert std.runs["vector"].status == S.STALLED and std.runs["jordan"].status == S.OPTIMAL and std.runs["nt"].status == S.OPTIMAL
    assert 8 <= std.runs["jordan"].iterations and std.runs["nt"].iterations <= 11 and "8 bis 11 Iterationen" in C.PRESET_HELP[DEFAULT]
    cube = ev.analyse(ev.Settings(**C.PRESETS["Der Würfel: ein LP"]))
    assert len({r.iterations for r in cube.runs.values()}) == 1 and all(r.status == S.OPTIMAL for r in cube.runs.values())
    sdp = ev.analyse(ev.Settings(**C.PRESETS["Dieselbe Kugel als SDP"]))
    assert sdp.runs["vector"].status != S.OPTIMAL and sdp.runs["jordan"].status == S.OPTIMAL and sdp.runs["nt"].status == S.OPTIMAL


def test_the_ellipse_help_compares_with_the_ball_and_with_the_barrier():
    ball = ev.analyse(ev.Settings(**C.PRESETS[DEFAULT])).nt.iterations
    ell = ev.analyse(ev.Settings(**C.PRESETS["Allgemeine Ellipse (PSD)"]))
    assert 1.5 <= ell.nt.iterations / ball <= 2.6 and ell.nt.iterations < ell.barrier.newton_steps and ell.nt.status == S.OPTIMAL


def test_the_hard_form_help_names_the_rounding_limit_and_the_jordan_rescue():
    a = ev.analyse(ev.Settings(**C.PRESETS["Die schwierige Form"]))
    t, z_max = a.nt.x[2], max(np.abs(a.nt.z).max(), np.abs(a.runs["jordan"].z).max())
    assert 70 <= t <= 90 and 1.5e3 <= z_max <= 5e3                                                   # „t ≈ 80, Dualwerte bis 3·10³“
    assert a.runs["jordan"].status == S.OPTIMAL and 18 <= a.runs["jordan"].iterations <= 30          # „in 22 Iterationen“ (Band: Rundung)
    assert a.nt.status == S.OPTIMAL or 1e-7 < ev.deviation(a, a.nt) < 1e-3                          # NT endet an der Rechengrenze (Abweichung um 2·10⁻⁵) oder löst gerade noch
    assert "2·10⁻⁵" in C.PRESET_HELP["Die schwierige Form"] and "22 Iterationen" in C.PRESET_HELP["Die schwierige Form"]


def test_the_distorted_preset_matches_experiment_three():
    rows = ev.invariance_rows(kappas=(1, 10000), count=C.INVARIANCE_PROGRAMS)
    by = {(r["kappa"], r["scaling"]): r for r in rows}
    assert by[(1, "nt")]["solved"] == C.INVARIANCE_PROGRAMS and by[(10000, "nt")]["solved"] < by[(1, "nt")]["solved"]               # weniger Läufe gelingen
    assert by[(10000, "jordan")]["solved"] >= by[(10000, "nt")]["solved"]                                                            # beim Jordan-Schritt mehr
    assert abs(by[(10000, "nt")]["median"] - by[(1, "nt")]["median"]) <= 3                                                           # die Iterationszahl des skalierten Schritts bleibt


def test_permalink_takes_valid_values_and_drops_invalid_ones():
    at = AppTest.from_file(APP, default_timeout=180)
    at.query_params["form"] = "ellipse_psd"
    at.query_params["n"] = "999"            # über der Grenze: wird begrenzt
    at.query_params["stretch"] = "nan"      # keine endliche Zahl: bleibt Standard
    at.query_params["angle"] = "abc"        # keine Zahl: bleibt Standard
    at.query_params["kappa"] = "7"          # kein erlaubter Wert: bleibt Standard
    at.run()
    assert not at.exception
    assert at.sidebar.selectbox(key="form_select").value == "ellipse_psd"
    assert at.sidebar.slider(key="n_slider").value == C.N_MAX
    assert at.sidebar.slider(key="stretch_slider").value == C.DEFAULT_STRETCH
    assert at.sidebar.slider(key="angle_slider").value == C.DEFAULT_ANGLE
    assert at.session_state["kappa_select"] == C.DEFAULT_KAPPA
    bad = AppTest.from_file(APP, default_timeout=180)
    bad.query_params["form"] = "dreieck"
    bad.query_params["kappa"] = "10000"
    bad.run()
    assert not bad.exception and bad.sidebar.selectbox(key="form_select").value == C.DEFAULT_FORM and bad.session_state["kappa_select"] == 10000
