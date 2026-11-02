"""Abbildungen: Achsen gesperrt, Spuren stimmen mit den Zahlen überein, Diagramme bauen an den Rändern."""

import pytest

import kpd_constants as C
import kpd_evaluation as ev
import kpd_forms as F
import kpd_visualization as V


def _locked(fig):
    axes = [v for k, v in fig.layout.to_plotly_json().items() if k.startswith(("xaxis", "yaxis"))]
    return bool(axes) and all(ax.get("fixedrange") is True for ax in axes)


@pytest.fixture(scope="module")
def a():
    return ev.analyse(ev.Settings())


def test_convergence_chart_has_one_line_per_direction_and_a_cross_for_each_failed_run(a):
    fig = V.build_convergence(a.runs)
    lines = [t for t in fig.data if t.mode == "lines+markers"]
    assert [t.name for t in lines] == [C.SCALING_NAMES[s] for s in ("vector", "jordan", "nt")] and _locked(fig) and fig.layout.yaxis.type == "log"
    assert [len(t.x) for t in lines] == [len(a.runs[s].history) for s in ("vector", "jordan", "nt")]
    crosses = [t for t in fig.data if t.mode == "markers"]
    assert len(crosses) == 1 and crosses[0].name.endswith(C.STATUS_NAMES["stalled"]) and crosses[0].marker.symbol == "x"            # nur der wörtliche LP-Schritt endet nicht gelöst
    assert all(y > 0 for t in fig.data for y in t.y)                                                                                    # logarithmische Achse: keine Nullen


def test_convergence_chart_without_failures_has_no_cross():
    cube = ev.analyse(ev.Settings(form="cube"))
    fig = V.build_convergence(cube.runs)
    assert not [t for t in fig.data if t.mode == "markers"] and len(fig.data) == 3


def test_variants_chart_is_a_heatmap_with_the_iteration_counts_in_the_cells(a):
    rows = ev.variant_rows(ev.Settings(n=10))
    fig = V.build_variants(rows)
    h = fig.data[0]
    assert _locked(fig) and list(h.y) == [F.FORM_NAMES[f] for f in F.FORMS] and len(h.x) == 3
    assert list(h.z[0]) == [1, 1, 1] and h.z[2][0] == 0 and h.z[2][2] == 1                                                                   # Würfel: alle gelöst; Kugel: wörtlich nicht, NT ja
    assert all(("Iter." in c) == bool(z) for zr, tr in zip(h.z, h.text) for z, c in zip(zr, tr)) and list(h.x) == ["wörtlich", "Jordan", "NT"]
    assert h.text[2][0] == "steht still<br>(6)" or h.text[2][0].startswith("steht still<br>(")


def test_commutator_chart_has_two_log_axes():
    res = ev.commutator_rows(ev.Settings(n=10))
    fig = V.build_commutator(res)
    assert _locked(fig) and len(fig.data) == 2 and fig.layout.yaxis.type == "log" and fig.layout.yaxis2.type == "log"
    assert len(fig.data[0].x) == len(res["rows"]) and min(fig.data[0].y) > 0


def test_invariance_chart_has_the_iterations_left_and_the_solved_share_right():
    rows = ev.invariance_rows(kappas=(1, 100), count=3)
    fig = V.build_invariance(rows)
    assert _locked(fig) and len(fig.data) == 4 and fig.layout.xaxis.type == "log" and fig.layout.yaxis2.range == (0, 105)
    assert list(fig.data[1].y) == [100.0 * r["solved"] / r["count"] for r in rows if r["scaling"] == "jordan"]


def test_nu_chart_compares_nt_jordan_and_the_barrier_in_both_panels():
    rows = ev.nu_rows(ns=(4, 6), seeds=(100000,))
    drows = ev.dim_rows(ds=(2, 3), seeds=(100000,))
    fig = V.build_nu(rows, drows)
    assert _locked(fig) and len(fig.data) == 6 and fig.layout.yaxis.rangemode == "tozero" and fig.layout.xaxis.type == "log"
    assert [t.name for t in fig.data[:3]] == [C.SCALING_NAMES["nt"], C.SCALING_NAMES["jordan"], "Barriere: Newton-Schritte (Stück 5)"]
    assert list(fig.data[0].x) == [r["nu"] for r in rows] and list(fig.data[3].x) == [2, 3]


def test_certificate_chart_shows_gap_and_both_residuals():
    fig = V.build_certificate(ev.certificate_experiment())
    assert _locked(fig) and [t.name for t in fig.data] == ["Lücke sᵀz", "Primalresiduum", "Dualresiduum"] and fig.layout.yaxis.type == "log"
    assert all(y > 0 for t in fig.data for y in t.y)


def test_charts_survive_a_failed_nt_run():
    hard = ev.analyse(ev.Settings(form="ellipse_axis"))
    fig = V.build_convergence(hard.runs)
    assert len([t for t in fig.data if t.mode == "markers"]) >= 1
