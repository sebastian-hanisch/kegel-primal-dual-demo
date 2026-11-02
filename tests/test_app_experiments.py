"""Pfade der App, die der Rauchtest nicht erreicht: jede Form, die fünf Experimente im Seiteninhalt, PDF, Eckfälle aller Regler zugleich (DEMO-PLAYBOOK §1d)."""

import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import kpd_constants as C
import kpd_forms as F

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _app(**state) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    for key, value in state.items():
        widget = at.sidebar.selectbox if key == "form_select" else at.sidebar.number_input if key == "seed_input" else at.sidebar.select_slider if key == "kappa_select" else at.sidebar.slider
        widget(key=key).set_value(value)
    if state:
        at.run()
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _texts(at):
    return " ".join(str(e.value) for e in list(at.success) + list(at.info) + list(at.warning) + list(at.error) + list(at.markdown))


def _count(text):
    return int(re.match(r"(\d+)", text).group(1))


def test_default_page_shows_the_metrics_and_all_experiment_verdicts():
    at = _app()
    values = {m.label: m.value for m in at.metric}
    assert 9 <= _count(values["NT-Schritt (Iterationen)"]) <= 13 and 6 <= _count(values["Jordan-Schritt (Iterationen)"]) <= 10 and values["LP-Formel wörtlich"] == C.STATUS_SHORT["stalled"]
    assert 25 <= _count(values["Barriere (Stück 5): Newton-Schritte"]) <= 40 and values["Zertifikat cᵀx + bᵀz"].endswith(("⁻⁷", "⁻⁸", "⁻⁹", "⁻¹⁰", "⁻¹¹", "⁻¹²"))
    text = _texts(at)
    for part in ("Für die Form **Kugel (Lorentz-Kegel)**", "Das Zertifikat des NT-Laufs", "sind alle drei Schritte derselbe", "Zufallsprogramme (streng zulässig durch Konstruktion)", "Von Hand: für S = diag(2, 1)",
                 "Identitäten der NT-Skalierung", "Zwei PSD-Blöcke 3×3", "Kugel mit n = 4 bis 80 Punkten", "**beide klammern das Optimum**", "Lückenbeispiel: Primal und Dual sind für jedes ε > 0"):
        assert part in text, part


def test_the_five_experiments_the_limits_table_and_the_formulation_are_on_the_page():
    at = _app()
    headings = [s.value for s in at.subheader]
    assert [h.split(":")[0] for h in headings if h.startswith("🔬")] == [f"🔬 Experiment {i}" for i in range(1, 6)] and "🚧 Wo die Annahmen enden" in headings
    assert [e.label for e in at.expander] == ["So funktioniert der Schritt", "📐 Mathematische Formulierung"]
    assert len(at.dataframe) >= 6 and not at.error


def test_pdf_button_produces_a_download():
    at = _app()
    at.button(key="pdf_make").click().run()
    assert not at.exception
    assert at.session_state["pdf_made"][1][:4] == b"%PDF"


@pytest.mark.parametrize("form", F.FORMS)
def test_every_form_renders_without_error(form):
    at = _app(form_select=form)
    assert not at.error
    assert any(f"Für die Form **{F.FORM_NAMES[form]}**" in s.value for s in list(at.success) + list(at.warning))


def test_the_hard_form_is_flagged_not_hidden():
    at = _app(form_select="ellipse_axis")
    assert any("kein Wert, dem man trauen sollte" in w.value for w in at.warning)


@pytest.mark.parametrize("edge", ["min", "max"])
def test_all_sliders_at_their_minimum_or_maximum_together(edge):
    pick = (lambda lo, hi: lo) if edge == "min" else (lambda lo, hi: hi)
    state = {"n_slider": pick(C.N_MIN, C.N_MAX), "stretch_slider": pick(C.STRETCH_MIN, C.STRETCH_MAX), "angle_slider": pick(C.ANGLE_MIN, C.ANGLE_MAX), "kappa_select": pick(C.KAPPAS[0], C.KAPPAS[-1])}
    at = _app(**state)
    assert not at.exception and not at.error


def test_the_slider_corners_work_in_every_form():
    """Regression aus Stück 4 (Eckfall-Test fand dort einen Absturz): Streckung 6, 45 Grad, 20 Punkte in jeder Form."""
    for form in F.FORMS:
        at = _app(form_select=form, stretch_slider=C.STRETCH_MAX)
        assert not at.error, form


@pytest.mark.parametrize("kappa", C.KAPPAS)
def test_every_distortion_level_renders_the_invariance_verdict(kappa):
    at = _app(kappa_select=kappa)
    text = _texts(at)
    assert "Zwei PSD-Blöcke 3×3" in text and not at.error
