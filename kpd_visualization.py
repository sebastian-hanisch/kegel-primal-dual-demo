"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

from __future__ import annotations

import math

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import kpd_constants as C
import kpd_forms as A
import kpd_solver as S


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=380, legend_below=False):
    """Legende über dem Diagramm; bei Diagrammen mit Unterüberschriften darunter (sonst liegt sie auf den Überschriften)."""
    legend = dict(orientation="h", yanchor="top", y=-0.22, x=0) if legend_below else dict(orientation="h", yanchor="bottom", y=1.02, x=0)
    fig.update_layout(height=height + (40 if legend_below else 0), margin=dict(l=10, r=10, t=50, b=10), legend=legend, plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def _positive(values, floor=1e-16):
    return [max(float(v), floor) for v in values]


def build_convergence(runs: dict, title: str = ""):
    """Die Lücke sᵀz je Iteration der drei Richtungen (logarithmisch); ein Kreuz markiert einen Lauf, der nicht gelöst endet."""
    fig = go.Figure()
    for sc in ("vector", "jordan", "nt"):
        r = runs[sc]
        its = [h["it"] for h in r.history]
        fig.add_trace(go.Scatter(x=its, y=_positive(h["gap"] for h in r.history), mode="lines+markers", name=C.SCALING_NAMES[sc], line=dict(color=C.COLORS[sc], width=3 if sc == "nt" else 2),
                                 hovertemplate="Iteration %{x}<br>sᵀz = %{y:.2e}<extra></extra>"))
        if r.status != S.OPTIMAL:
            fig.add_trace(go.Scatter(x=[its[-1]], y=_positive([r.history[-1]["gap"]]), mode="markers", marker=dict(symbol="x", size=14, color=C.COLOR_BAD, line=dict(width=2)),
                                     name=f"{C.SCALING_NAMES[sc]}: {C.STATUS_NAMES[r.status]}", showlegend=True))
    fig.update_xaxes(title_text="Iteration")
    fig.update_yaxes(title_text="Lücke sᵀz (logarithmisch)", type="log")
    return _base(fig)


def build_variants(rows: list):
    """Je Form und Richtung eine Zelle: grün gelöst (Iterationen), rot sonst (Ausgang und Iterationen in Klammern); Kurznamen, damit es auf dem Telefon lesbar bleibt."""
    forms = list(dict.fromkeys(r["form"] for r in rows))
    scalings = ["vector", "jordan", "nt"]
    z, text = [], []
    for form in forms:
        zr, tr = [], []
        for sc in scalings:
            r = next(x for x in rows if x["form"] == form and x["scaling"] == sc)
            ok = r["status"] == S.OPTIMAL
            zr.append(1 if ok else 0)
            tr.append(f"{r['iterations']} Iter." if ok else f"{C.STATUS_SHORT[r['status']]}<br>({r['iterations']})")
        z.append(zr)
        text.append(tr)
    fig = go.Figure(go.Heatmap(z=z, x=[C.SCALING_SHORT[s] for s in scalings], y=[A.FORM_NAMES[f] for f in forms], text=text, texttemplate="%{text}", textfont=dict(size=10), showscale=False,
                               colorscale=[[0, "rgba(179,74,74,0.55)"], [1, "rgba(47,107,101,0.6)"]], zmin=0, zmax=1, xgap=2, ygap=2, hoverinfo="skip"))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(side="top")
    return _base(fig, height=330)


def build_commutator(res: dict):
    """‖SZ − ZS‖ / (‖S‖‖Z‖) und die Lücke sᵀz je Iteration des NT-Laufs (beide logarithmisch)."""
    rows = res["rows"]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=[r["it"] for r in rows], y=_positive((r["ratio"] for r in rows), 1e-17), mode="lines+markers", name="‖SZ − ZS‖ / (‖S‖‖Z‖)", line=dict(color=C.COLOR_BAD, width=3)),
                  secondary_y=False)
    fig.add_trace(go.Scatter(x=[r["it"] for r in rows], y=_positive(r["gap"] for r in rows), mode="lines", name="Lücke sᵀz", line=dict(color=C.COLORS["barrier"], dash="dot")), secondary_y=True)
    fig.update_xaxes(title_text="Iteration")
    fig.update_yaxes(title_text="Nichtkommutativität", type="log", exponentformat="power", dtick=4, secondary_y=False)
    fig.update_yaxes(title_text="sᵀz", type="log", exponentformat="power", dtick=4, showgrid=False, secondary_y=True)     # feste Teilung: sonst richtet die rechte Achse ihre Marken krumm an der linken aus
    return _base(fig)


def build_invariance(rows: list):
    """Links Iterationen (Median der gelösten Läufe), rechts Anteil gelösten Läufe, über der Verzerrung κ."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Iterationen (Median der gelösten Läufe)", "gelöste Läufe in %"))
    for sc in ("jordan", "nt"):
        sub = [r for r in rows if r["scaling"] == sc]
        fig.add_trace(go.Scatter(x=[r["kappa"] for r in sub], y=[r["median"] for r in sub], mode="lines+markers", name=C.SCALING_NAMES[sc], line=dict(color=C.COLORS[sc], width=3 if sc == "nt" else 2)),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=[r["kappa"] for r in sub], y=[100.0 * r["solved"] / r["count"] for r in sub], mode="lines+markers", name=C.SCALING_NAMES[sc], showlegend=False,
                                 line=dict(color=C.COLORS[sc], width=3 if sc == "nt" else 2)), row=1, col=2)
    fig.update_xaxes(title_text="Verzerrung κ des Kegelautomorphismus", type="log")
    fig.update_yaxes(rangemode="tozero", row=1, col=1)
    fig.update_yaxes(range=[0, 105], row=1, col=2)
    return _base(fig, legend_below=True)


def build_nu(rows: list, drows: list):
    """Iterationen des primal-dualen Schritts gegen die Newton-Schritte der Barriere, über ν (Kugel) und über die Dimension (Ellipse)."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Kugel: ν = 2n", "allgemeine Ellipse: Dimension d"))
    for col, data, key in ((1, rows, "nu"), (2, drows, "d")):
        fig.add_trace(go.Scatter(x=[r[key] for r in data], y=[r["nt"] for r in data], mode="lines+markers", name=C.SCALING_NAMES["nt"], line=dict(color=C.COLORS["nt"], width=3), showlegend=col == 1), row=1, col=col)
        fig.add_trace(go.Scatter(x=[r[key] for r in data], y=[r["jordan"] for r in data], mode="lines+markers", name=C.SCALING_NAMES["jordan"], line=dict(color=C.COLORS["jordan"]), showlegend=col == 1),
                      row=1, col=col)
        fig.add_trace(go.Scatter(x=[r[key] for r in data], y=[r["newton"] for r in data], mode="lines+markers", name="Barriere: Newton-Schritte (Stück 5)", line=dict(color=C.COLORS["barrier"], dash="dot"),
                                 showlegend=col == 1), row=1, col=col)
    fig.update_xaxes(title_text="ν", type="log", tickvals=[r["nu"] for r in rows], ticktext=[str(r["nu"]) for r in rows], row=1, col=1)
    fig.update_xaxes(title_text="d", row=1, col=2)
    fig.update_yaxes(title_text="Iterationen bzw. Newton-Schritte", rangemode="tozero")
    return _base(fig, legend_below=True)


def build_certificate(res: dict):
    """Verlauf des primal-dualen Laufs für den Ausreißer-Fall: Lücke sᵀz und die beiden Residuen (logarithmisch)."""
    pd = res["pd"]
    fig = go.Figure()
    for key, name, color in (("gap", "Lücke sᵀz", C.COLORS["nt"]), ("pres", "Primalresiduum", C.COLORS["jordan"]), ("dres", "Dualresiduum", C.COLOR_WARN)):
        fig.add_trace(go.Scatter(x=[h["it"] for h in pd.history], y=_positive(h[key] for h in pd.history), mode="lines+markers", name=name, line=dict(color=color, width=3 if key == "gap" else 2)))
    fig.update_xaxes(title_text="Iteration")
    fig.update_yaxes(title_text="logarithmisch", type="log")
    return _base(fig, height=320)
