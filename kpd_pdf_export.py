"""PDF-Export der aktuellen Wolke mit den drei Richtungen (fpdf2). Kernschriften sind latin-1: Umlaute gehen, Sonderzeichen werden ersetzt."""

from __future__ import annotations

from fpdf import FPDF

import kpd_constants as C
import kpd_forms as A
from kpd_evaluation import Analysis, de, sci, status_text

_REPLACE = {"→": "->", "–": "-", "—": "-", "−": "-", "≤": "<=", "≥": ">=", "·": "*", "×": "x", "ν": "nu", "λ": "lambda", "κ": "kappa", "€": "EUR", "⁻": "-", "⁰": "0", "¹": "1", "²": "2", "³": "3",
            "⁴": "4", "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9", "„": '"', "“": '"', "ᵀ": "^T", "∘": "o", "√": "sqrt"}


def _s(text) -> str:
    out = str(text).replace("**", "").replace("`", "")
    for old, new in _REPLACE.items():
        out = out.replace(old, new)
    return out.encode("latin-1", "replace").decode("latin-1")


def build_pdf(a: Analysis) -> bytes:
    s = a.settings
    pdf = FPDF(orientation="L", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _s("Primal-dualer Schritt im Kegel"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, _s(f"Form: {A.FORM_NAMES[s.form]}; Wolke: {s.n} Punkte, Streckung {de(s.stretch, 1)}, um {s.angle} Grad gedreht, Seed {s.seed}. Referenzwert der Barriere: {de(a.reference, 6)}. Alles simuliert."),
                   new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    head, widths = ["Richtung", "Ausgang", "Iterationen", "Primal", "Dual", "Lücke c^T x + b^T z", "Abweichung"], [66, 56, 24, 34, 34, 40, 24]
    pdf.set_font("Helvetica", "B", 9)
    for h, w in zip(head, widths):
        pdf.cell(w, 6, _s(h), border=1)
    pdf.ln()
    pdf.set_font("Helvetica", "", 9)
    for sc in ("vector", "jordan", "nt"):
        r = a.runs[sc]
        for v, w in zip([C.SCALING_NAMES[sc], status_text(r.status), r.iterations, de(r.value, 6), de(r.dual_value, 6), sci(abs(r.certificate_gap)), sci(abs(r.value - a.reference))], widths):
            pdf.cell(w, 5.5, _s(v), border=1)
        pdf.ln()
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, _s("Die Barriere aus Stück 5 braucht auf derselben Wolke " + f"{a.barrier.newton_steps} Newton-Schritte in {a.barrier.rounds} Runden. Quelle: kegel-primal-dual-demo, Sebastian Hanisch."),
                   new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
