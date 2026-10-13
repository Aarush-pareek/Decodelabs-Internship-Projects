"""Small HTML/CSS building blocks for the dashboard.

These are pure functions that return strings (no Streamlit import), so they can
be unit-tested. All dynamic text is HTML-escaped because class labels and
column names come from user-supplied datasets.
"""

from __future__ import annotations

from html import escape

CUSTOM_CSS = """
<style>
.block-container { padding-top: 2.2rem; padding-bottom: 4rem; max-width: 1180px; }
h1 { font-size: 1.85rem !important; letter-spacing: -0.01em; }
h2 { font-size: 1.3rem !important; margin-top: 0.4rem; }
section[data-testid="stSidebar"] .stButton > button { width: 100%; }

.section-kicker { color: #52514e; text-transform: uppercase; letter-spacing: .08em;
                  font-size: .72rem; font-weight: 600; margin: 2.4rem 0 .15rem 0; }
.section-sub { color: #52514e; font-size: .92rem; margin: -.4rem 0 1rem 0; }

.card { border: 1px solid #e4e3df; border-radius: 12px; padding: .95rem 1.1rem; background: #ffffff;
        min-height: 7.4rem; }
.card .label { color: #52514e; font-size: .72rem; text-transform: uppercase; letter-spacing: .06em;
               font-weight: 600; }
.card .value { color: #0b0b0b; font-size: 1.55rem; font-weight: 700; line-height: 1.25; margin-top: .15rem;
               overflow-wrap: anywhere; }
.card .hint { color: #52514e; font-size: .8rem; margin-top: .3rem; line-height: 1.35; }

.pill { display: inline-block; padding: .12rem .6rem; border-radius: 999px; font-size: .8rem;
        font-weight: 600; border: 1px solid transparent; }
.pill.ok { background: #e6f4ec; color: #0f5a33; border-color: #bfe0cd; }
.pill.idle { background: #f1f0ed; color: #52514e; border-color: #e4e3df; }
.pill.warn { background: #fdf3dc; color: #6b4a00; border-color: #f0dba5; }

.result-card { border: 1px solid #e4e3df; border-left-width: 6px; border-radius: 12px;
               padding: 1.2rem 1.4rem; background: #ffffff; }
.result-card .label { color: #52514e; font-size: .72rem; text-transform: uppercase;
                      letter-spacing: .06em; font-weight: 600; }
.result-card .class-name { font-size: 2.1rem; font-weight: 700; line-height: 1.2; margin: .1rem 0 .9rem 0;
                           overflow-wrap: anywhere; }
.result-card .confidence { font-size: 1.6rem; font-weight: 700; }
.result-card .bar { height: 8px; background: #ecebe7; border-radius: 4px; margin-top: .5rem; overflow: hidden; }
.result-card .bar > span { display: block; height: 100%; }
</style>
"""


def section_header(number: int, title: str, subtitle: str = "") -> str:
    """Numbered section heading with an optional one-line description."""
    html = f'<div class="section-kicker">Section {number}</div>'
    html += f"<h2>{escape(title)}</h2>"
    if subtitle:
        html += f'<div class="section-sub">{escape(subtitle)}</div>'
    return html


def metric_card(label: str, value: str, hint: str = "") -> str:
    """A bordered card with a small label, a large value and an optional hint."""
    hint_html = f'<div class="hint">{escape(hint)}</div>' if hint else ""
    return (
        f'<div class="card"><div class="label">{escape(label)}</div>'
        f'<div class="value">{escape(value)}</div>{hint_html}</div>'
    )


def status_pill(text: str, kind: str = "idle") -> str:
    """Compact status badge. ``kind`` is one of ``ok``, ``idle`` or ``warn``."""
    if kind not in {"ok", "idle", "warn"}:
        raise ValueError(f"Unknown pill kind: {kind}")
    return f'<span class="pill {kind}">{escape(text)}</span>'


def prediction_card(label: str, confidence: float | None, color: str) -> str:
    """Large result card showing the predicted class and its confidence."""
    if confidence is None:
        confidence_html = '<div class="confidence">Not available</div>'
    else:
        percent = max(0.0, min(confidence, 1.0)) * 100
        confidence_html = (
            f'<div class="confidence">{percent:.1f}%</div>'
            f'<div class="bar"><span style="width:{percent:.1f}%; background:{escape(color)}"></span></div>'
        )
    return (
        f'<div class="result-card" style="border-left-color:{escape(color)}">'
        f'<div class="label">Predicted class</div>'
        f'<div class="class-name">{escape(label)}</div>'
        f'<div class="label">Prediction confidence</div>{confidence_html}</div>'
    )
