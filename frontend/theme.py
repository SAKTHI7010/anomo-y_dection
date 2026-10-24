"""Visual theme: CSS injection + shared Plotly styling for the dashboard."""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

PALETTE = {
    "bg": "#0b1020",
    "panel": "rgba(255,255,255,0.045)",
    "border": "rgba(148,163,184,0.18)",
    "text": "#e2e8f0",
    "muted": "#94a3b8",
    "cyan": "#22d3ee",
    "violet": "#a78bfa",
    "amber": "#fbbf24",
    "rose": "#fb7185",
    "green": "#34d399",
    "orange": "#fb923c",
}
SENSOR_COLORS = {"vibration": "#22d3ee", "torque": "#a78bfa", "temp_solid": "#fbbf24", "temp_liquid": "#fb7185"}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
html, body, [class*="css"]  { font-family: 'Inter', system-ui, sans-serif; }
.stApp {
  background:
    radial-gradient(1200px 600px at 8% -10%, rgba(34,211,238,0.10), transparent 60%),
    radial-gradient(900px 500px at 100% 0%, rgba(167,139,250,0.12), transparent 55%),
    linear-gradient(180deg, #0b1020 0%, #0a0f1d 100%);
}
header[data-testid="stHeader"] { background: transparent; }
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, rgba(15,23,42,0.96), rgba(11,16,32,0.98));
  border-right: 1px solid rgba(148,163,184,0.14);
}
.block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1400px; }
h1, h2, h3 { letter-spacing: -0.02em; }

.hero {
  padding: 1.4rem 1.6rem; border-radius: 20px; margin-bottom: 1.1rem;
  background: linear-gradient(135deg, rgba(34,211,238,0.14), rgba(167,139,250,0.14));
  border: 1px solid rgba(148,163,184,0.20);
  box-shadow: 0 10px 40px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.06);
  animation: fadeUp .6s ease both;
}
.hero h1 { margin: 0; font-size: 1.85rem; font-weight: 800;
  background: linear-gradient(90deg, #e2e8f0, #67e8f9 55%, #c4b5fd); -webkit-background-clip: text; background-clip: text; color: transparent; }
.hero p { margin: .35rem 0 0 0; color: #94a3b8; font-size: .95rem; }
.badge-row { margin-top: .7rem; display:flex; gap:.5rem; flex-wrap:wrap; }
.badge { font-size:.72rem; font-weight:600; padding:.22rem .65rem; border-radius:999px;
  border:1px solid rgba(148,163,184,0.28); color:#cbd5e1; background:rgba(15,23,42,.55); letter-spacing:.02em; }

.card {
  border-radius: 16px; padding: 1rem 1.1rem; background: rgba(255,255,255,0.045);
  border: 1px solid rgba(148,163,184,0.16); backdrop-filter: blur(10px);
  transition: transform .18s ease, border-color .18s ease, box-shadow .18s ease; animation: fadeUp .5s ease both;
}
.card:hover { transform: translateY(-3px); border-color: rgba(103,232,249,0.45); box-shadow: 0 12px 30px rgba(34,211,238,0.10); }
.card .k { font-size:.72rem; text-transform:uppercase; letter-spacing:.09em; color:#94a3b8; font-weight:600; }
.card .v { font-size:1.7rem; font-weight:700; color:#f1f5f9; margin-top:.15rem; font-family:'JetBrains Mono',monospace; }
.card .s { font-size:.78rem; color:#94a3b8; margin-top:.1rem; }

.state-pill { display:inline-block; padding:.45rem 1.1rem; border-radius:999px; font-weight:700; font-size:1.05rem;
  letter-spacing:.04em; border:1px solid; animation: pulse 2.6s ease-in-out infinite; }
.note { border-left: 3px solid #22d3ee; background: rgba(34,211,238,0.07); padding:.7rem 1rem; border-radius:8px;
  color:#cbd5e1; font-size:.86rem; margin: .4rem 0 1rem 0; }
.warn { border-left: 3px solid #fbbf24; background: rgba(251,191,36,0.08); padding:.7rem 1rem; border-radius:8px;
  color:#e2e8f0; font-size:.86rem; margin: .4rem 0 1rem 0; }
.legend-chip { display:inline-flex; align-items:center; gap:.4rem; margin-right:1rem; font-size:.78rem; color:#cbd5e1; }
.dot { width:.6rem; height:.6rem; border-radius:50%; display:inline-block; }

div[data-testid="stMetric"] { background: rgba(255,255,255,0.045); border:1px solid rgba(148,163,184,0.16);
  padding:.7rem .9rem; border-radius:14px; }
button[kind="primary"], .stButton>button { border-radius: 10px; font-weight:600; transition: all .15s ease; }
.stButton>button:hover { transform: translateY(-1px); box-shadow: 0 6px 18px rgba(34,211,238,0.25); border-color:#22d3ee; }
div[data-baseweb="tab-list"] { gap: .4rem; }
button[data-baseweb="tab"] { border-radius: 10px 10px 0 0; }

/* ── Force dark background on ALL selectboxes / dropdowns ── */
div[data-baseweb="select"] > div,
div[data-baseweb="select"] > div:hover,
div[data-baseweb="select"] > div:focus-within {
  background-color: #0f172a !important;
  border-color: rgba(148,163,184,0.30) !important;
  color: #e2e8f0 !important;
}
div[data-baseweb="select"] svg { fill: #94a3b8 !important; }

/* Dropdown option list panel */
ul[data-baseweb="menu"],
div[data-baseweb="popover"] > div,
div[role="listbox"] {
  background-color: #0f172a !important;
  border: 1px solid rgba(148,163,184,0.20) !important;
  box-shadow: 0 8px 32px rgba(0,0,0,0.5) !important;
}
/* Individual options */
li[role="option"],
div[role="option"] {
  background-color: #0f172a !important;
  color: #e2e8f0 !important;
}
li[role="option"]:hover,
div[role="option"]:hover,
li[aria-selected="true"],
div[aria-selected="true"] {
  background-color: rgba(34,211,238,0.12) !important;
  color: #22d3ee !important;
}

/* Number inputs */
div[data-baseweb="input"] > div,
div[data-baseweb="base-input"] {
  background-color: #0f172a !important;
  border-color: rgba(148,163,184,0.30) !important;
  color: #e2e8f0 !important;
}
input[type="number"], input[type="text"] {
  background-color: #0f172a !important;
  color: #e2e8f0 !important;
}

/* Slider track */
div[data-testid="stSlider"] div[role="slider"] { background: #22d3ee !important; }

/* Checkbox */
div[data-baseweb="checkbox"] > div:first-child {
  background-color: rgba(34,211,238,0.15) !important;
  border-color: rgba(34,211,238,0.55) !important;
}

@keyframes fadeUp { from { opacity: 0; transform: translateY(8px);} to { opacity: 1; transform: none;} }
@keyframes pulse { 0%,100% { box-shadow: 0 0 0 0 rgba(255,255,255,0.0);} 50% { box-shadow: 0 0 22px 0 rgba(255,255,255,0.12);} }
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str, badges: list[str] | None = None) -> None:
    b = "".join(f'<span class="badge">{x}</span>' for x in (badges or []))
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p><div class="badge-row">{b}</div></div>', unsafe_allow_html=True)


def card(label: str, value: str, sub: str = "", color: str | None = None) -> str:
    style = f' style="color:{color}"' if color else ""
    return f'<div class="card"><div class="k">{label}</div><div class="v"{style}>{value}</div><div class="s">{sub}</div></div>'


def note(text: str, kind: str = "note") -> None:
    st.markdown(f'<div class="{kind}">{text}</div>', unsafe_allow_html=True)


def legend(items: list[tuple[str, str]]) -> None:
    html = "".join(f'<span class="legend-chip"><span class="dot" style="background:{c}"></span>{t}</span>' for t, c in items)
    st.markdown(f"<div style='margin:.2rem 0 .6rem 0'>{html}</div>", unsafe_allow_html=True)


def style_fig(fig: go.Figure, height: int = 320, title: str | None = None) -> go.Figure:
    fig.update_layout(
        template="plotly_dark", height=height, margin=dict(l=10, r=10, t=44 if title else 14, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.35)",
        font=dict(family="Inter, sans-serif", color="#cbd5e1", size=12),
        title=dict(text=title, x=0.01, font=dict(size=14, color="#e2e8f0")) if title else None,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        hoverlabel=dict(bgcolor="#0f172a", font_size=12),
    )
    fig.update_xaxes(gridcolor="rgba(148,163,184,0.10)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.10)", zeroline=False)
    return fig
