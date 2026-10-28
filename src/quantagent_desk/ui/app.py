"""Streamlit Bloomberg-style multi-panel risk terminal."""

from __future__ import annotations

import html
from datetime import datetime, timezone
from uuid import uuid4

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

API_URL = "http://localhost:8000"
USE_LOCAL = False

BOOKS = ["FIRM", "RATES_DESK", "FX_DESK", "EQ_DESK"]
SUGGESTIONS = [
    ("Firm VaR / ES", "Show me firm-wide VaR and ES"),
    ("2008 GFC", "Run the 2008 GFC stress"),
    ("AI unwind", "Generate an AI bubble unwind stress and show impact"),
    ("Stagflation", "Generate a stagflation shock stress"),
    ("What is ES?", "What is Expected Shortfall?"),
    ("Limits", "Show limit utilization"),
]
NODE_CODES = {
    "guardrail": "GRD",
    "router": "RTE",
    "planner": "PLN",
    "retrieve": "RAG",
    "risk": "RSK",
    "stress": "STR",
    "limits": "LIM",
    "rag": "DOC",
    "critic": "CRT",
}

THEME = """
<style>
  .stApp {
    background: #070707;
    color: #e6e1d6;
    font-family: Consolas, "Cascadia Mono", ui-monospace, monospace;
  }
  [data-testid="stHeader"] { background: transparent; }
  [data-testid="stDecoration"], footer { display: none; }
  [data-testid="stMainBlockContainer"] {
    padding-top: 0.7rem;
    padding-bottom: 0.45rem;
    max-width: 1520px;
  }
  [data-testid="stMainBlockContainer"] [data-testid="stVerticalBlock"] {
    gap: 0.45rem !important;
  }
  [data-testid="stSidebar"] {
    background: #0c0c0c;
    border-right: 1px solid #2a2a2a;
  }
  [data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 0 !important;
    border-color: #2a2a2a !important;
    background: #0d0d0d !important;
  }
  .stButton > button {
    border-radius: 0;
    letter-spacing: 0.08em;
    font-family: Consolas, ui-monospace, monospace;
  }
  [data-testid="stTextInput"] input {
    border: 1px solid #ff9f1a !important;
    border-radius: 0 !important;
    background: #0c0c0c !important;
    color: #ffd7a1 !important;
    font-family: Consolas, ui-monospace, monospace !important;
  }
  [data-testid="stBaseButton-secondary"] {
    border-radius: 0;
    border-color: #3a3328;
    background: #12110e;
    color: #e6e1d6;
    font-family: Consolas, ui-monospace, monospace;
    letter-spacing: 0.04em;
  }
  [data-testid="stBaseButton-primary"] {
    border-radius: 0;
    background: #ff9f1a;
    color: #14120e;
    font-family: Consolas, ui-monospace, monospace;
    letter-spacing: 0.08em;
  }
  [data-testid="stChatMessage"] { background: transparent; }
  div[data-testid="stMarkdown"]:has(.qad-tight) { margin-bottom: -0.15rem; }

  .qad-mast {
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
    gap: 1rem;
    border-bottom: 2px solid #ff9f1a;
    padding-bottom: 0.4rem;
    margin-bottom: 0.55rem;
  }
  .qad-brand { display: flex; align-items: baseline; flex-wrap: wrap; gap: 0.15rem 0; }
  .qad-logo {
    background: #ff9f1a;
    color: #14120e;
    font-weight: 700;
    letter-spacing: 0.08em;
    padding: 0.08rem 0.38rem;
    margin-right: 0.55rem;
  }
  .qad-title {
    color: #ff9f1a;
    letter-spacing: 0.16em;
    font-size: 1.2rem;
    font-weight: 700;
  }
  .qad-sub {
    color: #8d887c;
    letter-spacing: 0.18em;
    font-size: 0.68rem;
    margin-left: 0.75rem;
  }
  .qad-clock {
    color: #e6e1d6;
    letter-spacing: 0.08em;
    font-size: 0.78rem;
    white-space: nowrap;
  }
  .qad-kpis {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 8px;
    margin: 0.15rem 0 0.55rem;
  }
  .qad-kpi {
    background: #101010;
    border: 1px solid #2a2a2a;
    border-top: 2px solid #ff9f1a;
    padding: 0.4rem 0.65rem 0.45rem;
    min-width: 0;
  }
  .qad-kpi-label {
    color: #8d887c;
    font-size: 0.66rem;
    letter-spacing: 0.14em;
  }
  .qad-kpi-value {
    color: #ff9f1a;
    font-size: 1.28rem;
    font-weight: 700;
    line-height: 1.1;
    font-variant-numeric: tabular-nums;
  }
  .qad-kpi-sub {
    color: #6f6a60;
    font-size: 0.66rem;
    letter-spacing: 0.05em;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .qad-kpi-value.ok, .ok { color: #3dd68c; }
  .qad-kpi-value.warn, .warn { color: #ffd15c; }
  .qad-kpi-value.breach, .breach, .down { color: #ff5c4d; }
  .up { color: #3dd68c; }
  .qad-factors {
    display: flex;
    flex-wrap: wrap;
    gap: 0.35rem 1.05rem;
    align-items: baseline;
    border-bottom: 1px solid #2a2a2a;
    padding: 0.15rem 0 0.45rem;
    margin: -0.15rem 0 0.45rem;
    font-size: 0.78rem;
  }
  .qad-factor {
    color: #e6e1d6;
    font-variant-numeric: tabular-nums;
  }
  .qad-factor b {
    color: #ff9f1a;
    font-weight: 700;
    letter-spacing: 0.08em;
    margin-right: 0.35rem;
  }
  .qad-tape {
    display: flex;
    flex-wrap: wrap;
    gap: 0.35rem 0.85rem;
    align-items: center;
    margin: 0 0 0.65rem;
    font-size: 0.78rem;
  }
  .qad-tag {
    font-size: 0.64rem;
    letter-spacing: 0.12em;
    font-weight: 700;
    padding: 0.05rem 0.32rem;
    border: 1px solid currentColor;
  }
  .qad-panel-head {
    display: flex;
    align-items: baseline;
    gap: 0.45rem;
    border-bottom: 1px solid #2a2a2a;
    padding-bottom: 0.28rem;
    margin: 0 0 0.5rem;
  }
  .qad-fn {
    color: #14120e;
    background: #ff9f1a;
    font-size: 0.64rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    padding: 0.02rem 0.28rem;
  }
  .qad-panel-name {
    color: #f3efe6;
    letter-spacing: 0.16em;
    font-size: 0.75rem;
    font-weight: 700;
  }
  .qad-hint {
    margin-left: auto;
    color: #6f6a60;
    font-size: 0.64rem;
    letter-spacing: 0.08em;
  }
  .qad-chips { display: flex; flex-wrap: wrap; gap: 6px; margin: 0.25rem 0 0.45rem; }
  .qad-chip {
    border: 1px solid #3a3328;
    background: #16120c;
    color: #ffd7a1;
    padding: 0.08rem 0.4rem;
    font-size: 0.74rem;
  }
  .qad-chip b { color: #ff9f1a; font-weight: 700; margin-right: 0.3rem; }
  .qad-pair { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin: 0.35rem 0; }
  .qad-stat {
    border: 1px solid #2a2a2a;
    background: #101010;
    padding: 0.35rem 0.5rem;
  }
  .qad-stat-label { color: #8d887c; font-size: 0.64rem; letter-spacing: 0.12em; }
  .qad-stat-value {
    font-size: 1.2rem;
    font-weight: 700;
    font-variant-numeric: tabular-nums;
    color: #ff9f1a;
  }
  .qad-limit { margin: 0 0 0.55rem; }
  .qad-limit-top {
    display: flex;
    justify-content: space-between;
    font-size: 0.75rem;
    letter-spacing: 0.06em;
    margin-bottom: 0.18rem;
  }
  .qad-limit-top span:first-child { color: #e6e1d6; }
  .qad-track {
    position: relative;
    height: 8px;
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
  }
  .qad-mark {
    position: absolute;
    left: 80%;
    top: -2px;
    bottom: -2px;
    width: 1px;
    background: #ffd15c;
  }
  .qad-fill { height: 100%; }
  .qad-fill.ok { background: #3dd68c; }
  .qad-fill.warn { background: #ffd15c; }
  .qad-fill.breach { background: #ff5c4d; }
  .qad-status {
    display: flex;
    flex-wrap: wrap;
    gap: 0.35rem 1.15rem;
    border-top: 1px solid #ff9f1a;
    margin-top: 0.35rem;
    padding-top: 0.35rem;
    color: #8d887c;
    font-size: 0.68rem;
    letter-spacing: 0.08em;
  }
  .qad-status .hot { color: #ff9f1a; }
  .qad-muted { color: #6f6a60; }
  @media (max-width: 900px) {
    .qad-kpis { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .qad-sub { display: none; }
    .qad-title { font-size: 0.92rem; letter-spacing: 0.1em; white-space: nowrap; }
    .qad-clock { font-size: 0.66rem; }
  }
</style>
"""


def inject_theme() -> None:
    st.markdown(THEME, unsafe_allow_html=True)


def panel_head(code: str, title: str, hint: str = "") -> None:
    hint_html = f'<span class="qad-hint">{html.escape(hint)}</span>' if hint else ""
    st.markdown(
        '<div class="qad-tight">'
        f'<div class="qad-panel-head"><span class="qad-fn">{html.escape(code)}</span>'
        f'<span class="qad-panel-name">{html.escape(title)}</span>{hint_html}</div></div>',
        unsafe_allow_html=True,
    )


def fmt_money(value: float, *, signed: bool = False) -> str:
    sign = ""
    if signed:
        if value > 0:
            sign = "+"
        elif value < 0:
            sign = "-"
    mag = abs(float(value))
    if mag >= 1_000_000_000:
        body = f"${mag / 1_000_000_000:.2f}bn"
    elif mag >= 1_000_000:
        body = f"${mag / 1_000_000:.1f}mm"
    elif mag >= 1_000:
        body = f"${mag / 1_000:.1f}k"
    else:
        body = f"${mag:,.0f}"
    return f"{sign}{body}"


def util_tone(value: float) -> str:
    if value >= 1:
        return "breach"
    if value >= 0.8:
        return "warn"
    return "ok"


def short_id(value: str | None) -> str:
    text = str(value or "")
    return text[:8] if text else "—"


def call_api(method: str, path: str, **kwargs):
    """Prefer live API; fall back to in-process engines (Streamlit Cloud / offline)."""
    global USE_LOCAL
    if not USE_LOCAL:
        try:
            url = f"{API_URL}{path}"
            if method.upper() == "GET":
                response = requests.get(url, timeout=3, **kwargs)
            else:
                response = requests.post(url, timeout=15, **kwargs)
            response.raise_for_status()
            return response.json()
        except Exception:
            USE_LOCAL = True

    try:
        from quantagent_desk.ui import backend_local as local

        path_only = path.split("?")[0]
        params = kwargs.get("params") or {}
        if "?" in path:
            from urllib.parse import parse_qs, urlparse

            qs = parse_qs(urlparse(path).query)
            for key, values in qs.items():
                params[key] = values[0] if len(values) == 1 else values
        body = kwargs.get("json") or {}

        if path_only.endswith("/market/snapshot"):
            return local.market_snapshot()
        if path_only.endswith("/dashboard/summary") or "/dashboard/summary" in path_only:
            return local.dashboard_summary(params.get("portfolio_id", "FIRM"))
        if path_only.endswith("/stress/historical") or path_only.rstrip("/").endswith("historical"):
            return local.list_historical_scenarios()
        if "/stress/run" in path_only:
            return local.run_stress(
                params.get("scenario_id", "2008_GFC"),
                params.get("portfolio_id", "FIRM"),
            )
        if path_only.endswith("/query") or path_only.rstrip("/").endswith("query"):
            return local.query_desk(
                body.get("query", ""),
                body.get("portfolio_id", "FIRM"),
                body.get("session_id"),
            )
        return None
    except Exception as exc:
        st.error(f"Backend error: {exc}")
        return None


def stream_api(path: str, json_body: dict):
    """Stream events from live API SSE endpoint or fall back to in-process stream_query_desk."""
    global USE_LOCAL
    import json

    if not USE_LOCAL:
        try:
            url = f"{API_URL}{path}"
            with requests.post(url, json=json_body, stream=True, timeout=60) as response:
                response.raise_for_status()
                for line in response.iter_lines(decode_unicode=True):
                    if line and line.startswith("data: "):
                        data_str = line[6:].strip()
                        if data_str:
                            yield json.loads(data_str)
            return
        except Exception:
            USE_LOCAL = True

    try:
        from quantagent_desk.ui import backend_local as local

        yield from local.stream_query_desk(
            query=json_body.get("query", ""),
            portfolio_id=json_body.get("portfolio_id", "FIRM"),
            session_id=json_body.get("session_id"),
        )
    except Exception as exc:
        yield {"event": "error", "message": str(exc)}


def init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid4())
    if "stress_result" not in st.session_state:
        st.session_state.stress_result = None


def on_book_change() -> None:
    st.session_state.stress_result = None


def render_sidebar() -> None:
    with st.sidebar:
        panel_head("FN", "KEYS")
        st.markdown(
            "**01 RISK** — VaR and expected shortfall by desk\n\n"
            "**02 LIMITS** — utilization against hard limits\n\n"
            "**03 STRESS** — historical scenario lab\n\n"
            "**04 DESK** — multi-agent command line"
        )
        st.divider()
        st.markdown(
            "**COLOR**\n\n"
            "Amber figures are the primary risk numbers. "
            "Green is inside limit. Yellow is a warning at 80%. "
            "Red is a breach, or a stress loss."
        )
        st.divider()
        st.caption("The command line on the desk panel routes to risk, stress, limits, or methodology.")


_FACTOR_ORDER = ["SPX", "VIX", "USD_10Y", "EURUSD", "WTI", "IG_OAS"]
_FACTOR_LABELS = {
    "SPX": "SPX",
    "VIX": "VIX",
    "USD_10Y": "USD 10Y",
    "EURUSD": "EURUSD",
    "WTI": "WTI",
    "IG_OAS": "IG OAS",
}
_SHOCK_SPOT = {
    "SPX": "SPX",
    "USD_10Y": "USD_10Y",
    "CDX_IG": "IG_OAS",
    "VIX": "VIX",
    "WTI": "WTI",
    "EURUSD": "EURUSD",
}


def _format_level(key: str, value: float) -> str:
    if key == "USD_10Y":
        return f"{value * 100:.2f}%"
    if key == "IG_OAS":
        return f"{value * 10_000:.0f} bp"
    if key == "EURUSD":
        return f"{value:.4f}"
    if key == "VIX":
        return f"{value:.2f}"
    if key in {"SPX", "WTI"}:
        return f"{value:,.2f}"
    return f"{value:.4g}"


def render_masthead() -> None:
    clock = datetime.now(timezone.utc).strftime("%d %b %Y   %H:%M UTC")
    st.markdown(
        f"""
        <div class="qad-mast">
          <div class="qad-brand">
            <span class="qad-logo">QAD</span>
            <span class="qad-title">QUANTAGENT DESK</span>
            <span class="qad-sub">RISK TERMINAL</span>
          </div>
          <div class="qad-clock">{html.escape(clock)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_factor_strip(market: dict | None) -> None:
    levels = (market or {}).get("levels") or {}
    if not levels:
        return
    labels = (market or {}).get("labels") or _FACTOR_LABELS
    parts = []
    for key in _FACTOR_ORDER:
        raw = levels.get(key)
        if raw is None:
            continue
        parts.append(
            f"<span class='qad-factor'><b>{html.escape(str(labels.get(key, key)))}</b>"
            f"{html.escape(_format_level(key, float(raw)))}</span>"
        )
    as_of = str((market or {}).get("as_of") or "")
    pinned = str((market or {}).get("requested_as_of") or "")
    origin = str((market or {}).get("source") or "").upper()
    stamp = f"AS OF {as_of}"
    if pinned and pinned != as_of:
        stamp = f"{stamp} · PINNED {pinned}"
    parts.append(
        f"<span class='qad-muted'>{html.escape(stamp)} · {html.escape(origin)}</span>"
    )
    st.markdown(f"<div class='qad-factors'>{''.join(parts)}</div>", unsafe_allow_html=True)


def render_command_bar():
    book_col, refresh_col, context_col = st.columns([2, 1, 3], vertical_alignment="bottom")
    with book_col:
        portfolio_id = st.selectbox(
            "BOOK",
            BOOKS,
            key="book",
            on_change=on_book_change,
        )
    with refresh_col:
        st.button("REFRESH", use_container_width=True)
    return portfolio_id, context_col


def render_kpis(data: dict | None) -> None:
    if not data or "snapshot" not in data:
        tiles = [
            ("VAR 95", "—", "NO FEED", ""),
            ("VAR 99", "—", "NO FEED", ""),
            ("ES 97.5", "—", "NO FEED", ""),
            ("MAX UTIL", "—", "NO FEED", ""),
        ]
    else:
        snap = data["snapshot"]
        util = data.get("utilization") or {}
        max_util = max(util.values()) if util else 0.0
        calc = short_id(snap.get("calculation_id"))
        method = str(snap.get("methodology") or "Historical")
        method_label = "HIST FACTORS" if method.lower().startswith("historical factors") else method.upper()
        tiles = [
            ("VAR 95", fmt_money(snap.get("var_95", 0)), f"1D {method_label}", ""),
            ("VAR 99", fmt_money(snap.get("var_99", 0)), f"CALC {calc}", ""),
            ("ES 97.5", fmt_money(snap.get("es_97_5", 0)), data.get("name") or data.get("portfolio_id", ""), ""),
            ("MAX UTIL", f"{max_util:.0%}", "PEAK LIMIT", util_tone(max_util)),
        ]
    cells = []
    for label, value, sub, tone in tiles:
        value_class = f"qad-kpi-value {tone}".strip()
        cells.append(
            "<div class='qad-kpi'>"
            f"<div class='qad-kpi-label'>{html.escape(label)}</div>"
            f"<div class='{value_class}'>{html.escape(str(value))}</div>"
            f"<div class='qad-kpi-sub'>{html.escape(str(sub))}</div>"
            "</div>"
        )
    st.markdown(f"<div class='qad-kpis'>{''.join(cells)}</div>", unsafe_allow_html=True)


def render_tape(data: dict | None) -> None:
    breaches = (data or {}).get("breaches") or []
    if not breaches:
        st.markdown(
            "<div class='qad-tape'><span class='qad-tag ok'>CLEAR</span>"
            "<span>No limit warnings on this book</span></div>",
            unsafe_allow_html=True,
        )
        return
    parts = []
    for breach in breaches:
        tone = "breach" if breach.get("severity") == "Breach" else "warn"
        current = fmt_money(breach.get("current_value", 0))
        limit = fmt_money(breach.get("limit_value", 0))
        parts.append(
            f"<span class='qad-tag {tone}'>{html.escape(str(breach.get('severity', '')).upper())}</span>"
            f"<span>{html.escape(str(breach.get('limit_type', '')))}</span>"
            f"<span>{html.escape(current)}</span>"
            f"<span class='qad-muted'>/ {html.escape(limit)}</span>"
        )
    st.markdown(f"<div class='qad-tape'>{''.join(parts)}</div>", unsafe_allow_html=True)


def _desk_frame(data: dict) -> pd.DataFrame:
    snap = data["snapshot"]
    rows = data.get("children") or [
        {
            "portfolio_id": data.get("portfolio_id", ""),
            "name": data.get("name") or data.get("portfolio_id", ""),
            "var_95": snap.get("var_95", 0),
            "es_97_5": snap.get("es_97_5", 0),
            "n_positions": None,
        }
    ]
    frame = pd.DataFrame(rows)
    frame["DESK"] = frame["name"] if "name" in frame else frame.get("portfolio_id")
    frame["ID"] = frame["portfolio_id"] if "portfolio_id" in frame else ""
    frame["VAR 95"] = frame["var_95"].map(lambda v: fmt_money(v or 0))
    frame["ES 97.5"] = frame["es_97_5"].map(lambda v: fmt_money(v or 0))
    if "n_positions" in frame:
        frame["N"] = frame["n_positions"].map(lambda v: "—" if v is None else int(v))
    else:
        frame["N"] = "—"
    return frame[["ID", "DESK", "VAR 95", "ES 97.5", "N"]]


def render_risk(data: dict | None) -> None:
    panel_head("01", "RISK", "VAR / ES")
    if not data or "snapshot" not in data:
        st.caption("Risk feed unavailable.")
        return
    if data.get("error"):
        st.error(data["error"])
        return

    frame = _desk_frame(data)
    st.dataframe(frame, hide_index=True, height=160)

    chart_source = data.get("children") or [
        {
            "name": data.get("name") or data.get("portfolio_id", "BOOK"),
            "var_95": data["snapshot"].get("var_95", 0),
            "es_97_5": data["snapshot"].get("es_97_5", 0),
        }
    ]
    chart = pd.DataFrame(chart_source)
    fig = go.Figure(
        data=[
            go.Bar(name="VAR 95", x=chart["name"], y=chart["var_95"], marker_color="#FF9F1A"),
            go.Bar(name="ES 97.5", x=chart["name"], y=chart["es_97_5"], marker_color="#8EB6FF"),
        ]
    )
    fig.update_layout(
        barmode="group",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Consolas, monospace", color="#C8C2B4", size=11),
        margin=dict(l=4, r=4, t=4, b=4),
        height=140,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        bargap=0.28,
        xaxis=dict(showgrid=False),
        yaxis=dict(gridcolor="#2a2a2a", tickformat="$.3s", title=None),
    )
    fig.update_traces(marker_line_width=0)
    st.plotly_chart(
        fig,
        width="stretch",
        theme=None,
        config={"displayModeBar": "hover", "displaylogo": False},
    )
    if not data.get("children"):
        st.caption("Leaf book — no child desks under this node.")


def render_limits(data: dict | None) -> None:
    panel_head("02", "LIMITS", "80% WARN · 100% BREACH")
    util = (data or {}).get("utilization") or {}
    if not util:
        st.caption("No utilization on this book.")
        return
    rows = []
    for name, value in util.items():
        tone = util_tone(value)
        width = max(0.0, min(float(value) * 100, 100))
        label = html.escape(
            {
                "VaR_95": "VAR 95",
                "VaR_99": "VAR 99",
                "ES_97_5": "ES 97.5",
                "Peak_PFE": "PEAK PFE",
                "CVA": "CVA",
            }.get(str(name), str(name).replace("_", " ").upper())
        )
        rows.append(
            "<div class='qad-limit'>"
            "<div class='qad-limit-top'>"
            f"<span>{label}</span><span class='{tone}'>{value:.0%}</span>"
            "</div>"
            "<div class='qad-track'>"
            "<div class='qad-mark'></div>"
            f"<div class='qad-fill {tone}' style='width:{width:.1f}%'></div>"
            "</div></div>"
        )
    st.markdown("".join(rows), unsafe_allow_html=True)
    st.caption("Yellow tick marks the 80% warning line.")


def _shock_chips(shocks: list, levels: dict | None = None) -> None:
    chips = []
    spots = levels or {}
    for shock in shocks or []:
        factor_name = str(shock.get("factor", ""))
        factor = html.escape(factor_name)
        value = float(shock.get("value", 0) or 0)
        shock_type = shock.get("shock_type", "relative")
        if shock_type == "relative":
            shown = f"{value:+.0%}" if abs(value) >= 0.1 else f"{value:+.1%}"
        elif abs(value) >= 1:
            shown = f"{value:+.0f}"
        else:
            shown = f"{value * 10_000:+.0f} bp"
        spot_key = _SHOCK_SPOT.get(factor_name)
        spot_html = ""
        if spot_key and spots.get(spot_key) is not None:
            spot_html = (
                f" <span class='qad-muted'>{html.escape(_format_level(spot_key, float(spots[spot_key])))}</span>"
            )
        chips.append(f"<span class='qad-chip'><b>{factor}</b>{html.escape(shown)}{spot_html}</span>")
    if chips:
        st.markdown(f"<div class='qad-chips'>{''.join(chips)}</div>", unsafe_allow_html=True)


def render_stress(portfolio_id: str, market: dict | None = None) -> None:
    panel_head("03", "STRESS", portfolio_id)
    scenarios = call_api("GET", "/stress/historical") or []
    if not scenarios:
        st.caption("No historical scenarios loaded.")
        return

    names = [s["name"] for s in scenarios]
    choice = st.selectbox("SCENARIO", names, key="stress_scenario")
    selected = next(s for s in scenarios if s["name"] == choice)
    st.caption(selected.get("description") or "")
    _shock_chips(selected.get("shocks") or [], (market or {}).get("levels") or {})

    if st.button("RUN STRESS", type="primary", use_container_width=True):
        scenario_id = selected.get("scenario_id") or "2008_GFC"
        result = call_api("POST", f"/stress/run?scenario_id={scenario_id}&portfolio_id={portfolio_id}")
        st.session_state.stress_result = {
            "result": result,
            "portfolio_id": portfolio_id,
            "name": choice,
        }

    stored = st.session_state.get("stress_result")
    if not stored:
        st.caption("Pick a scenario and run it. P&L stays on the panel until you change books.")
        return
    if stored.get("portfolio_id") != portfolio_id:
        st.caption("Last stress run was for another book.")
        return
    result = stored.get("result") or {}
    if result.get("error"):
        st.error(result["error"])
        return

    pnl = float(result.get("pnl_impact") or 0)
    var_impact = float(result.get("var_impact") or 0)
    pnl_tone = "down" if pnl < 0 else ("up" if pnl > 0 else "")
    st.markdown(
        "<div class='qad-pair'>"
        "<div class='qad-stat'><div class='qad-stat-label'>P&L IMPACT</div>"
        f"<div class='qad-stat-value {pnl_tone}'>{html.escape(fmt_money(pnl, signed=True))}</div></div>"
        "<div class='qad-stat'><div class='qad-stat-label'>VAR IMPACT</div>"
        f"<div class='qad-stat-value'>{html.escape(fmt_money(var_impact, signed=True))}</div></div>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.caption(f"{stored.get('name', '')}  ·  CALC {short_id(result.get('calculation_id'))}")

    breakdown = result.get("breakdown") or {}
    if breakdown:
        ranked = sorted(breakdown.items(), key=lambda item: abs(item[1]), reverse=True)[:6]
        show = pd.DataFrame(
            [{"INSTRUMENT": name, "IMPACT": fmt_money(value, signed=True)} for name, value in ranked]
        )
        st.dataframe(show, hide_index=True, height=180)
        if len(breakdown) > 6:
            st.caption(f"Top 6 of {len(breakdown)} instruments by absolute impact.")


def render_desk_history(hide_suggestions: bool = False):
    head, clear = st.columns([5, 1], vertical_alignment="top")
    with head:
        panel_head("04", "DESK", "LANGGRAPH")
    with clear:
        if st.button("CLEAR", use_container_width=True, disabled=not st.session_state.messages):
            st.session_state.messages = []
            st.session_state.session_id = str(uuid4())
            st.rerun()

    history = st.container(height=176)
    with history:
        if not st.session_state.messages and not hide_suggestions:
            st.caption("Ask the desk. Every figure comes back with a calculation id.")
            left, right = st.columns(2)
            for index, (label, query) in enumerate(SUGGESTIONS):
                column = left if index % 2 == 0 else right
                if column.button(label, key=f"suggest_{index}", use_container_width=True):
                    st.session_state.queued_prompt = query
                    st.rerun()
        else:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    for warning in message.get("warnings") or []:
                        st.warning(warning)

    with st.form("desk_command", clear_on_submit=True, border=False):
        ask_col, send_col = st.columns([5, 1], vertical_alignment="bottom")
        typed = ask_col.text_input(
            "ASK",
            placeholder="ASK>  VaR, stress, limits, methodology",
            label_visibility="collapsed",
        )
        submitted = send_col.form_submit_button("SEND", type="primary", use_container_width=True)
    command = typed.strip() if submitted and typed.strip() else None
    return history, command


def run_turn(history, prompt: str, portfolio_id: str) -> None:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with history:
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            status_box = st.status("Routing the desk...", expanded=True)
            text_placeholder = st.empty()
            accumulated = ""
            final_resp = None

            for event in stream_api(
                "/query/stream",
                {
                    "query": prompt,
                    "portfolio_id": portfolio_id,
                    "session_id": st.session_state.session_id,
                    "include_charts": True,
                },
            ):
                event_type = event.get("event")
                if event_type == "node_update":
                    code = NODE_CODES.get(event.get("node"), "RUN")
                    status_box.write(f"`{code}`  {event.get('status', '')}")
                elif event_type == "token":
                    accumulated += event.get("chunk", "")
                    text_placeholder.markdown(accumulated + "▌")
                elif event_type == "final":
                    final_resp = event.get("response")
                elif event_type == "error":
                    st.error(f"Streaming error: {event.get('message')}")

            status_box.update(label="Desk complete", state="complete", expanded=False)
            text_placeholder.markdown(accumulated or "_No answer returned._")
            warnings = (final_resp.get("warnings") if final_resp else []) or []
            for warning in warnings:
                st.warning(warning)
            st.session_state.messages.append(
                {"role": "assistant", "content": accumulated, "warnings": warnings}
            )


def render_status(data: dict | None, portfolio_id: str, source: str) -> None:
    calc = "—"
    if data and data.get("snapshot"):
        calc = short_id(data["snapshot"].get("calculation_id"))
    clock = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
    feed = "API" if source == "API" else "LOCAL"
    st.markdown(
        "<div class='qad-status'>"
        "<span class='hot'>QAD 0.1</span>"
        f"<span>{html.escape(portfolio_id)}</span>"
        f"<span>FEED {html.escape(feed)}</span>"
        f"<span>CALC {html.escape(calc)}</span>"
        f"<span>{html.escape(clock)}</span>"
        "<span>USD</span>"
        "</div>",
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="QuantAgent Desk",
        page_icon="QAD",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    inject_theme()
    init_state()
    render_sidebar()
    render_masthead()
    market = call_api("GET", "/market/snapshot") or {}
    render_factor_strip(market)
    portfolio_id, context_col = render_command_bar()

    data = call_api("GET", f"/dashboard/summary?portfolio_id={portfolio_id}")
    source = "LOCAL" if USE_LOCAL else "API"
    book_name = (data or {}).get("name") or portfolio_id
    factor_as_of = market.get("as_of")
    horizon = f"factors as of {factor_as_of}" if factor_as_of else "synthetic"
    with context_col:
        st.caption(f"{book_name}   ·   USD   ·   {horizon}   ·   feed {source}")

    render_kpis(data)
    render_tape(data)

    risk_col, limits_col = st.columns([1.35, 1], gap="small")
    with risk_col:
        with st.container(border=True):
            render_risk(data)
    with limits_col:
        with st.container(border=True):
            render_limits(data)

    queued = st.session_state.pop("queued_prompt", None)

    stress_col, desk_col = st.columns([1, 1.2], gap="small")
    with stress_col:
        with st.container(border=True):
            render_stress(portfolio_id, market)
    with desk_col:
        with st.container(border=True):
            history, command = render_desk_history(hide_suggestions=bool(queued))

    render_status(data, portfolio_id, source)

    prompt = command or queued
    if prompt:
        run_turn(history, prompt, portfolio_id)


main()
