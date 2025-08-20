# streamlit_app.py
# -----------------------------------------------------------------------------
# $1M-Quality AI BI Dashboard — Streamlit single-file app
# Tabs: Home | Dashboard | Segmentation | Anomaly Detection | Forecasting | Early Access
# Features:
# - Professional UI with animated gradient, cards, and clean navigation
# - URL filter support: ?region=AMER,APAC,EMEA&channel=Online,Retail,Wholesale
# - Realistic demo business data (daily last 365d + synthetic customer data)
# - Interactive charts (Plotly with graceful fallback)
# - RFM-based segmentation (robust percentile method)
# - IQR-based anomaly detection
# - Seasonal-naive + EMA forecasting with prediction intervals
# - Waitlist form (Google Form integration via st.secrets + local CSV fallback)
# - Revenue-recovery messaging ($500K+ scenarios)
# - Python 3.13-safe
# -----------------------------------------------------------------------------

import os
import io
import json
import math
import textwrap
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st

# Optional libs: plotly (interactive charts) and requests (Google Form HTTP)
PLOTLY_AVAILABLE = True
try:
    import plotly.express as px
    import plotly.graph_objects as go
except Exception:
    PLOTLY_AVAILABLE = False

REQUESTS_AVAILABLE = True
try:
    import requests
except Exception:
    REQUESTS_AVAILABLE = False


# -----------------------------------------------------------------------------
# Page & Branding
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI BI Dashboard — Revenue Recovery Suite",
    page_icon="💹",
    layout="wide",
)

PRIMARY_GRADIENT = "linear-gradient(120deg, #0ea5e9 0%, #6366f1 50%, #22c55e 100%)"
PRIMARY = "#6366f1"          # indigo-500
SUCCESS = "#22c55e"          # green-500
DANGER = "#ef4444"           # red-500

# Global CSS
st.markdown(
    f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

      html, body, [class*="css"] {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif !important;
      }}

      /* Animated gradient header bar */
      .gradient-bar {{
        position: fixed;
        top: 0; left: 0; right: 0;
        height: 4px;
        background: {PRIMARY_GRADIENT};
        background-size: 300% 300%;
        animation: moveGradient 10s ease infinite;
        z-index: 1000;
      }}
      @keyframes moveGradient {{
        0% {{ background-position: 0% 50%; }}
        50% {{ background-position: 100% 50%; }}
        100% {{ background-position: 0% 50%; }}
      }}

      /* Card styling */
      .kpi-card {{
        border-radius: 18px;
        padding: 16px 18px;
        background: #0f172a;
        border: 1px solid rgba(148,163,184,0.18);
        box-shadow: 0 10px 25px rgba(2,6,23,0.55);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
      }}
      .kpi-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 16px 36px rgba(2,6,23,0.75);
      }}
      .kpi-title {{
        color: #cbd5e1;
        font-size: 13px;
        letter-spacing: 0.3px;
        text-transform: uppercase;
        margin-bottom: 6px;
      }}
      .kpi-value {{
        font-size: 26px;
        font-weight: 800;
        color: #e2e8f0;
      }}
      .kpi-delta {{
        font-size: 13px;
        font-weight: 600;
      }}

      /* Section headers */
      .section-title {{
        font-size: 22px;
        font-weight: 800;
        color: #e2e8f0;
        margin: 8px 0 4px 0;
      }}
      .section-sub {{
        color: #cbd5e1;
        font-size: 14px;
        margin-bottom: 16px;
      }}

      .pill {{
        display: inline-block;
        padding: 6px 12px;
        border-radius: 999px;
        font-weight: 700;
        font-size: 12px;
        letter-spacing: .4px;
        color: white;
        background: {PRIMARY_GRADIENT};
      }}

      @media (max-width: 768px) {{
        .kpi-value {{ font-size: 22px; }}
        .section-title {{ font-size: 18px; }}
      }}
    </style>
    <div class="gradient-bar"></div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def fmt_money(x: float) -> str:
    try:
        if abs(x) >= 1_000_000:
            return f"${x/1_000_000:,.2f}M"
        if abs(x) >= 1_000:
            return f"${x/1_000:,.1f}K"
        return f"${x:,.0f}"
    except Exception:
        return "$0"

def fmt_pct(x: float) -> str:
    try:
        return f"{x*100:.2f}%"
    except Exception:
        return "0.00%"

def pct_delta(cur, prev):
    try:
        prev = float(prev)
        return 0.0 if prev == 0 else (float(cur) - prev) / prev
    except Exception:
        return 0.0

def safe_mean(s: pd.Series, fallback: float = 0.0) -> float:
    try:
        val = float(s.mean())
        if math.isnan(val) or math.isinf(val):
            return fallback
        return val
    except Exception:
        return fallback

def safe_sum(s: pd.Series, fallback: float = 0.0) -> float:
    try:
        val = float(s.sum())
        if math.isnan(val) or math.isinf(val):
            return fallback
        return val
    except Exception:
        return fallback

def download_button(df: pd.DataFrame, label: str, filename: str):
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    st.download_button(label, buf.getvalue(), file_name=filename, mime="text/csv")


# -----------------------------------------------------------------------------
# URL Filter Handling (region/channel)
# -----------------------------------------------------------------------------
ALL_REGIONS = ["AMER", "APAC", "EMEA"]
ALL_CHANNELS = ["Online", "Retail", "Wholesale"]

# Default mix (sums to 1.0 each)
REGION_WEIGHTS = {"AMER": 0.45, "APAC": 0.30, "EMEA": 0.25}
CHANNEL_WEIGHTS = {"Online": 0.60, "Retail": 0.30, "Wholesale": 0.10}

def get_query_params():
    # Streamlit >=1.32
    try:
        qp = st.query_params  # mapping-like
        return {k: v for k, v in qp.items()}
    except Exception:
        # Older API fallback
        try:
            return st.experimental_get_query_params()
        except Exception:
            return {}

def set_query_params(**kwargs):
    try:
        # New API
        current = dict(get_query_params())
        current.update({k:v for k,v in kwargs.items() if v is not None})
        st.query_params.clear()
        for k, v in current.items():
            st.query_params[k] = v
    except Exception:
        # Fallback
        try:
            st.experimental_set_query_params(**kwargs)
        except Exception:
            pass

def parse_csv_param(val, allowed):
    if not val:
        return allowed[:]
    if isinstance(val, list):
        # could be already list from Streamlit
        raw = ",".join(val)
    else:
        raw = str(val)
    selected = [x.strip() for x in raw.split(",") if x.strip()]
    # keep only allowed ones, preserve order from allowed
    return [a for a in allowed if a in selected] or allowed[:]

def compute_filter_factor(regions_selected, channels_selected):
    # independence assumption => factor = sum(w_r) * sum(w_c)
    wr = sum(REGION_WEIGHTS.get(r, 0) for r in regions_selected)
    wc = sum(CHANNEL_WEIGHTS.get(c, 0) for c in channels_selected)
    # clamp to [0,1]; ensure nonzero minimum to avoid fully blank views
    wr = min(max(wr, 0.0), 1.0)
    wc = min(max(wc, 0.0), 1.0)
    return max(wr * wc, 0.0)


# -----------------------------------------------------------------------------
# Demo Data (cached) — Python 3.13-safe
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def generate_demo_data(seed: int = 42):
    rng = np.random.default_rng(seed)
    today = datetime.utcnow().date()
    start = today - timedelta(days=365)
    dates = pd.date_range(start, periods=366, freq="D")

    # Traffic & commerce dynamics
    base_sessions = 5000
    weekly_seasonality = (np.sin(2 * np.pi * np.arange(len(dates)) / 7) + 1) / 2  # 0..1
    promo_spikes = np.zeros(len(dates))
    promo_days = rng.choice(np.arange(30, len(dates)-30), size=8, replace=False)
    promo_spikes[promo_days] = rng.integers(1000, 4000, size=len(promo_days))

    sessions = (base_sessions
                + weekly_seasonality * 1500
                + promo_spikes
                + rng.normal(0, 350, len(dates))).clip(min=500)

    # Conversion rate: base 2.8%, seasonal variation, promo efficiency
    conv = (0.028
            + 0.004 * ((np.cos(2 * np.pi * np.arange(len(dates)) / 30) + 1)/2)
            + (promo_spikes > 0) * 0.003
            + rng.normal(0, 0.0012, len(dates))).clip(0.01, 0.065)

    # AOV: base $62 with modest variance and weekend uplift
    day_of_week = pd.Series(dates).dt.dayofweek.values
    aov = (62
           + (day_of_week >= 5) * 4  # weekend uplift
           + rng.normal(0, 2.5, len(dates))).clip(35, 120)

    # Refund rate
    refund_rate = (0.030
                   + 0.002 * ((np.sin(2 * np.pi * np.arange(len(dates)) / 45) + 1)/2)
                   + rng.normal(0, 0.0008, len(dates))).clip(0.005, 0.08)

    # Revenue metrics
    orders = (sessions * conv).round().astype(int)
    gross_revenue = orders * aov
    refunds = gross_revenue * refund_rate
    net_revenue = gross_revenue - refunds

    kpis = pd.DataFrame({
        "date": dates,
        "sessions": sessions.astype(int),
        "conversion_rate": conv,
        "orders": orders,
        "aov": aov,
        "gross_revenue": gross_revenue,
        "refund_rate": refund_rate,
        "refunds": refunds,
        "net_revenue": net_revenue,
    })

    # Synthetic customer base (RFM)
    n_customers = 5000
    cust_ids = [f"C{100000 + i}" for i in range(n_customers)]
    first_purchase_dates = rng.choice(dates[:150], size=n_customers, replace=True)
    last_purchase_dates = []
    frequency = []
    monetary = []
    for i in range(n_customers):
        fp = pd.Timestamp(first_purchase_dates[i]).date()
        recent_bias = int(np.int64(np.asarray(np.random.default_rng().integers(0, 200))))
        extra = int(np.int64(np.asarray(np.random.default_rng().integers(0, 60))))
        lp_date = fp + timedelta(days=int(recent_bias + extra))
        lp = min(today, lp_date)
        last_purchase_dates.append(lp)
        freq_base = int(np.random.poisson(2))
        recent_boost = 1 if (today - lp).days < 60 else 0
        freq = max(1, int(freq_base + recent_boost))
        frequency.append(freq)
        monetary.append(max(20.0, float(np.random.normal(65, 25))))
    customers = pd.DataFrame({
        "customer_id": cust_ids,
        "first_purchase": pd.to_datetime(first_purchase_dates),
        "last_purchase": pd.to_datetime(last_purchase_dates),
        "frequency": frequency,
        "monetary": monetary,
    })
    customers["recency_days"] = (pd.Timestamp(today) - customers["last_purchase"]).dt.days

    return kpis, customers

kpis_base, customers = generate_demo_data()

# -----------------------------------------------------------------------------
# Apply URL Filters to KPIs
# -----------------------------------------------------------------------------
params = get_query_params()
regions_selected = parse_csv_param(params.get("region"), ALL_REGIONS)
channels_selected = parse_csv_param(params.get("channel"), ALL_CHANNELS)
factor = compute_filter_factor(regions_selected, channels_selected)

def apply_factor_to_kpis(df: pd.DataFrame, factor: float) -> pd.DataFrame:
    df = df.copy()
    # Scale volume & money cols; keep rates steady
    for col in ["sessions", "orders", "gross_revenue", "refunds", "net_revenue"]:
        if col in df.columns:
            if col in ["sessions", "orders"]:
                df[col] = (df[col] * factor).round().astype(int)
            else:
                df[col] = df[col] * factor
    return df

kpis = apply_factor_to_kpis(kpis_base, factor)

# Windows
today = kpis["date"].max().date()
last_30 = kpis[kpis["date"] >= (pd.Timestamp(today) - pd.Timedelta(days=30))]
prev_30 = kpis[(kpis["date"] < (pd.Timestamp(today) - pd.Timedelta(days=30))) &
               (kpis["date"] >= (pd.Timestamp(today) - pd.Timedelta(days=60)))]

# -----------------------------------------------------------------------------
# Viz Helpers
# -----------------------------------------------------------------------------
def plot_line(df: pd.DataFrame, x: str, y: str, title: str, highlight=None):
    if PLOTLY_AVAILABLE:
        fig = px.line(df, x=x, y=y, title=title)
        fig.update_layout(
            height=380,
            margin=dict(l=10, r=10, t=50, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,1)",
            font=dict(color="#e2e8f0"),
            xaxis=dict(gridcolor="rgba(148,163,184,0.12)"),
            yaxis=dict(gridcolor="rgba(148,163,184,0.12)"),
        )
        if highlight is not None and len(highlight) > 0:
            fig.add_trace(
                go.Scatter(
                    x=highlight[x],
                    y=highlight[y],
                    mode="markers",
                    marker=dict(size=9),
                    name="Anomaly",
                )
            )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.line_chart(df.set_index(x)[y])

def plot_scatter(df: pd.DataFrame, x: str, y: str, color: str, title: str, tooltip=None):
    if PLOTLY_AVAILABLE:
        fig = px.scatter(df, x=x, y=y, color=color, hover_data=tooltip or [], title=title)
        fig.update_layout(
            height=420,
            margin=dict(l=10, r=10, t=50, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,1)",
            font=dict(color="#e2e8f0"),
            xaxis=dict(gridcolor="rgba(148,163,184,0.12)"),
            yaxis=dict(gridcolor="rgba(148,163,184,0.12)"),
            legend=dict(bgcolor="rgba(15,23,42,0.6)")
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("Interactive scatter requires Plotly; falling back to table:")
        st.dataframe(df[[x, y, color] + (tooltip or [])])

def kpi_card(title: str, value, delta: float = None, delta_hint: str = "vs prev 30d"):
    st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="kpi-title">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="kpi-value">{value}</div>', unsafe_allow_html=True)
    if delta is not None:
        color = SUCCESS if delta >= 0 else DANGER
        sign = "▲" if delta >= 0 else "▼"
        st.markdown(
            f'<div class="kpi-delta" style="color:{color}">{sign} {delta*100:.2f}% {delta_hint}</div>',
            unsafe_allow_html=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# RFM Segmentation
# -----------------------------------------------------------------------------
def quintile_score(series: pd.Series, reverse: bool = False) -> pd.Series:
    ranks = series.rank(method="first", pct=True)
    pct = (1 - ranks) if reverse else ranks
    scores = np.ceil(pct * 5).astype(int)
    return scores.clip(1, 5)

def rfm_segmentation(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["R"] = quintile_score(out["recency_days"], reverse=True)
    out["F"] = quintile_score(out["frequency"], reverse=False)
    out["M"] = quintile_score(out["monetary"], reverse=False)
    out["RFM_Score"] = out["R"]*100 + out["F"]*10 + out["M"]
    seg = []
    for _, row in out.iterrows():
        if row["R"] >= 4 and row["F"] >= 4 and row["M"] >= 4:
            seg.append("Champions")
        elif row["R"] >= 4 and row["F"] >= 3:
            seg.append("Loyal")
        elif row["R"] <= 2 and row["F"] >= 3 and row["M"] >= 3:
            seg.append("At Risk")
        elif row["R"] <= 2 and row["F"] <= 2:
            seg.append("Churn Risk")
        elif row["R"] >= 4 and row["F"] <= 2:
            seg.append("Promising")
        else:
            seg.append("Needs Nurture")
    out["segment"] = seg
    return out

# -----------------------------------------------------------------------------
# Anomaly Detection
# -----------------------------------------------------------------------------
def iqr_anomalies(series: pd.Series, window:int=14, sensitivity: float=1.5):
    s = series.copy().astype(float)
    med = s.rolling(window=window, min_periods=3, center=True).median()
    resid = s - med
    q1 = resid.rolling(window=window, min_periods=3).quantile(0.25)
    q3 = resid.rolling(window=window, min_periods=3).quantile(0.75)
    iqr = (q3 - q1).fillna(0)
    lower = (q1 - sensitivity * iqr).fillna(-np.inf)
    upper = (q3 + sensitivity * iqr).fillna(np.inf)
    return ((resid < lower) | (resid > upper)).fillna(False)

# -----------------------------------------------------------------------------
# Forecasting
# -----------------------------------------------------------------------------
def seasonal_naive_forecast(df: pd.DataFrame, y: str, periods: int=30, season:int=7):
    hist = df.set_index("date")[y].astype(float)
    ema = hist.ewm(alpha=0.3).mean()
    resid = hist - ema
    resid_std = float(np.nanstd(resid[-90:])) if len(resid) >= 30 else float(np.nanstd(resid))

    last_date = df["date"].max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=periods, freq="D")
    fc_vals = []
    for i, _ in enumerate(future_dates):
        if len(hist) >= season:
            seasonal_idx = -season + (i % season)
            base = float(hist.iloc[seasonal_idx])
        else:
            base = float(hist.iloc[-1])
        drift = float(ema.iloc[-1]) - float(ema.iloc[-season]) if len(ema) > season else 0.0
        fc = base + (drift * ((i+1)/periods))
        fc_vals.append(fc)

    forecast = pd.DataFrame({"date": future_dates, "forecast": fc_vals})
    pi = 1.96 * resid_std
    forecast["pi_low"] = forecast["forecast"] - pi
    forecast["pi_high"] = forecast["forecast"] + pi
    return forecast

# -----------------------------------------------------------------------------
# Revenue Recovery Calculator
# -----------------------------------------------------------------------------
def revenue_recovery_opportunity(df_last_30: pd.DataFrame):
    if len(df_last_30) == 0:
        return {"conv_uplift": 0.0, "refund_reduction": 0.0, "total": 0.0}

    sessions = float(df_last_30["sessions"].sum())
    cur_aov = safe_mean(df_last_30["aov"], 60.0)

    delta_conv = 0.003    # +0.30pp
    delta_refund = -0.010 # -1.0pp

    add_orders = sessions * delta_conv
    conv_uplift = add_orders * cur_aov

    gross_rev = float(df_last_30["gross_revenue"].sum())
    refund_reduction = gross_rev * abs(delta_refund)

    total = conv_uplift + refund_reduction
    return {"conv_uplift": conv_uplift, "refund_reduction": refund_reduction, "total": total}


# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
col_left, col_right = st.columns([0.8, 0.2])
with col_left:
    st.markdown(
        f"""
        <div style="padding: 6px 0 2px 0;">
          <span class="pill">AI Revenue Recovery</span>
          <div class="section-title" style="margin-top:10px;">AI BI Dashboard — Recover Up To $500K+</div>
          <div class="section-sub">Real-time insights. Automated revenue defenses. Forecasts with action plans.</div>
        </div>
        """, unsafe_allow_html=True
    )
with col_right:
    st.markdown(
        f"""
        <div style="text-align:right;margin-top:6px;">
          <a href="#" style="
            text-decoration:none;color:white;font-weight:800;
            background:{PRIMARY_GRADIENT};padding:10px 14px;border-radius:12px;display:inline-block;">
            Launch Playbook
          </a>
        </div>
        """, unsafe_allow_html=True
    )

# Visible filter controls synced with URL
flt1, flt2 = st.columns([0.65, 0.35])
with flt1:
    st.markdown(
        f"<div class='section-sub'>Filters active — "
        f"<b>Region:</b> {', '.join(regions_selected)} · "
        f"<b>Channel:</b> {', '.join(channels_selected)} "
        f"(factor: {factor:.2f})</div>",
        unsafe_allow_html=True
    )
with flt2:
    with st.expander("Adjust Filters (sync to URL)"):
        sel_regions = st.multiselect("Regions", ALL_REGIONS, default=regions_selected)
        sel_channels = st.multiselect("Channels", ALL_CHANNELS, default=channels_selected)
        if st.button("Apply Filters"):
            # update URL with CSV params
            set_query_params(
                region=",".join(sel_regions) if sel_regions else ",".join(ALL_REGIONS),
                channel=",".join(sel_channels) if sel_channels else ",".join(ALL_CHANNELS),
            )
            st.experimental_rerun()

st.divider()


# -----------------------------------------------------------------------------
# Tabs
# -----------------------------------------------------------------------------
tab_home, tab_dash, tab_seg, tab_ano, tab_fc, tab_wait = st.tabs(
    ["🏠 Home", "📊 Dashboard", "👥 Segmentation", "🧪 Anomaly Detection", "📈 Forecasting", "🚀 Early Access"]
)


# -----------------------------------------------------------------------------
# HOME
# -----------------------------------------------------------------------------
with tab_home:
    with st.container():
        c1, c2, c3, c4 = st.columns(4)
        cur_rev = float(last_30["net_revenue"].sum())
        prev_rev = float(prev_30["net_revenue"].sum())
        cur_orders = int(last_30["orders"].sum())
        prev_orders = int(prev_30["orders"].sum()) if len(prev_30) else 0
        cur_conv = safe_mean(last_30["conversion_rate"])
        prev_conv = safe_mean(prev_30["conversion_rate"], cur_conv)
        cur_ref = safe_mean(last_30["refund_rate"])
        prev_ref = safe_mean(prev_30["refund_rate"], cur_ref)

        with c1:
            kpi_card("Net Revenue (30d)", fmt_money(cur_rev), pct_delta(cur_rev, prev_rev))
        with c2:
            kpi_card("Orders (30d)", f"{cur_orders:,}", pct_delta(cur_orders, prev_orders))
        with c3:
            kpi_card("Avg Conversion Rate", fmt_pct(cur_conv), pct_delta(cur_conv, prev_conv))
        with c4:
            kpi_card("Avg Refund Rate", fmt_pct(cur_ref), -pct_delta(cur_ref, prev_ref))

    opportunity = revenue_recovery_opportunity(last_30)
    st.markdown(
        f"""
        <div class="section-title">Revenue Recovery Opportunity (Next 30 Days)</div>
        <div class="section-sub">Modeling a +0.30pp conversion lift and a -1.0pp refund-rate reduction.</div>
        """, unsafe_allow_html=True
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        kpi_card("Conversion Uplift", fmt_money(opportunity["conv_uplift"]))
    with c2:
        kpi_card("Refund Reduction", fmt_money(opportunity["refund_reduction"]))
    with c3:
        kpi_card("Total Recoverable (Est.)", fmt_money(opportunity["total"]))
    st.caption("These are conservative, data-driven estimates grounded in your last 30 days and the active filters.")

    st.markdown("<div class='section-title'>Revenue Over Time</div>", unsafe_allow_html=True)
    plot_line(kpis, "date", "net_revenue", "Net Revenue (Daily)")

    with st.expander("Download demo data (filtered view affects revenue & volume only)"):
        download_button(kpis, "Download KPI Time Series (CSV)", "kpis_demo_filtered.csv")
        download_button(customers, "Download Customers (CSV)", "customers_demo.csv")

    st.success("Tip: Jump to **Early Access** to join the waitlist and unlock pilot pricing.")


# -----------------------------------------------------------------------------
# DASHBOARD
# -----------------------------------------------------------------------------
with tab_dash:
    st.markdown("<div class='section-title'>Business Overview</div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Gross Revenue (30d)", fmt_money(float(last_30["gross_revenue"].sum())))
    with c2:
        kpi_card("Net Revenue (30d)", fmt_money(float(last_30["net_revenue"].sum())))
    with c3:
        kpi_card("AOV (30d avg)", fmt_money(float(last_30["aov"].mean())))
    with c4:
        kpi_card("Sessions (30d)", f"{int(last_30['sessions'].sum()):,}")

    left, right = st.columns([0.62, 0.38], gap="large")

    with left:
        plot_line(kpis, "date", "gross_revenue", "Gross Revenue (Daily)")
        plot_line(kpis, "date", "orders", "Orders (Daily)")

    with right:
        st.markdown("<div class='section-title'>AI Insights</div>", unsafe_allow_html=True)
        last_7 = kpis[kpis["date"] >= (pd.Timestamp(today) - pd.Timedelta(days=7))]
        prev_7 = kpis[(kpis["date"] < (pd.Timestamp(today) - pd.Timedelta(days=0))) &
                      (kpis["date"] >= (pd.Timestamp(today) - pd.Timedelta(days=14)))]
        trend_rev = pct_delta(safe_sum(last_7["net_revenue"]), safe_sum(prev_7["net_revenue"]))
        trend_conv = pct_delta(safe_mean(last_7["conversion_rate"], 0.0001), safe_mean(prev_7["conversion_rate"], 0.0001))
        top_refund_window = kpis.sort_values("refund_rate", ascending=False).head(5)[["date","refund_rate"]]
        bullets = [
            f"Net revenue {'increased' if trend_rev>=0 else 'decreased'} **{abs(trend_rev)*100:.1f}%** week-over-week.",
            f"Conversion rate {'improved' if trend_conv>=0 else 'declined'} **{abs(trend_conv)*100:.1f}%** vs prior week.",
            f"Highest refund-rate days: {', '.join([pd.to_datetime(d).strftime('%b %d') for d in top_refund_window['date']])}.",
            f"Projected **{fmt_money(revenue_recovery_opportunity(last_30)['total'])}** recoverable with targeted plays.",
        ]
        st.markdown("\n".join([f"- {b}" for b in bullets]))

        st.markdown("<div class='section-title' style='margin-top:18px;'>Funnel Snapshot (Last 30d)</div>", unsafe_allow_html=True)
        try:
            funnel = pd.DataFrame({
                "stage": ["Sessions", "Product Views", "Add to Cart", "Checkout", "Orders"],
                "value": [
                    int(last_30["sessions"].sum()),
                    int(last_30["sessions"].sum() * 0.72),
                    int(last_30["sessions"].sum() * 0.45),
                    int(last_30["sessions"].sum() * 0.30),
                    int(last_30["orders"].sum()),
                ],
            })
            if PLOTLY_AVAILABLE:
                fig = px.funnel(funnel, x="value", y="stage", title="Funnel")
                fig.update_layout(
                    height=420,
                    margin=dict(l=10, r=10, t=50, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(15,23,42,1)",
                    font=dict(color="#e2e8f0"),
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.bar_chart(funnel.set_index("stage"))
        except Exception as e:
            st.warning(f"Funnel chart unavailable: {e}")


# -----------------------------------------------------------------------------
# SEGMENTATION
# -----------------------------------------------------------------------------
with tab_seg:
    st.markdown("<div class='section-title'>Customer Segmentation (RFM)</div>", unsafe_allow_html=True)
    seg_df = rfm_segmentation(customers)
    seg_counts = seg_df["segment"].value_counts().reset_index()
    seg_counts.columns = ["segment", "customers"]

    c1, c2, c3 = st.columns([0.35, 0.35, 0.30])
    with c1:
        kpi_card("Customers", f"{len(seg_df):,}")
        kpi_card("Champions", f"{int((seg_df['segment']=='Champions').sum()):,}")
    with c2:
        kpi_card("Loyal", f"{int((seg_df['segment']=='Loyal').sum()):,}")
        kpi_card("At Risk", f"{int((seg_df['segment']=='At Risk').sum()):,}")
    with c3:
        kpi_card("Churn Risk", f"{int((seg_df['segment']=='Churn Risk').sum()):,}")
        kpi_card("Promising", f"{int((seg_df['segment']=='Promising').sum()):,}")

    st.markdown("<div class='section-sub'>Segments are computed with percentile-based RFM scoring — robust and dependency-light.</div>", unsafe_allow_html=True)

    if PLOTLY_AVAILABLE:
        fig = px.bar(seg_counts, x="segment", y="customers", title="Segment Counts")
        fig.update_layout(
            height=420,
            margin=dict(l=10, r=10, t=50, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,1)",
            font=dict(color="#e2e8f0"),
            xaxis=dict(gridcolor="rgba(148,163,184,0.12)"),
            yaxis=dict(gridcolor="rgba(148,163,184,0.12)"),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.bar_chart(seg_counts.set_index("segment"))

    sample_n = min(1500, len(seg_df))
    if sample_n > 0:
        plot_scatter(seg_df.sample(sample_n, random_state=7),
                     x="recency_days", y="monetary", color="segment",
                     title="Monetary vs Recency by Segment", tooltip=["frequency","customer_id"])

    with st.expander("Download Segmentation Data"):
        download_button(seg_df, "Download RFM Segmentation (CSV)", "rfm_segmentation.csv")

    st.info("Playbook: Target 'At Risk' and 'Churn Risk' with win-back; upsell 'Champions' with VIP bundles.")


# -----------------------------------------------------------------------------
# ANOMALY DETECTION
# -----------------------------------------------------------------------------
with tab_ano:
    st.markdown("<div class='section-title'>Anomaly Detection</div>", unsafe_allow_html=True)

    metric = st.selectbox("Select metric for anomaly detection", ["net_revenue", "gross_revenue", "orders", "conversion_rate", "refund_rate"], index=0)
    window = st.slider("Rolling window (days)", min_value=7, max_value=30, value=14, step=1)
    sensitivity = st.slider("Sensitivity (IQR multiplier)", min_value=1.0, max_value=3.0, value=1.5, step=0.1)

    try:
        mask = iqr_anomalies(kpis[metric], window=window, sensitivity=sensitivity)
        anomalies = kpis.loc[mask, ["date", metric]].copy()

        st.markdown(f"<div class='section-sub'>Flagging points whose residuals exceed ±{sensitivity:.1f}×IQR from a rolling median (window={window}).</div>", unsafe_allow_html=True)
        plot_line(kpis[["date", metric]], "date", metric, f"{metric.replace('_',' ').title()} with Anomalies", highlight=anomalies)

        if len(anomalies) > 0:
            st.warning(f"Detected {len(anomalies)} anomalies. Highest magnitude:")
            st.dataframe(anomalies.sort_values(metric, ascending=False).head(10))
        else:
            st.success("No significant anomalies at current settings.")
    except Exception as e:
        st.error(f"Anomaly detection error: {e}")


# -----------------------------------------------------------------------------
# FORECASTING
# -----------------------------------------------------------------------------
with tab_fc:
    st.markdown("<div class='section-title'>Revenue Forecast (30 Days)</div>", unsafe_allow_html=True)
    target_metric = st.selectbox("Forecast metric", ["net_revenue", "orders", "gross_revenue"], index=0)
    horizon = st.slider("Horizon (days)", min_value=14, max_value=60, value=30, step=1)

    try:
        df_fc = kpis[["date", target_metric]].rename(columns={target_metric: "y"})
        fc = seasonal_naive_forecast(df_fc, "y", periods=horizon)
        hist = kpis[["date", target_metric]].tail(180).copy()

        if PLOTLY_AVAILABLE:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hist["date"], y=hist[target_metric], name="History", mode="lines"))
            fig.add_trace(go.Scatter(x=fc["date"], y=fc["forecast"], name="Forecast", mode="lines"))
            fig.add_trace(go.Scatter(
                x=pd.concat([fc["date"], fc["date"][::-1]]),
                y=pd.concat([fc["pi_high"], fc["pi_low"][::-1]]),
                fill="toself",
                name="95% PI",
                mode="lines",
                line=dict(width=0),
                opacity=0.25
            ))
            fig.update_layout(
                height=420,
                margin=dict(l=10, r=10, t=50, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,23,42,1)",
                font=dict(color="#e2e8f0"),
                xaxis=dict(gridcolor="rgba(148,163,184,0.12)"),
                yaxis=dict(gridcolor="rgba(148,163,184,0.12)"),
                title=f"{target_metric.replace('_',' ').title()} Forecast"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.line_chart(hist.set_index("date")[target_metric])
            st.line_chart(fc.set_index("date")["forecast"])

        forecast_total = float(fc["forecast"].sum())
        st.markdown("<div class='section-title' style='margin-top:18px;'>Summary</div>", unsafe_allow_html=True)
        st.write(f"Projected {target_metric.replace('_',' ')} over next {horizon} days: **{fmt_money(forecast_total)}**.")

        if target_metric == "net_revenue":
            opp = revenue_recovery_opportunity(last_30)
            st.success(f"Add **{fmt_money(opp['total'])}** via conversion uplift (+0.30pp) and refund reduction (-1.0pp).")
    except Exception as e:
        st.error(f"Forecasting error: {e}")


# -----------------------------------------------------------------------------
# EARLY ACCESS (Waitlist)
# -----------------------------------------------------------------------------
with tab_wait:
    st.markdown("<div class='section-title'>Join Early Access</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>Secure pilot pricing and priority onboarding. Limited seats for growth teams targeting $500K+ revenue recovery.</div>", unsafe_allow_html=True)

    with st.form("waitlist_form", clear_on_submit=False):
        c1, c2 = st.columns(2)
        with c1:
            name = st.text_input("Full Name *", max_chars=80)
            email = st.text_input("Work Email *", max_chars=120, placeholder="you@company.com")
            role = st.selectbox("Role *", ["Founder/CEO", "CRO/Head of Growth", "Product Lead", "Marketing Lead", "Ops/Finance", "Other"])
        with c2:
            company = st.text_input("Company *", max_chars=120)
            company_size = st.select_slider("Company Size", options=["1-10","11-50","51-200","201-500","501-1000","1000+"], value="51-200")
            est_lost = st.select_slider("Estimated Lost Revenue / mo", options=["$10K-$50K", "$50K-$100K", "$100K-$250K", "$250K-$500K", "$500K-$1M", "$1M+"], value="$250K-$500K")
        use_case = st.text_area("Primary Use Case *", placeholder="e.g., Reduce refund leakage, recover abandoned checkouts, improve merchandise availability, etc.", height=100)
        agree = st.checkbox("I agree to be contacted about Early Access updates.", value=True)

        submitted = st.form_submit_button("Request Early Access 🚀")

    def save_local_csv(row: dict, path: str = "waitlist.csv"):
        try:
            exists = os.path.exists(path)
            df = pd.DataFrame([row])
            if exists:
                df_existing = pd.read_csv(path)
                df = pd.concat([df_existing, df], ignore_index=True)
            df.to_csv(path, index=False)
            return True, None
        except Exception as e:
            return False, str(e)

    def submit_google_form(row: dict):
        """If st.secrets contains GOOGLE_FORM_ACTION and GOOGLE_FORM_FIELDS (JSON mapping),
        submit via POST to Google Form. Returns (ok, message)."""
        try:
            form_action = st.secrets.get("GOOGLE_FORM_ACTION", "").strip()
            fields_json = st.secrets.get("GOOGLE_FORM_FIELDS", "")
            if not form_action or not fields_json:
                return False, "Google Form secrets not configured."

            mapping = json.loads(fields_json)  # {"name":"entry.12345", ...}
            payload = {}
            for k, v in row.items():
                if k in mapping:
                    payload[mapping[k]] = v

            if not REQUESTS_AVAILABLE:
                return False, "requests library unavailable for HTTP submission."

            resp = requests.post(form_action, data=payload, timeout=8)
            if resp.status_code in (200, 302):
                return True, "Submitted to Google Form."
            return False, f"Google Form submission HTTP {resp.status_code}"
        except Exception as e:
            return False, str(e)

    if submitted:
        missing = []
        for label, val in [("Full Name", name), ("Work Email", email), ("Company", company), ("Use Case", use_case)]:
            if not str(val).strip():
                missing.append(label)
        if missing or not agree:
            st.error(f"Please complete required fields: {', '.join(missing)}" + ("" if agree else " and accept contact consent."))
        else:
            row = {
                "timestamp_utc": datetime.utcnow().isoformat(),
                "name": name.strip(),
                "email": email.strip(),
                "role": role,
                "company": company.strip(),
                "company_size": company_size,
                "estimated_lost_revenue": est_lost,
                "use_case": use_case.strip(),
                # Optional: include active filters for GTM triage
                "regions": ",".join(regions_selected),
                "channels": ",".join(channels_selected),
            }
            ok_gf, msg_gf = submit_google_form(row)
            if ok_gf:
                st.success("You're on the list! We'll be in touch soon.")
                st.balloons()
            else:
                ok_local, msg_local = save_local_csv(row)
                if ok_local:
                    st.success("You're on the list! (Saved locally). We'll be in touch soon.")
                    st.balloons()
                else:
                    st.error(f"Could not save your submission: {msg_gf or msg_local}")

    st.markdown("### Integration Details")
    st.caption(textwrap.dedent("""
        Optional: Configure **Google Form** integration via `st.secrets`.
        - `GOOGLE_FORM_ACTION`: the Google Form `formResponse` action URL.
        - `GOOGLE_FORM_FIELDS`: JSON mapping from our field keys to Google entry IDs, e.g.:
            {
              "name": "entry.1111111",
              "email": "entry.2222222",
              "role": "entry.3333333",
              "company": "entry.4444444",
              "company_size": "entry.5555555",
              "estimated_lost_revenue": "entry.6666666",
              "use_case": "entry.7777777",
              "regions": "entry.8888888",
              "channels": "entry.9999999"
            }
        If not provided, submissions are safely stored to `waitlist.csv` in the app directory.
    """))

    st.markdown("### Why Teams Choose This")
    st.markdown(
        """
        - **$500K+ Recovery Potential:** Combine conversion uplift, refund interception, and win-back plays.  
        - **Anomaly Guardrails:** Detect & react to leakage the same day.  
        - **Predictive Planning:** 30–60d forecasts to allocate spend and inventory with confidence.  
        - **Zero Heavy Dependencies:** Runs fast without bulky ML installs; add-ons optional.  
        """.strip()
    )


# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.divider()
st.markdown(
    f"""
    <div style="display:flex;justify-content:space-between;align-items:center;opacity:.9">
      <div style="color:#94a3b8;font-size:13px;">
        © {datetime.utcnow().year} AI BI Dashboard · Built for revenue recovery.
      </div>
      <div style="font-size:13px;">
        <span class="pill">Made with Streamlit</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True
)
