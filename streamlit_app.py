# AI BI Dashboard – Streamlit (Polished + Waitlist)
# -------------------------------------------------
# Copy-paste this into `streamlit_app.py` and deploy on Streamlit Cloud.
# 
# Optional: add the following to `.streamlit/secrets.toml` to enable Google Form direct/iframe modes
#
# [WAITLIST]
# # Option A: simple iframe embed (recommended if you already have a Form URL)
# google_form_iframe_url = "https://docs.google.com/forms/d/e/FORM_ID/viewform?embedded=true"
#
# # Option B: Direct POST to Google Form (requires the form_id and entry ids)
# use_direct_post = true
# form_id = "FORM_ID"                    # e.g., 1FAIpQLSc... (keep the whole ID from the /d/e/<id>/ path)
# email_entry_id = "entry.123456789"     # replace with your Google Form field id for Email
# name_entry_id = "entry.987654321"      # optional name field id
# extra_entry_id = "entry.135792468"     # optional free-text field id
#
# TIP: To find entry ids: open the live form > Inspect the Email input element > look for name="entry.xxxxxxx".
# -------------------------------------------------

import io
import os
import sys
import time
import json
import math
import textwrap
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

import streamlit as st
import altair as alt

# ML / Stats (lightweight)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Networking (for Google Forms direct submit)
try:
    import requests
except Exception:
    requests = None  # We'll handle gracefully

st.set_page_config(
    page_title="AI BI Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------
# Small CSS polish
# ----------------------------
st.markdown(
    """
    <style>
      .metric-card {background: #fafafa; border: 1px solid #eee; padding: 14px 16px; border-radius: 14px;}
      .small-muted {color: #666; font-size: 12px;}
      .stTabs [data-baseweb="tab-list"] {gap: 6px;}
      .stTabs [data-baseweb="tab"] {background: #f7f7f9; padding: 10px 12px; border-radius: 10px; border: 1px solid #eee;}
      .stTabs [aria-selected="true"] {background: white; border-color: #ddd;}
      .ok {color: #0fa;}
      .warn {color: #e6a700;}
      .err {color: #e63946;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# Config via Secrets
# ----------------------------
@dataclass
class WaitlistConfig:
    google_form_iframe_url: Optional[str] = None
    use_direct_post: bool = False
    form_id: Optional[str] = None
    email_entry_id: Optional[str] = None
    name_entry_id: Optional[str] = None
    extra_entry_id: Optional[str] = None


def get_waitlist_config() -> WaitlistConfig:
    cfg = WaitlistConfig()
    try:
        w = st.secrets.get("WAITLIST", {})
        cfg.google_form_iframe_url = w.get("google_form_iframe_url")
        cfg.use_direct_post = bool(w.get("use_direct_post", False))
        cfg.form_id = w.get("form_id")
        cfg.email_entry_id = w.get("email_entry_id")
        cfg.name_entry_id = w.get("name_entry_id")
        cfg.extra_entry_id = w.get("extra_entry_id")
    except Exception:
        pass
    return cfg


WAITLIST_CFG = get_waitlist_config()

# ----------------------------
# Utility / Data Helpers
# ----------------------------

def _find_datetime_column(df: pd.DataFrame) -> Optional[str]:
    for col in df.columns:
        if np.issubdtype(df[col].dtype, np.datetime64):
            return col
    # try parse common names
    for guess in ["date", "timestamp", "time", "datetime"]:
        if guess in df.columns:
            try:
                df[guess] = pd.to_datetime(df[guess])
                return guess
            except Exception:
                continue
    return None


@st.cache_data(show_spinner=False)
def load_data(file) -> pd.DataFrame:
    if file is None:
        return pd.DataFrame()
    name = getattr(file, "name", "uploaded")
    try:
        if name.lower().endswith(".csv"):
            df = pd.read_csv(file)
        elif name.lower().endswith(".parquet"):
            df = pd.read_parquet(file)
        elif name.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(file)
        else:
            df = pd.read_csv(file)  # best effort
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        return pd.DataFrame()

    # best-effort parse datetimes
    for col in df.columns:
        if df[col].dtype == object:
            # try only obvious datetime columns
            if any(k in col.lower() for k in ["date", "time"]):
                try:
                    df[col] = pd.to_datetime(df[col])
                except Exception:
                    pass
    return df


@st.cache_data(show_spinner=False)
def make_demo_data(n_days: int = 365, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = pd.Timestamp.today().normalize() - pd.Timedelta(days=n_days)
    dates = pd.date_range(start, periods=n_days, freq="D")
    categories = ["Online", "Retail", "Wholesale"]
    regions = ["EMEA", "APAC", "AMER"]

    data = []
    base = 1000
    season = np.sin(np.linspace(0, 6 * np.pi, n_days)) * 150
    trend = np.linspace(0, 250, n_days)
    for i, d in enumerate(dates):
        for c in categories:
            for r in regions:
                noise = rng.normal(0, 80)
                amount = max(0, base + season[i] + trend[i] + noise + rng.normal(0, 30))
                units = max(1, int(10 + season[i]/20 + rng.normal(0, 3)))
                price = amount / units
                data.append({
                    "date": d,
                    "channel": c,
                    "region": r,
                    "revenue": round(amount, 2),
                    "units": units,
                    "price": round(price, 2),
                })
    df = pd.DataFrame(data)
    return df


def safe_number(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def kpi_card(label: str, value: str, helptext: str = ""):
    with st.container(border=True):
        st.markdown(f"<div class='metric-card'><h4>{label}</h4><h2>{value}</h2>\n<p class='small-muted'>{helptext}</p></div>", unsafe_allow_html=True)


def format_pct(x: float) -> str:
    if pd.isna(x):
        return "–"
    sign = "+" if x >= 0 else ""
    return f"{sign}{x:.1f}%"


# ----------------------------
# Sidebar – data & global filters
# ----------------------------
st.sidebar.title("AI BI Dashboard")

with st.sidebar:
    st.caption("Upload data or use demo. We auto-detect date & basic fields.")
    data_src = st.radio("Data source", ["Use demo data", "Upload CSV/Parquet/Excel"], index=0, horizontal=False)

    if data_src == "Upload CSV/Parquet/Excel":
        upl = st.file_uploader("Upload a file", type=["csv", "parquet", "xlsx", "xls"])
        df = load_data(upl)
        if df.empty:
            st.info("No data yet – using demo until a valid file is uploaded.")
            df = make_demo_data()
    else:
        df = make_demo_data()

    # Auto-detect date column
    date_col = _find_datetime_column(df)
    if not date_col:
        # last-resort: try parse first column as date
        try:
            df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
            date_col = df.columns[0]
        except Exception:
            date_col = None

    if date_col:
        min_d, max_d = pd.to_datetime(df[date_col]).min(), pd.to_datetime(df[date_col]).max()
        date_range = st.date_input("Date range", value=(min_d.date(), max_d.date()), min_value=min_d.date(), max_value=max_d.date())
    else:
        date_range = None

    # Category/segment filters (best-effort)
    cat_cols = [c for c in df.columns if df[c].dtype == object]
    cat_filters = {}
    for c in cat_cols[:3]:  # avoid bloating the sidebar
        vals = sorted(df[c].dropna().unique().tolist())[:50]
        sel = st.multiselect(f"Filter {c}", vals, default=vals)
        cat_filters[c] = sel

    st.divider()
    st.subheader("Navigation")
    nav = st.radio(
        "Go to",
        ["📊 Dashboard", "🧩 Segmentation", "🚨 Anomalies", "📈 Forecast", "🧠 AI Insights", "🕒 Early Access Waitlist"],
        index=0,
    )

# Apply filters
if date_col and date_range is not None and len(date_range) == 2:
    start_d, end_d = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    df = df[(pd.to_datetime(df[date_col]) >= start_d) & (pd.to_datetime(df[date_col]) <= end_d)]

for c, allowed in (cat_filters or {}).items():
    if allowed:
        df = df[df[c].isin(allowed)]

# Prepare numeric columns
num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

# ----------------------------
# Top Tabs (content)
# ----------------------------
TABS = st.tabs([
    "📊 Dashboard",
    "🧩 Segmentation",
    "🚨 Anomalies",
    "📈 Forecast",
    "🧠 AI Insights",
    "🕒 Early Access Waitlist",
])

# ----------------------------
# TAB 1: 📊 Dashboard
# ----------------------------
with TABS[0]:
    st.markdown("### Overview")
    if date_col is None:
        st.warning("No datetime column detected. Some charts may be limited.")

    # Aggregate by date
    if date_col and not df.empty:
        by_date = (
            df.assign(_date=pd.to_datetime(df[date_col]).dt.date)
              .groupby("_date")
              .agg({c: "sum" for c in num_cols})
              .reset_index()
              .rename(columns={"_date": "date"})
        )

        # KPIs
        rev_col = next((c for c in ["revenue", "sales", "amount", "value"] if c in df.columns), None)
        units_col = "units" if "units" in df.columns else None

        if rev_col is None and num_cols:
            rev_col = num_cols[0]

        col1, col2, col3, col4 = st.columns(4)
        if rev_col in by_date.columns:
            total_rev = float(by_date[rev_col].sum())
            last = by_date.iloc[-1][rev_col]
            prev = by_date.iloc[-2][rev_col] if len(by_date) > 1 else last
            mom = ((last - prev) / prev * 100) if prev else np.nan
            with col1: kpi_card("Total Revenue", f"£{total_rev:,.0f}", f"Yesterday vs prior: {format_pct(mom)}")
        else:
            with col1: kpi_card("Total (first numeric)", f"{by_date[num_cols[0]].sum():,.0f}")

        if units_col and units_col in by_date.columns:
            with col2:
                kpi_card("Units Sold", f"{int(by_date[units_col].sum()):,}")
        else:
            with col2:
                any_second = num_cols[1] if len(num_cols) > 1 else None
                if any_second:
                    kpi_card("Secondary Total", f"{by_date[any_second].sum():,.0f}")
                else:
                    kpi_card("Rows", f"{len(df):,}")

        if rev_col in by_date.columns and units_col and units_col in by_date.columns:
            avg_price = (by_date[rev_col].sum() / max(1, by_date[units_col].sum()))
            with col3:
                kpi_card("Avg Price", f"£{avg_price:,.2f}")
        else:
            with col3:
                kpi_card("Columns", f"{len(df.columns):,}")

        with col4:
            kpi_card("Active Days", f"{by_date['date'].nunique():,}")

        # Line chart
        if rev_col in by_date.columns:
            chart = alt.Chart(by_date).mark_line().encode(
                x="date:T",
                y=alt.Y(f"{rev_col}:Q", title=rev_col.capitalize()),
                tooltip=["date:T", alt.Tooltip(f"{rev_col}:Q", format=",.0f")],
            ).properties(height=320)
            st.altair_chart(chart, use_container_width=True)

    # Category breakdowns (top 2 cats)
    cat_cols_all = [c for c in df.columns if df[c].dtype == object]
    if cat_cols_all:
        rev_col = next((c for c in ["revenue", "sales", "amount", "value"] if c in df.columns), None)
        if rev_col is None and num_cols:
            rev_col = num_cols[0]

        top_cols = cat_cols_all[:2]
        cols = st.columns(len(top_cols))
        for i, c in enumerate(top_cols):
            try:
                agg = df.groupby(c)[rev_col].sum().reset_index().sort_values(rev_col, ascending=False).head(15)
                bar = alt.Chart(agg).mark_bar().encode(
                    x=alt.X(f"{rev_col}:Q", title=rev_col.capitalize()),
                    y=alt.Y(f"{c}:N", sort='-x'),
                    tooltip=[c, alt.Tooltip(f"{rev_col}:Q", format=",.0f")],
                ).properties(height=360)
                cols[i].altair_chart(bar, use_container_width=True)
            except Exception as e:
                cols[i].warning(f"Cannot plot breakdown for {c}: {e}")

    with st.expander("Peek at data"):
        st.dataframe(df.head(500), use_container_width=True)

# ----------------------------
# TAB 2: 🧩 Segmentation
# ----------------------------
with TABS[1]:
    st.markdown("### Customer/Product Segmentation")
    if not num_cols:
        st.warning("No numeric columns available for clustering.")
    else:
        with st.form("seg_form"):
            feats = st.multiselect("Features for clustering", num_cols, default=num_cols[: min(4, len(num_cols))])
            k = st.slider("Number of clusters (k)", 2, 10, 4)
            submitted = st.form_submit_button("Run KMeans")

        if submitted and feats:
            try:
                X = df[feats].dropna()
                scaler = StandardScaler()
                Xs = scaler.fit_transform(X)
                km = KMeans(n_clusters=k, n_init=20, random_state=42)
                labels = km.fit_predict(Xs)

                prof = pd.DataFrame(X, index=X.index)
                prof["cluster"] = labels
                profile = prof.groupby("cluster").agg("mean").round(2)

                st.success("Segmentation complete.")
                st.dataframe(profile, use_container_width=True)

                # 2D projection via PCA
                pca = PCA(n_components=2, random_state=42)
                pts = pca.fit_transform(Xs)
                plot_df = pd.DataFrame({"pc1": pts[:, 0], "pc2": pts[:, 1], "cluster": labels.astype(str)})
                sc = alt.Chart(plot_df).mark_circle(size=60, opacity=0.6).encode(
                    x="pc1:Q", y="pc2:Q", color="cluster:N",
                    tooltip=["cluster", alt.Tooltip("pc1:Q", format=".2f"), alt.Tooltip("pc2:Q", format=".2f")],
                ).properties(height=420)
                st.altair_chart(sc, use_container_width=True)

                # Download segmented data
                out = df.copy()
                out.loc[X.index, "cluster"] = labels
                st.download_button("Download segmented data (CSV)", out.to_csv(index=False).encode("utf-8"), file_name="segments.csv", mime="text/csv")
            except Exception as e:
                st.error(f"Segmentation failed: {e}")

# ----------------------------
# TAB 3: 🚨 Anomalies
# ----------------------------
with TABS[2]:
    st.markdown("### Anomaly Detection")
    if date_col is None:
        st.warning("Needs a datetime column to aggregate over time.")
    else:
        target_col = st.selectbox("Target metric", options=[c for c in ["revenue", "sales", "amount", "value"] if c in df.columns] or num_cols)
        gran = st.selectbox("Granularity", ["D", "W", "M"], index=0, help="Aggregate by Day/Week/Month")
        method = st.selectbox("Method", ["IsolationForest", "Z-Score"], index=0)

        agg = (
            df.assign(_date=pd.to_datetime(df[date_col]).dt.to_period(gran).dt.start_time)
              .groupby("_date")[target_col].sum()
              .reset_index().rename(columns={"_date": "date", target_col: "y"})
        )

        if len(agg) < 10:
            st.warning("Not enough data points after aggregation.")
        else:
            try:
                if method == "IsolationForest":
                    X = agg[["y"]].values
                    iso = IsolationForest(contamination=0.05, random_state=42)
                    preds = iso.fit_predict(X)
                    scores = iso.decision_function(X)
                    agg["anomaly"] = (preds == -1)
                    agg["score"] = scores
                else:
                    m, s = agg["y"].mean(), agg["y"].std()
                    z = (agg["y"] - m) / (s if s else 1)
                    agg["anomaly"] = z.abs() > 2.5
                    agg["score"] = -z.abs()

                st.success("Anomaly scoring complete.")

                # Plot
                base = alt.Chart(agg).encode(x="date:T")
                line = base.mark_line().encode(y="y:Q")
                pts = base.mark_circle(size=80, opacity=0.9).encode(
                    y="y:Q",
                    color=alt.condition("datum.anomaly", alt.value("crimson"), alt.value("steelblue")),
                    tooltip=["date:T", alt.Tooltip("y:Q", format=",.0f"), "anomaly:N", alt.Tooltip("score:Q", format=".3f")],
                )
                st.altair_chart(line + pts, use_container_width=True)

                st.dataframe(agg[agg["anomaly"]], use_container_width=True)
                st.download_button("Download anomalies (CSV)", agg[agg["anomaly"]].to_csv(index=False).encode("utf-8"), file_name="anomalies.csv")
            except Exception as e:
                st.error(f"Anomaly detection failed: {e}")

# ----------------------------
# TAB 4: 📈 Forecast
# ----------------------------
with TABS[3]:
    st.markdown("### Forecasting (Exponential Smoothing)")
    if date_col is None:
        st.warning("Needs a datetime column.")
    else:
        target_col = st.selectbox("Target metric", options=[c for c in ["revenue", "sales", "amount", "value"] if c in df.columns] or num_cols)
        gran = st.selectbox("Granularity", ["D", "W", "M"], index=0)
        horizon = st.slider("Forecast horizon (periods)", 7 if gran == "D" else 8, 60, 30)

        ts = (
            df.assign(_date=pd.to_datetime(df[date_col]).dt.to_period(gran).dt.start_time)
              .groupby("_date")[target_col].sum()
              .asfreq(pd.infer_freq(pd.Series(pd.to_datetime(df[date_col]).dt.to_period(gran).dt.start_time).sort_values()) or gran)
        )

        if ts.isna().any():
            ts = ts.fillna(method="ffill").fillna(method="bfill")

        if len(ts) < 12:
            st.warning("Need at least 12 periods for a sensible fit.")
        else:
            try:
                seasonal = {"D": 7, "W": 52, "M": 12}[gran]
                model = ExponentialSmoothing(ts, trend="add", seasonal="add", seasonal_periods=seasonal)
                fit = model.fit(optimized=True, use_brute=True)
                fc = fit.forecast(horizon)

                # naive PI via residual std
                resid = fit.resid
                sigma = resid.std() if hasattr(resid, "std") else float(np.std(resid))
                ci_hi = fc + 1.96 * sigma
                ci_lo = fc - 1.96 * sigma

                fdf = pd.DataFrame({"date": fc.index, "forecast": fc.values, "lo": ci_lo.values, "hi": ci_hi.values})
                hist = ts.reset_index(); hist.columns = ["date", "y"]

                line_hist = alt.Chart(hist).mark_line().encode(x="date:T", y="y:Q")
                line_fc = alt.Chart(fdf).mark_line(strokeDash=[4,3]).encode(x="date:T", y="forecast:Q", tooltip=["date:T", alt.Tooltip("forecast:Q", format=",.0f")])
                band = alt.Chart(fdf).mark_area(opacity=0.2).encode(x="date:T", y="lo:Q", y2="hi:Q")
                st.altair_chart(line_hist + band + line_fc, use_container_width=True)

                st.download_button("Download forecast (CSV)", fdf.to_csv(index=False).encode("utf-8"), file_name="forecast.csv")
                st.success("Forecast ready.")
            except Exception as e:
                st.error(f"Forecasting failed: {e}")

# ----------------------------
# TAB 5: 🧠 AI Insights (rules-based, no API key needed)
# ----------------------------
with TABS[4]:
    st.markdown("### AI-ish Insights (No API Key)")
    st.caption("Heuristic insights + quick answers. For full LLM insights, hook up your own API or local model endpoint.")

    if df.empty:
        st.warning("No data loaded.")
    else:
        # Heuristic insights on top revenue groups
        rev_col = next((c for c in ["revenue", "sales", "amount", "value"] if c in df.columns), None)
        if rev_col is None and num_cols:
            rev_col = num_cols[0]

        insights = []
        try:
            # Top segments by revenue
            for c in [col for col in df.columns if df[col].dtype == object][:2]:
                top = df.groupby(c)[rev_col].sum().sort_values(ascending=False)
                if len(top) >= 2:
                    a, b = top.index[0], top.index[1]
                    uplift_pct = (top.iloc[0] - top.iloc[1]) / (top.iloc[1] if top.iloc[1] else 1) * 100
                    insights.append(f"Focus on **{a}** within **{c}** – it leads by {uplift_pct:,.1f}% vs {b}.")
        except Exception:
            pass

        # Momentum
        if date_col and rev_col in df.columns:
            tmp = df.assign(_d=pd.to_datetime(df[date_col]).dt.to_period("M").dt.start_time)
            m = tmp.groupby("_d")[rev_col].sum()
            if len(m) >= 3:
                gr1 = (m.iloc[-1] - m.iloc[-2]) / (m.iloc[-2] if m.iloc[-2] else 1) * 100
                gr2 = (m.iloc[-2] - m.iloc[-3]) / (m.iloc[-3] if m.iloc[-3] else 1) * 100
                dir_txt = "accelerating" if gr1 > gr2 else "slowing"
                insights.append(f"Growth is **{dir_txt}**: last month {gr1:,.1f}% vs prior {gr2:,.1f}%.")

        if not insights:
            insights.append("Data is diverse – consider defining a revenue/units column for sharper insights.")

        st.markdown("\n".join([f"- {x}" for x in insights]))

        # Quick QA
        st.divider()
        st.markdown("**Ask a quick question** (e.g., *Which region had highest revenue?*)")
        q = st.text_input("Question")
        if q:
            try:
                answer = ""
                ql = q.lower()
                if any(k in ql for k in ["highest", "top", "max"]) and rev_col:
                    # try find a categorical column requested
                    target_cat = None
                    for c in [col for col in df.columns if df[col].dtype == object]:
                        if c.lower() in ql:
                            target_cat = c; break
                    if target_cat is None and [col for col in df.columns if df[col].dtype == object]:
                        target_cat = [col for col in df.columns if df[col].dtype == object][0]
                    top = df.groupby(target_cat)[rev_col].sum().sort_values(ascending=False)
                    answer = f"**{top.index[0]}** leads in {target_cat} with £{top.iloc[0]:,.0f}."
                elif any(k in ql for k in ["trend", "growth"]) and date_col and rev_col:
                    m = df.assign(_d=pd.to_datetime(df[date_col]).dt.to_period("M").dt.start_time).groupby("_d")[rev_col].sum()
                    answer = f"Last 3 months: {', '.join([f'£{v:,.0f}' for v in m.tail(3)])}."
                else:
                    answer = "Try asking about *highest/lowest by [category]* or *trend/growth* questions."
                st.info(answer)
            except Exception as e:
                st.error(f"Could not answer: {e}")

# ----------------------------
# TAB 6: 🕒 Early Access Waitlist (Google Form)
# ----------------------------
with TABS[5]:
    st.markdown("### Early Access Waitlist")
    st.caption("Choose one: **iframe embed** (easiest) or **Direct POST** (captures email inside this app). Both options below.")

    # A) IFRAME EMBED – simplest
    with st.expander("Option A – Embed your Google Form (iframe)", expanded=bool(WAITLIST_CFG.google_form_iframe_url)):
        form_url = st.text_input("Google Form embed URL", value=WAITLIST_CFG.google_form_iframe_url or "", placeholder="https://docs.google.com/forms/d/e/<FORM_ID>/viewform?embedded=true")
        if form_url:
            st.components.v1.iframe(src=form_url, height=700, scrolling=True)
            st.success("Embedded form displayed above.")
        else:
            st.info("Paste the Google Form 'embedded' URL to show it here.")

    # B) DIRECT POST – capture email/name then submit to Google Forms
    with st.expander("Option B – Direct submit to Google Form (email capture)", expanded=WAITLIST_CFG.use_direct_post):
        if requests is None:
            st.error("The 'requests' library is required for direct submit. Add 'requests' to requirements.txt.")
        else:
            colA, colB = st.columns(2)
            with colA:
                form_id = st.text_input("Google Form ID", value=WAITLIST_CFG.form_id or "")
                email_entry = st.text_input("Email entry id (entry.xxxxx)", value=WAITLIST_CFG.email_entry_id or "")
                name_entry = st.text_input("Name entry id (optional)", value=WAITLIST_CFG.name_entry_id or "")
                extra_entry = st.text_input("Additional free-text entry id (optional)", value=WAITLIST_CFG.extra_entry_id or "")
            with colB:
                st.markdown("**What users will fill**")
                name_val = st.text_input("Your name (optional)")
                email_val = st.text_input("Your email", placeholder="you@example.com")
                extra_val = st.text_area("Anything else? (optional)")
                consent = st.checkbox("I agree to be contacted about Early Access.")
                submit = st.button("Join Waitlist ✅")

            if submit:
                if not (form_id and email_entry and email_val and consent):
                    st.error("Missing required fields: form id, email entry id, email, and consent.")
                else:
                    try:
                        url = f"https://docs.google.com/forms/d/e/{form_id}/formResponse"
                        payload = {email_entry: email_val}
                        if name_entry and name_val:
                            payload[name_entry] = name_val
                        if extra_entry and extra_val:
                            payload[extra_entry] = extra_val

                        # Optional Google Form params – not strictly necessary
                        payload.update({
                            "fvv": 1,
                            "partialResponse": [],
                            "pageHistory": 0,
                            "fbzx": "-1234567890"
                        })

                        resp = requests.post(url, data=payload, timeout=10)
                        if resp.status_code == 200:
                            st.success("You're on the waitlist! Check your inbox for a confirmation shortly.")
                            st.toast("Waitlist joined! 🎉")
                        else:
                            st.warning(f"Google Form returned status {resp.status_code}. We'll also capture locally as backup.")
                            # local backup
                            row = {"ts": pd.Timestamp.utcnow().isoformat(), "name": name_val, "email": email_val, "extra": extra_val}
                            try:
                                key = "_local_waitlist"
                                wl = st.session_state.get(key, [])
                                wl.append(row)
                                st.session_state[key] = wl
                                st.success("Saved to local session backup.")
                            except Exception:
                                st.error("Failed to store local backup.")
                    except Exception as e:
                        st.error(f"Submit failed: {e}")

        st.markdown("---")
        st.markdown("**Debug help**: Ensure the form is set to accept responses and Email/Name questions are not required unless you pass those fields.")

    st.divider()
    st.markdown("#### Simple native (no Google) capture – CSV download backup")
    st.caption("If you prefer not to use Google Forms, capture emails here and download a CSV.")
    with st.form("native_capture"):
        n_name = st.text_input("Name (optional)")
        n_email = st.text_input("Email")
        n_notes = st.text_area("Notes (optional)")
        n_ok = st.checkbox("I agree to be contacted about Early Access.")
        n_submit = st.form_submit_button("Add to local list")
    if n_submit:
        if not (n_email and n_ok):
            st.error("Email and consent required.")
        else:
            row = {"ts": pd.Timestamp.utcnow().isoformat(), "name": n_name, "email": n_email, "notes": n_notes}
            key = "_native_waitlist"
            bag = st.session_state.get(key, [])
            bag.append(row)
            st.session_state[key] = bag
            st.success("Added.")

    bag = st.session_state.get("_native_waitlist", [])
    if bag:
        nd = pd.DataFrame(bag)
        st.dataframe(nd.tail(200), use_container_width=True)
        st.download_button("Download local waitlist CSV", nd.to_csv(index=False).encode("utf-8"), file_name="waitlist_local.csv")

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.caption("AI BI Dashboard • MVP → GTM • Polished UI, error-handled, and Early Access ready.")
