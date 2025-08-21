# Minimal AI Revenue Recovery Dashboard – Day 2 Demo
# Run: streamlit run streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="AI Revenue Recovery", page_icon="💰", layout="wide")

# ---------- Styling ----------
st.markdown(
    """
    <style>
      .hero {padding:1.25rem 1.5rem;border-radius:16px;
              background:linear-gradient(135deg,#6366F1,#EC4899);color:white;}
      .card {background:white;border-radius:12px;padding:1rem;box-shadow:0 2px 12px rgba(0,0,0,.06);}
      .kpi {font-size:1.8rem;font-weight:800;margin:0;}
      .kpi-sub {opacity:.8;margin-top:.25rem;}
      .cta {padding:1rem;border-radius:12px;background:#111827;color:#fff;}
    </style>
    """,
    unsafe_allow_html=True
)

DEFAULT_CHANNELS = ["Direct Sales", "Partner", "Online", "Retail", "Wholesale"]
DEFAULT_REGIONS  = ["AMER", "EMEA", "APAC"]
DEFAULT_SEGMENTS = ["Enterprise", "Mid-Market", "SMB", "Startup"]

def parse_query_params():
    try:
        return st.query_params  # Streamlit ≥ 1.32
    except Exception:
        return st.experimental_get_query_params()

def set_query_params(**kwargs):
    try:
        qp = st.query_params
        for k, v in kwargs.items():
            if v is None:
                if k in qp: del qp[k]
            else:
                qp[k] = v
    except Exception:
        st.experimental_set_query_params(**{k: v for k, v in kwargs.items() if v is not None})

def split_or_all(s, all_vals):
    vals = [x.strip() for x in s.split(",")] if isinstance(s, str) and s else []
    vals = [v for v in vals if v in all_vals]
    return vals or all_vals

@st.cache_data
def make_sample_data(seed=17, days=120):
    np.random.seed(seed)
    end = datetime.now()
    dates = pd.date_range(end - timedelta(days=days-1), end, freq="D")
    rows = []
    for d in dates:
        week_mult = 0.7 if d.weekday() >= 5 else 1.0
        for ch in DEFAULT_CHANNELS:
            for rg in DEFAULT_REGIONS:
                base = {"Direct Sales":1.25,"Partner":1.1,"Online":1.0,"Retail":0.9,"Wholesale":0.85}[ch]
                regm = {"AMER":1.15,"EMEA":1.0,"APAC":0.95}[rg]
                segment = np.random.choice(DEFAULT_SEGMENTS, p=[0.25,0.3,0.3,0.15])
                product = np.random.choice(["Platform Pro","Analytics Suite","AI Insights","Basic"])
                revenue = 20000 * week_mult * base * regm * np.random.normal(1, 0.18)
                revenue = max(0, revenue)
                customers = max(1, int(np.random.poisson(lam=max(1, revenue/1800))))
                rows.append({
                    "date": d, "region": rg, "channel": ch, "segment": segment, "product": product,
                    "revenue": float(revenue), "customers": int(customers),
                })
    return pd.DataFrame(rows)

def coerce_numeric(s): return pd.to_numeric(s, errors="coerce")

def load_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    cols = {c.lower().strip(): c for c in df.columns}
    # date
    for cand in ["date","order_date","created_at","day"]:
        if cand in cols:
            df["date"] = pd.to_datetime(df[cols[cand]], errors="coerce"); break
    else:
        raise ValueError("CSV must include a 'date' (or 'order_date') column.")
    # region/channel
    df["region"] = df[cols["region"]].astype(str) if "region" in cols else "Unknown"
    df["channel"] = df[cols["channel"]].astype(str) if "channel" in cols else "Unknown"
    # optional
    df["segment"] = df[cols["segment"]].astype(str) if "segment" in cols else "All"
    df["product"] = df[cols["product"]].astype(str) if "product" in cols else "All"
    # revenue
    if "revenue" in cols:
        df["revenue"] = coerce_numeric(df[cols["revenue"]])
    elif "price" in cols and "quantity" in cols:
        df["revenue"] = coerce_numeric(df[cols["price"]]) * coerce_numeric(df[cols["quantity"]])
    else:
        raise ValueError("CSV must include 'revenue' or 'price' and 'quantity'.")
    # customers (optional)
    if "customers" in cols:
        df["customers"] = coerce_numeric(df[cols["customers"]]).fillna(1).astype(int)
    else:
        df["customers"] = np.maximum(1, (df["revenue"] / np.maximum(1.0, df["revenue"].median()/5)).round()).astype(int)
    df = df.dropna(subset=["date","revenue"]).copy()
    df["revenue"] = df["revenue"].clip(lower=0.0)
    return df[["date","region","channel","segment","product","revenue","customers"]]

def detect_anomalies(daily_df: pd.DataFrame) -> pd.DataFrame:
    dd = daily_df.sort_values("date").copy()
    dd["day_of_week"] = dd["date"].dt.dayofweek
    dd["month"] = dd["date"].dt.month
    dd["revenue_lag1"] = dd["revenue"].shift(1)
    dd["revenue_lag7"] = dd["revenue"].shift(7)
    dd = dd.dropna()
    if dd.empty: return dd
    feats = dd[["revenue","day_of_week","month","revenue_lag1","revenue_lag7"]].values
    iso = IsolationForest(contamination=0.1, random_state=42)
    dd["anomaly"] = iso.fit_predict(feats)
    dd["anomaly_score"] = iso.score_samples(feats)
    return dd[dd["anomaly"] == -1].copy()

def make_forecast(daily_df: pd.DataFrame, days: int = 30) -> pd.DataFrame:
    dd = daily_df.sort_values("date").copy()
    dd["day_num"] = (dd["date"] - dd["date"].min()).dt.days
    if dd["day_num"].nunique() < 2:
        dd["type"] = "Historical"
        return dd.rename(columns={"revenue":"value"})[["date","value","type"]]
    X = dd[["day_num"]].values; y = dd["revenue"].values
    model = LinearRegression().fit(X, y)
    future_days = np.arange(dd["day_num"].max()+1, dd["day_num"].max()+days+1).reshape(-1,1)
    pred = model.predict(future_days)
    future_dates = pd.date_range(dd["date"].max()+timedelta(days=1), periods=days)
    hist = dd[["date","revenue"]].rename(columns={"revenue":"value"}); hist["type"]="Historical"
    fcst = pd.DataFrame({"date":future_dates, "value":pred, "type":"Forecast"})
    return pd.concat([hist, fcst], ignore_index=True)

def money(x): return f"${x:,.0f}"

# ---------- Hero ----------
st.markdown(
    '<div class="hero"><h2 style="margin:0;">💰 Recover $500K in Lost Revenue</h2>'
    '<p style="margin:.2rem 0 0 0;">AI-powered insights for faster growth and fewer leaks.</p></div>',
    unsafe_allow_html=True
)

# ---------- Upload ----------
with st.expander("Upload your CSV (or use sample data)"):
    up = st.file_uploader("CSV with columns like: date, region, channel, revenue, customers…", type=["csv"])
    sample_btn = st.button("Download sample CSV")
    if sample_btn:
        sample = make_sample_data().to_csv(index=False)
        st.download_button("Save sample.csv", sample, file_name="sample_revenue_data.csv", mime="text/csv")

# Load data
if up is not None:
    try:
        df = load_csv(up)
        st.success("Data loaded from your CSV ✅")
    except Exception as e:
        st.error(f"CSV error: {e}")
        df = make_sample_data()
        st.info("Using sample data instead.")
else:
    df = make_sample_data()
    st.info("Using sample data. Upload your CSV to analyze your own revenue.")

# ---------- Filters with URL sync ----------
qp = parse_query_params()
regions_q = qp.get("region", "")
channels_q = qp.get("channel", "")
if isinstance(regions_q, list): regions_q = regions_q[0] if regions_q else ""
if isinstance(channels_q, list): channels_q = channels_q[0] if channels_q else ""

all_regions  = sorted(df["region"].dropna().unique().tolist())
all_channels = sorted(df["channel"].dropna().unique().tolist())
sel_regions  = split_or_all(regions_q,  all_regions or DEFAULT_REGIONS)
sel_channels = split_or_all(channels_q, all_channels or DEFAULT_CHANNELS)

st.sidebar.header("Filters")
regions_sel  = st.sidebar.multiselect("Region",  all_regions,  default=sel_regions)
channels_sel = st.sidebar.multiselect("Channel", all_channels, default=sel_channels)
set_query_params(region=",".join(regions_sel) if regions_sel else None,
                 channel=",".join(channels_sel) if channels_sel else None)

df_f = df[df["region"].isin(regions_sel) & df["channel"].isin(channels_sel)].copy()
if df_f.empty:
    st.warning("No data for the selected filters. Showing all data.")
    df_f = df.copy()

# ---------- KPIs ----------
daily = df_f.groupby("date", as_index=False)["revenue"].sum()
avg_day_rev = float(daily["revenue"].mean()) if not daily.empty else 0.0
anoms = detect_anomalies(daily)
potential_loss = float(max(0.0, (avg_day_rev * len(anoms) - anoms["revenue"].sum()) if not anoms.empty else 0.0))

by_channel = df_f.groupby("channel", as_index=False).agg(revenue=("revenue","sum"), customers=("customers","sum"))
by_channel["avg_deal_size"] = by_channel["revenue"] / by_channel["customers"].clip(lower=1)
target_ads = float(by_channel["avg_deal_size"].quantile(0.75)) if not by_channel.empty else 0.0
upsell_potential = float(((target_ads - by_channel["avg_deal_size"]).clip(lower=0) * by_channel["customers"]).sum()) if target_ads>0 else 0.0

fc = make_forecast(daily, days=30)
future  = fc[fc["type"]=="Forecast"]["value"].sum() if not fc.empty else 0.0
last30  = daily.tail(30)["revenue"].sum() if len(daily)>=1 else 0.0
forecast_uplift = float(max(0.0, future - last30))

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f"<div class='kpi'>{money(potential_loss)}</div><div class='kpi-sub'>Recoverable (Anomalies)</div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f"<div class='kpi'>{money(upsell_potential)}</div><div class='kpi-sub'>Upsell Potential</div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f"<div class='kpi'>{money(forecast_uplift)}</div><div class='kpi-sub'>30-day Forecast Uplift</div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.divider()
st.subheader("💡 Top 3 Recovery Moves")
moves = []

# 1) Plug anomaly losses
if potential_loss > 0:
    moves.append(f"Plug anomaly days → recover about {money(potential_loss)} (alert ops; investigate pricing/promos, payment errors).")

# 2) Lift underperforming channels
if upsell_potential > 0:
    worst = by_channel.sort_values("avg_deal_size").head(1)
    if not worst.empty:
        wc = worst["channel"].iloc[0]
        moves.append(f"Raise {wc} avg deal size to 75th-pct → unlock ~{money(upsell_potential)} (bundles, add-ons, min pricing).")

# 3) Capture forecast uplift
if forecast_uplift > 0:
    moves.append(f"Prepare capacity and promos for next 30 days → capture projected uplift of ~{money(forecast_uplift)}.")

if not moves:
    moves = ["Data looks healthy. Focus on targeted upsell and retention campaigns."]

for i, m in enumerate(moves, 1):
    st.markdown(f"- **{i}. {m}**")

# ---------- Charts ----------
if df_f.empty:
    st.info("No data to visualize.")
else:
    a, b = st.columns([2,1])
    with a:
        st.subheader("Revenue Trend & Anomalies")
        base = alt.Chart(daily).encode(x=alt.X("date:T", title="Date"))
        line = base.mark_line().encode(y=alt.Y("revenue:Q", title="Revenue ($)", axis=alt.Axis(format="~s")))
        if not anoms.empty:
            pts = alt.Chart(anoms).mark_point(size=80, filled=True, color="#EF4444").encode(
                x="date:T", y="revenue:Q",
                tooltip=["date:T", alt.Tooltip("revenue:Q", format=",.0f")]
            )
            st.altair_chart((line + pts).properties(height=340), use_container_width=True)
        else:
            st.altair_chart(line.properties(height=340), use_container_width=True)
    with b:
        st.subheader("Revenue by Channel")
        by_ch = df_f.groupby("channel", as_index=False)["revenue"].sum().sort_values("revenue", ascending=False)
        chart = alt.Chart(by_ch).mark_bar().encode(
            x=alt.X("revenue:Q", title="Revenue ($)", axis=alt.Axis(format="~s")),
            y=alt.Y("channel:N", sort="-x"),
            tooltip=["channel:N", alt.Tooltip("revenue:Q", format=",.0f")]
        ).properties(height=340)
        st.altair_chart(chart, use_container_width=True)

    st.subheader("Revenue by Region")
    by_rg = df_f.groupby("region", as_index=False)["revenue"].sum().sort_values("revenue", ascending=False)
    chart2 = alt.Chart(by_rg).mark_bar().encode(
        x=alt.X("region:N", title="Region"),
        y=alt.Y("revenue:Q", title="Revenue ($)", axis=alt.Axis(format="~s")),
        tooltip=["region:N", alt.Tooltip("revenue:Q", format=",.0f")]
    ).properties(height=300)
    st.altair_chart(chart2, use_container_width=True)

    st.subheader("30-Day Revenue Forecast")
    chart3 = alt.Chart(fc).mark_line(point=True).encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y("value:Q", title="Revenue ($)", axis=alt.Axis(format="~s")),
        color="type:N",
        tooltip=["type:N","date:T",alt.Tooltip("value:Q", format=",.0f")]
    ).properties(height=360)
    st.altair_chart(chart3, use_container_width=True)

st.divider()
st.divider()
st.subheader("🚀 Join Early Access")

with st.form("waitlist"):
    name = st.text_input("Full name")
    email = st.text_input("Work email")
    use_case = st.text_input("What do you want to recover or optimize?")
    submitted = st.form_submit_button("Request access")
    if submitted:
        if not name or not email:
            st.error("Please add name and email.")
        else:
            st.success("Thanks! We’ll be in touch within 24 hours. ✅")
st.markdown("<div class='cta'><b>Want a private pilot?</b> — Upload your latest CSV and we’ll surface your top recovery moves in minutes.</div>", unsafe_allow_html=True)
