# AI Revenue Recovery – Stable All-in-One (Filters Safe, Dates Safe, URL Sync, KPIs, Charts)
# Run: streamlit run streamlit_app.py

import os, sqlite3, io, json, re, uuid
from datetime import datetime, timedelta
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# Optional dependencies (all guarded)
PLOTLY_OK = True
try:
    import plotly.express as px
    import plotly.graph_objects as go
except Exception:
    PLOTLY_OK = False

SK_OK = True
try:
    from sklearn.ensemble import IsolationForest
except Exception:
    SK_OK = False
    IsolationForest = None

PROPHET_OK = True
try:
    from prophet import Prophet
except Exception:
    PROPHET_OK = False

# ---------------------------------------------------------------------
# Page config + crisp UI
# ---------------------------------------------------------------------
st.set_page_config(page_title="AI Revenue Recovery", page_icon="💰", layout="wide")
try:
    alt.data_transformers.disable_max_rows()
except Exception:
    pass

st.markdown("""
<style>
  html, body, .stApp { -webkit-font-smoothing: antialiased; -moz-osx-font-smoothing: grayscale; }
  * { transform: none !important; filter: none !important; backface-visibility: hidden; }
  .hero { padding:1rem 1.2rem; border-radius:14px; background:linear-gradient(135deg,#6366F1,#EC4899); color:#fff; box-shadow:0 6px 18px rgba(0,0,0,.12);}
  .card { background:#fff; border:1px solid #E5E7EB; border-radius:12px; padding:.9rem 1rem; }
  .kpi-title { font-size:.86rem; color:#6B7280; font-weight:600; margin-bottom:.2rem; }
  .kpi-value { font-size:1.5rem; font-weight:800; color:#111827; }
  .divider { height:1px; background:#E5E7EB; margin:1rem 0; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------
# Simple utils
# ---------------------------------------------------------------------
DB_PATH = "sales.db"

def money(x):
    try: return f"${float(x):,.0f}"
    except: return "$0"

def read_query_params():
    try: return st.query_params           # Streamlit >= 1.32
    except Exception: return st.experimental_get_query_params()

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

# ---------------------------------------------------------------------
# DB helpers (optional; app runs fine without an existing DB)
# ---------------------------------------------------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS sales_data(
      date TEXT, region TEXT, channel TEXT, segment TEXT, product TEXT, revenue REAL, customers INTEGER
    )""")
    conn.close()

@st.cache_data(show_spinner=False)
def load_from_db(db_path: str) -> pd.DataFrame:
    if not os.path.exists(db_path):
        return pd.DataFrame(columns=["date","region","channel","segment","product","revenue","customers"])
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM sales_data", conn)
    conn.close()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()
    return df

def save_to_db(df):
    conn = sqlite3.connect(DB_PATH)
    df.to_sql("sales_data", conn, if_exists="replace", index=False)
    conn.close()

# ---------------------------------------------------------------------
# Sample data (fallback)
# ---------------------------------------------------------------------
DEFAULT_CHANNELS = ["Online","Retail","Wholesale"]
DEFAULT_REGIONS  = ["AMER","EMEA","APAC"]
DEFAULT_SEGMENTS = ["Enterprise","Mid-Market","SMB","Startup"]

@st.cache_data
def make_sample_data(seed=42, days=90) -> pd.DataFrame:
    np.random.seed(seed)
    end = datetime.now()
    dates = pd.date_range(end - timedelta(days=days-1), periods=days, freq="D")
    products = ["Sports Gear","Video Games","Music Albums","Clothing"]
    rows = []
    for d in dates:
        week_mult = 0.7 if d.weekday() >= 5 else 1.0
        for rg in DEFAULT_REGIONS:
            for ch in DEFAULT_CHANNELS:
                for prod in products:
                    seg = np.random.choice(DEFAULT_SEGMENTS)
                    base = {"Sports Gear":1.2,"Video Games":1.1,"Music Albums":0.8,"Clothing":1.0}[prod]
                    regm = {"AMER":1.15,"EMEA":1.0,"APAC":0.9}[rg]
                    chm  = {"Online":1.2,"Retail":1.0,"Wholesale":0.85}[ch]
                    revenue = 5000 * week_mult * base * regm * chm * np.random.normal(1,0.25)
                    revenue = max(0, revenue)
                    cust = max(1, int(np.random.poisson(lam=max(1, revenue/200))))
                    rows.append({"date": d, "region": rg, "channel": ch, "segment": seg, "product": prod,
                                 "revenue": float(revenue), "customers": int(cust)})
    return pd.DataFrame(rows)

# ---------------------------------------------------------------------
# CSV loader (robust)
# ---------------------------------------------------------------------
def coerce_numeric(s): return pd.to_numeric(s, errors="coerce")

def load_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    cols = {c.lower().strip(): c for c in df.columns}

    # Date
    for cand in ["date","order_date","created_at","day"]:
        if cand in cols:
            df["date"] = pd.to_datetime(df[cols[cand]], errors="coerce"); break
    else:
        raise ValueError("CSV must include 'date' (or order_date/created_at/day).")

    # Categorical
    df["region"]  = df[cols["region"]].astype(str)  if "region"  in cols else "Unknown"
    df["channel"] = df[cols["channel"]].astype(str) if "channel" in cols else "Unknown"
    df["segment"] = df[cols["segment"]].astype(str) if "segment" in cols else "All"
    df["product"] = df[cols["product"]].astype(str) if "product" in cols else "All"

    # Revenue
    if "revenue" in cols:
        df["revenue"] = coerce_numeric(df[cols["revenue"]])
    elif "price" in cols and "quantity" in cols:
        df["revenue"] = coerce_numeric(df[cols["price"]]) * coerce_numeric(df[cols["quantity"]])
    else:
        raise ValueError("CSV must include 'revenue' or ('price' and 'quantity').")

    # Customers
    if "customers" in cols:
        df["customers"] = coerce_numeric(df[cols["customers"]]).fillna(1).astype(int)
    else:
        df["customers"] = np.maximum(1, (df["revenue"]/np.maximum(1.0, df["revenue"].median()/5)).round()).astype(int)

    df = df.dropna(subset=["date","revenue"]).copy()
    df["revenue"] = df["revenue"].clip(lower=0)
    return df[["date","region","channel","segment","product","revenue","customers"]]

# ---------------------------------------------------------------------
# Anomalies + Forecast (guarded fallbacks)
# ---------------------------------------------------------------------
def detect_anomalies(daily_df: pd.DataFrame) -> pd.DataFrame:
    dd = daily_df.sort_values("date").copy()
    if dd.empty or "revenue" not in dd:
        return dd.head(0)

    if SK_OK:
        x = dd.copy()
        x["day_of_week"] = x["date"].dt.dayofweek
        x["month"] = x["date"].dt.month
        x["revenue_lag1"] = x["revenue"].shift(1)
        x["revenue_lag7"] = x["revenue"].shift(7)
        x = x.dropna()
        if x.empty: return x
        feats = x[["revenue","day_of_week","month","revenue_lag1","revenue_lag7"]].values
        iso = IsolationForest(contamination=0.1, random_state=42)
        x["anomaly"] = iso.fit_predict(feats)
        x["anomaly_score"] = iso.score_samples(feats)
        return x[x["anomaly"] == -1][["date","revenue","anomaly_score"]].copy()

    # Fallback: rolling z-score
    r = dd.copy()
    r["rev_ma7"] = r["revenue"].rolling(7, min_periods=3).mean()
    r["rev_sd7"] = r["revenue"].rolling(7, min_periods=3).std().replace(0, np.nan)
    r["z"] = (r["revenue"] - r["rev_ma7"]) / r["rev_sd7"]
    out = r[(r["z"].abs() > 2.5) & r["rev_sd7"].notna()].copy()
    out["anomaly_score"] = -out["z"].abs()
    return out[["date","revenue","anomaly_score"]]

def forecast_prophet(daily: pd.DataFrame, days=30) -> pd.DataFrame:
    if not PROPHET_OK or daily.empty: return pd.DataFrame()
    df = daily.rename(columns={"date":"ds","revenue":"y"})
    try:
        m = Prophet()
        m.fit(df)
        future = m.make_future_dataframe(periods=days)
        fc = m.predict(future)
        out = fc[["ds","yhat"]].copy()
        out["type"] = np.where(out["ds"]<=df["ds"].max(),"Historical","Forecast")
        out = out.rename(columns={"ds":"date","yhat":"value"})
        return out
    except Exception:
        return pd.DataFrame()

def forecast_fallback(daily: pd.DataFrame, days=30) -> pd.DataFrame:
    if daily.empty: return pd.DataFrame()
    ser = daily.set_index("date")["revenue"].asfreq("D").fillna(method="ffill")
    df = ser.reset_index().rename(columns={"revenue":"revenue"})
    df["day_num"] = (df["date"] - df["date"].min()).dt.days
    # simple linear trend
    if df["day_num"].nunique() >= 2:
        X = df["day_num"].values; y = df["revenue"].values
        A = np.vstack([X, np.ones_like(X)]).T
        slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
        fut = np.arange(df["day_num"].max()+1, df["day_num"].max()+days+1)
        trend = slope * fut + intercept
    else:
        trend = np.repeat(df["revenue"].iloc[-1], days)
    # weekly seasonal naive
    dow_mean = df.groupby(df["date"].dt.weekday)["revenue"].mean()
    future_dates = pd.date_range(df["date"].max()+timedelta(days=1), periods=days)
    seas = np.array([dow_mean.get(d.weekday(), df["revenue"].mean()) for d in future_dates])
    pred = np.maximum(0.0, (trend + seas)/2.0)
    hist = df[["date","revenue"]].rename(columns={"revenue":"value"}); hist["type"]="Historical"
    fc = pd.DataFrame({"date":future_dates,"value":pred,"type":"Forecast"})
    return pd.concat([hist, fc], ignore_index=True)

# ---------------------------------------------------------------------
# Header + Hero
# ---------------------------------------------------------------------
st.markdown("<h2>💰 AI Revenue Recovery</h2>", unsafe_allow_html=True)
st.markdown("<div class='hero'>Upload your sales CSV → Filter → KPIs, anomalies, forecast, and charts.</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------
# Upload / Load
# ---------------------------------------------------------------------
init_db()
with st.expander("📂 Upload CSV (or use sample data)", expanded=False):
    up = st.file_uploader("CSV with columns: date, region, channel, segment, product, revenue, customers", type=["csv"])
    colu1, colu2 = st.columns(2)
    with colu1:
        if up is not None:
            try:
                df_up = load_csv(up)
                save_to_db(df_up)
                st.success("✅ Data uploaded & saved")
                st.cache_data.clear()  # clear DB cache
            except Exception as e:
                st.error(f"CSV error: {e}")
    with colu2:
        st.download_button("Download sample CSV", make_sample_data().to_csv(index=False),
                           file_name="sample_revenue_data.csv", mime="text/csv")

# Load source of truth (DB or sample)
df = load_from_db(DB_PATH)
if df.empty:
    df = make_sample_data()  # fallback
    try:
        save_to_db(df)
    except Exception:
        pass

# Ensure types
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"]).copy()
df["region"]  = df["region"].astype(str)
df["channel"] = df["channel"].astype(str)
df["segment"] = df["segment"].astype(str)
df["product"] = df["product"].astype(str)

# ---------------------------------------------------------------------
# HARDENED FILTERS (string options, clamped defaults, safe dates)
# ---------------------------------------------------------------------
def _parse_csv_list(val):
    if not val:
        return []
    if isinstance(val, list):  # streamlit >=1.32 may return list
        val = val[0] if val else ""
    return [x.strip() for x in str(val).split(",") if x.strip()]

def _clamp(defaults, options):
    if not isinstance(defaults, list):
        defaults = [defaults] if defaults else []
    return [d for d in defaults if d in options]

# Options as plain strings
all_regions  = sorted(df["region"].dropna().astype(str).unique().tolist())
all_channels = sorted(df["channel"].dropna().astype(str).unique().tolist())
all_segments = sorted(df["segment"].dropna().astype(str).unique().tolist())
all_products = sorted(df["product"].dropna().astype(str).unique().tolist())
# Safety: non-empty lists for widgets
if not all_regions:  all_regions  = ["—"]
if not all_channels: all_channels = ["—"]
if not all_segments: all_segments = ["—"]
if not all_products: all_products = ["—"]

# Safe date bounds (handle empty robustly)
if df["date"].isna().all():
    min_ts = pd.Timestamp(datetime.today().date())
    max_ts = min_ts
else:
    min_ts = pd.to_datetime(df["date"].min())
    max_ts = pd.to_datetime(df["date"].max())
# Fallback 60-day window (clamped)
fallback_start = max((max_ts - pd.Timedelta(days=60)).date(), min_ts.date())
fallback_end   = max_ts.date()

def _safe_date(val, fb, lo, hi):
    try:
        d = pd.to_datetime(val).date()
    except Exception:
        d = fb
    if d < lo: d = lo
    if d > hi: d = hi
    return d

qp = read_query_params()
regions_default  = _clamp(_parse_csv_list(qp.get("region","")),  all_regions)
channels_default = _clamp(_parse_csv_list(qp.get("channel","")), all_channels)
segments_default = _clamp(_parse_csv_list(qp.get("segment","")), all_segments)
products_default = _clamp(_parse_csv_list(qp.get("product","")), all_products)
start_default    = _safe_date(qp.get("start",""), fallback_start, min_ts.date(), max_ts.date())
end_default      = _safe_date(qp.get("end",""),   fallback_end,   min_ts.date(), max_ts.date())
if start_default > end_default:
    start_default, end_default = end_default, start_default

with st.sidebar:
    st.header("Filters")
    regions_sel  = st.multiselect("🌍 Region",  options=all_regions,  default=(regions_default or all_regions),  key="flt_regions")
    channels_sel = st.multiselect("📊 Channel", options=all_channels, default=(channels_default or all_channels), key="flt_channels")
    segments_sel = st.multiselect("👥 Segment",  options=all_segments, default=(segments_default or all_segments), key="flt_segments")
    products_sel = st.multiselect("📦 Product",  options=all_products, default=(products_default or all_products), key="flt_products")

    date_range = st.date_input(
        "📅 Date Range",
        value=(start_default, end_default),
        min_value=min_ts.date(),
        max_value=max_ts.date(),
        key="flt_dates"
    )

    st.markdown("---")
    use_plotly   = st.toggle("📈 Use Plotly charts", value=PLOTLY_OK, key="opt_plotly")
    run_forecast = st.toggle("🔮 Forecast (Prophet if available)", value=True, key="opt_fc")
    apply        = st.button("✅ Apply filters", key="btn_apply")

# Apply or initialize
if "applied_filters" not in st.session_state or apply:
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    else:
        start_date, end_date = min_ts, max_ts
    st.session_state.applied_filters = {
        "regions":  regions_sel or all_regions,
        "channels": channels_sel or all_channels,
        "segments": segments_sel or all_segments,
        "products": products_sel or all_products,
        "start": start_date,
        "end": end_date,
        "use_plotly": use_plotly,
        "run_forecast": run_forecast
    }

cfg = st.session_state.applied_filters
regions_sel, channels_sel = cfg["regions"], cfg["channels"]
segments_sel, products_sel = cfg["segments"], cfg["products"]
start_date, end_date = cfg["start"], cfg["end"]
use_plotly, run_forecast = cfg["use_plotly"], cfg["run_forecast"]

# Keep URL in sync (only valid values)
set_query_params(
    region=",".join(regions_sel),
    channel=",".join(channels_sel),
    segment=",".join(segments_sel),
    product=",".join(products_sel),
    start=start_date.date().isoformat(),
    end=end_date.date().isoformat(),
)

# Filtered data
mask = (
    df["region"].isin(regions_sel)
    & df["channel"].isin(channels_sel)
    & df["segment"].isin(segments_sel)
    & df["product"].isin(products_sel)
    & df["date"].between(start_date, end_date)
)
filtered = df.loc[mask].copy()
if filtered.empty:
    st.warning("No rows match these filters. Showing all data.")
    filtered = df.copy()

# ---------------------------------------------------------------------
# KPIs
# ---------------------------------------------------------------------
daily = filtered.groupby("date", as_index=False)["revenue"].sum().sort_values("date")
anoms = detect_anomalies(daily)
avg_day_rev = float(daily["revenue"].mean()) if not daily.empty else 0.0
potential_loss = float(max(0.0, (avg_day_rev*len(anoms) - anoms["revenue"].sum()) if not anoms.empty else 0.0))
by_channel = filtered.groupby("channel", as_index=False).agg(revenue=("revenue","sum"), customers=("customers","sum"))
by_channel["avg_deal_size"] = by_channel["revenue"] / by_channel["customers"].clip(lower=1)
target_ads = float(by_channel["avg_deal_size"].quantile(0.75)) if not by_channel.empty else 0.0
upsell_potential = float(((target_ads - by_channel["avg_deal_size"]).clip(lower=0) * by_channel["customers"]).sum()) if target_ads>0 else 0.0

# Forecast
fc = forecast_prophet(daily, 30) if (run_forecast and PROPHET_OK) else forecast_fallback(daily, 30)
future_sum = float(fc[fc["type"]=="Forecast"]["value"].sum()) if not fc.empty else 0.0
baseline_mean = float(daily["revenue"].tail(30).mean() if len(daily)>=30 else (daily["revenue"].mean() if not daily.empty else 0.0))
forecast_uplift = float(max(0.0, future_sum - baseline_mean*30))

# 30d comps
today = daily["date"].max() if not daily.empty else pd.Timestamp.today().normalize()
last_30_start = today - pd.Timedelta(days=29)
prev_30_start = today - pd.Timedelta(days=59)
prev_30_end   = today - pd.Timedelta(days=30)
last30 = daily[(daily["date"] >= last_30_start) & (daily["date"] <= today)]
prev30 = daily[(daily["date"] >= prev_30_start) & (daily["date"] <= prev_30_end)]
def pct_delta(cur, prev): 
    try:
        return 0.0 if prev in (None, 0) or np.isnan(prev) else 100.0*(cur-prev)/prev
    except Exception:
        return 0.0
total_rev_30 = float(last30["revenue"].sum()) if not last30.empty else float(daily["revenue"].sum() if not daily.empty else 0.0)
total_rev_prev = float(prev30["revenue"].sum()) if not prev30.empty else None
delta_total_rev = pct_delta(total_rev_30, total_rev_prev)
avg_rev_30 = float(last30["revenue"].mean()) if not last30.empty else float(daily["revenue"].mean() if not daily.empty else 0.0)
avg_rev_prev = float(prev30["revenue"].mean()) if not prev30.empty else None
anomaly_days = int(len(anoms))
anomaly_pct = (anomaly_days / max(1, len(daily))) * 100.0
top_ch = by_channel.sort_values("revenue", ascending=False).head(1)
top_channel_name = (top_ch["channel"].iloc[0] if not top_ch.empty else "—")
top_channel_share = float(top_ch["revenue"].iloc[0] / max(1.0, by_channel["revenue"].sum()) * 100.0) if not top_ch.empty else 0.0

# KPI cards
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"<div class='card'><div class='kpi-title'>Total Revenue (30d)</div><div class='kpi-value'>{money(total_rev_30)}</div><div class='kpi-title'>{delta_total_rev:+.1f}% vs prev</div></div>", unsafe_allow_html=True)
with c2:
    delta_avg = pct_delta(avg_rev_30, avg_rev_prev or avg_rev_30)
    st.markdown(f"<div class='card'><div class='kpi-title'>Avg Daily Revenue (30d)</div><div class='kpi-value'>{money(avg_rev_30)}</div><div class='kpi-title'>{delta_avg:+.1f}% vs prev</div></div>", unsafe_allow_html=True)
with c3:
    st.markdown(f"<div class='card'><div class='kpi-title'>Anomaly Days</div><div class='kpi-value'>{anomaly_days}</div><div class='kpi-title'>{anomaly_pct:.1f}% of days</div></div>", unsafe_allow_html=True)
with c4:
    st.markdown(f"<div class='card'><div class='kpi-title'>Top Channel</div><div class='kpi-value'>{top_channel_name}</div><div class='kpi-title'>{top_channel_share:.1f}% share</div></div>", unsafe_allow_html=True)

c5, c6, c7 = st.columns(3)
with c5: st.markdown(f"<div class='card'><div class='kpi-title'>Recoverable (Anomalies)</div><div class='kpi-value'>{money(potential_loss)}</div></div>", unsafe_allow_html=True)
with c6: st.markdown(f"<div class='card'><div class='kpi-title'>Upsell Potential</div><div class='kpi-value'>{money(upsell_potential)}</div></div>", unsafe_allow_html=True)
with c7: st.markdown(f"<div class='card'><div class='kpi-title'>30d Forecast Uplift</div><div class='kpi-value'>{money(forecast_uplift)}</div></div>", unsafe_allow_html=True)

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# Recovery moves
st.subheader("💡 Top Recovery Moves")
moves=[]
if potential_loss>0: moves.append(f"Plug anomaly days → recover ~{money(potential_loss)} (pricing/promos, billing, ops).")
if upsell_potential>0 and not by_channel.empty:
    worst = by_channel.nsmallest(1, "avg_deal_size")
    if not worst.empty: moves.append(f"Lift {worst['channel'].iloc[0]} avg deal size to 75th pct → unlock ~{money(upsell_potential)}.")
if forecast_uplift>0: moves.append(f"Prep capacity & promos for next 30 days → capture ~{money(forecast_uplift)}.")
if not moves: moves=["🎉 No major gaps detected — focus on targeted retention & upsell."]
for i,m in enumerate(moves,1): st.markdown(f"- **{i}. {m}**")

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------
# Charts (Plotly or Altair)
# ---------------------------------------------------------------------
a, b = st.columns([2,1])
with a:
    st.subheader("Revenue Trend & Anomalies")
    if use_plotly and PLOTLY_OK:
        fig = px.line(daily, x="date", y="revenue", title=None)
        if not anoms.empty:
            fig.add_trace(go.Scatter(
                x=anoms["date"], y=anoms["revenue"], mode="markers",
                marker=dict(size=9, color="#EF4444"), name="Anomalies"
            ))
        fig.update_layout(margin=dict(l=0,r=0,t=0,b=0), height=360)
        st.plotly_chart(fig, use_container_width=True, config={"responsive": True, "displayModeBar": False})
    else:
        base = alt.Chart(daily).encode(x=alt.X("date:T", title="Date"))
        line = base.mark_line(color="#6366F1").encode(
            y=alt.Y("revenue:Q", title="Revenue ($)", axis=alt.Axis(format="~s")),
            tooltip=[alt.Tooltip("date:T"), alt.Tooltip("revenue:Q", format=",.0f")]
        )
        if not anoms.empty:
            pts = alt.Chart(anoms).mark_point(size=85, filled=True, color="#EF4444").encode(
                x="date:T", y="revenue:Q", tooltip=["date:T", alt.Tooltip("revenue:Q", format=",.0f")]
            )
            st.altair_chart((line+pts).properties(height=360), use_container_width=True)
        else:
            st.altair_chart(line.properties(height=360), use_container_width=True)

with b:
    st.subheader("Revenue by Channel")
    by_ch = filtered.groupby("channel", as_index=False)["revenue"].sum().sort_values("revenue", ascending=False)
    if use_plotly and PLOTLY_OK:
        fig = px.bar(by_ch, x="revenue", y="channel", orientation="h", title=None)
        fig.update_layout(margin=dict(l=0,r=0,t=0,b=0), height=360)
        st.plotly_chart(fig, use_container_width=True, config={"responsive": True, "displayModeBar": False})
    else:
        st.altair_chart(alt.Chart(by_ch).mark_bar().encode(
            x=alt.X("revenue:Q", title="Revenue ($)", axis=alt.Axis(format="~s")),
            y=alt.Y("channel:N", sort="-x"),
            tooltip=["channel:N", alt.Tooltip("revenue:Q", format=",.0f")]
        ).properties(height=360), use_container_width=True)

st.subheader("Revenue by Region")
by_rg = filtered.groupby("region", as_index=False)["revenue"].sum().sort_values("revenue", ascending=False)
if use_plotly and PLOTLY_OK:
    fig = px.bar(by_rg, x="region", y="revenue", title=None)
    fig.update_layout(margin=dict(l=0,r=0,t=0,b=0), height=320)
    st.plotly_chart(fig, use_container_width=True, config={"responsive": True, "displayModeBar": False})
else:
    st.altair_chart(alt.Chart(by_rg).mark_bar().encode(
        x=alt.X("region:N", title="Region"),
        y=alt.Y("revenue:Q", title="Revenue ($)", axis=alt.Axis(format="~s")),
        tooltip=["region:N", alt.Tooltip("revenue:Q", format=",.0f")]
    ).properties(height=320), use_container_width=True)

st.subheader("30-Day Revenue Forecast " + ("(Prophet)" if run_forecast and PROPHET_OK and not fc.empty else "(Fallback)"))
if fc.empty:
    st.info("Not enough history to forecast yet or forecast failed.")
else:
    if use_plotly and PLOTLY_OK:
        fig = go.Figure()
        hist = fc[fc["type"]=="Historical"]
        fut  = fc[fc["type"]=="Forecast"]
        fig.add_trace(go.Scatter(x=hist["date"], y=hist["value"], name="Historical", mode="lines+markers"))
        fig.add_trace(go.Scatter(x=fut["date"], y=fut["value"], name="Forecast", mode="lines+markers"))
        fig.update_layout(margin=dict(l=0,r=0,t=0,b=0), height=360)
        st.plotly_chart(fig, use_container_width=True, config={"responsive": True, "displayModeBar": False})
    else:
        st.altair_chart(alt.Chart(fc).mark_line(point=True).encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("value:Q", title="Revenue ($)", axis=alt.Axis(format="~s")),
            color="type:N",
            tooltip=["type:N","date:T",alt.Tooltip("value:Q", format=",.0f")]
        ).properties(height=360), use_container_width=True)

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------
# Export & Share
# ---------------------------------------------------------------------
kpi_df = pd.DataFrame([
    {"metric":"Total Revenue (30d)","value": total_rev_30},
    {"metric":"Avg Daily Revenue (30d)","value": avg_rev_30},
    {"metric":"Anomaly Days","value": anomaly_days},
    {"metric":"Top Channel","value": top_channel_name},
    {"metric":"Recoverable (Anomalies)","value": potential_loss},
    {"metric":"Upsell Potential","value": upsell_potential},
    {"metric":"30d Forecast Uplift","value": forecast_uplift},
])
cA, cB, cC = st.columns([1,1,2])
with cA: st.download_button("Filtered CSV", filtered.to_csv(index=False), "filtered_data.csv", "text/csv")
with cB: st.download_button("KPIs CSV", kpi_df.to_csv(index=False), "kpis.csv", "text/csv")
with cC:
    cur_qp = read_query_params()
    share_url = "?" + "&".join([f"{k}={','.join(v) if isinstance(v,list) else v}" for k,v in cur_qp.items()])
    st.text_input("Share this exact view (copy URL)", value=share_url)
