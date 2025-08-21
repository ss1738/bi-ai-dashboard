# AI Revenue Recovery – Polished Demo (filters, downloads, assistant)
# Run: streamlit run streamlit_app.py

import os
import uuid
import json
import requests
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="AI Revenue Recovery", page_icon="💰", layout="wide")

# =============================================================================
# Styling (AI-inspired gradient + clean cards)
# =============================================================================
st.markdown(
    """
    <style>
      :root {
        --primary:#6366F1; --secondary:#EC4899; --success:#10B981; --dark:#111827;
      }
      .hero {padding:1.25rem 1.5rem;border-radius:16px;
             background:linear-gradient(135deg,var(--primary),var(--secondary));
             color:white; box-shadow:0 10px 30px rgba(0,0,0,.12);}
      .card {background:white;border-radius:14px;padding:1rem;box-shadow:0 4px 16px rgba(0,0,0,.07);}
      .kpi  {font-size:1.8rem;font-weight:800;margin:0;}
      .kpi-sub {opacity:.8;margin-top:.25rem;}
      .cta  {padding:1rem;border-radius:12px;background:var(--dark);color:#fff;}
      .muted { color:#6b7280; }
      .pill  {display:inline-block;padding:.15rem .6rem;border-radius:999px;background:#EEF2FF;color:#4338CA;font-weight:600;}
      .divider { height:1px;background:linear-gradient(90deg,#fff, #e5e7eb, #fff); margin:1rem 0;}
      .sticky-actions { position: sticky; top: 0; z-index: 5; background: transparent; padding-top: .25rem; }
      .chatbox { border-radius:16px; background:white; box-shadow:0 6px 24px rgba(0,0,0,.08); border:1px solid #eee; }
    </style>
    """,
    unsafe_allow_html=True,
)

try:
    alt.data_transformers.disable_max_rows()
except Exception:
    pass

# =============================================================================
# (Optional) GA4 Measurement Protocol server-side fallback
# =============================================================================
GA_MEASUREMENT_ID = os.environ.get("GA_MEASUREMENT_ID", "")  # e.g., G-XXXX
GA_API_SECRET = os.environ.get("GA_API_SECRET", "")

def ga_event_server(name: str, **params):
    if not (GA_MEASUREMENT_ID and GA_API_SECRET):  # no-op if not configured
        return
    cid = st.session_state.get("client_id") or str(uuid.uuid4())
    st.session_state.client_id = cid
    payload = {"client_id": cid, "events": [{"name": name, "params": params}]}
    try:
        requests.post(
            f"https://www.google-analytics.com/mp/collect"
            f"?measurement_id={GA_MEASUREMENT_ID}&api_secret={GA_API_SECRET}",
            json=payload, timeout=3,
        )
    except Exception:
        pass

# =============================================================================
# Query params helpers (works on new/old Streamlit)
# =============================================================================
def read_query_params():
    try:
        return st.query_params
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

# =============================================================================
# Sample data generator
# =============================================================================
DEFAULT_CHANNELS = ["Direct Sales","Partner","Online","Retail","Wholesale"]
DEFAULT_REGIONS  = ["AMER","EMEA","APAC"]
DEFAULT_SEGMENTS = ["Enterprise","Mid-Market","SMB","Startup"]
DEFAULT_PRODUCTS = ["Platform Pro","Analytics Suite","AI Insights","Basic"]

@st.cache_data
def make_sample_data(seed=17, days=150) -> pd.DataFrame:
    np.random.seed(seed)
    end = datetime.now()
    dates = pd.date_range(end - timedelta(days=days-1), end, freq="D")
    rows = []
    for d in dates:
        week_mult = 0.7 if d.weekday() >= 5 else 1.0
        season = 1.15 if d.month in (11,12) else (0.9 if d.month in (7,8) else 1.0)
        for ch in DEFAULT_CHANNELS:
            base = {"Direct Sales":1.25,"Partner":1.1,"Online":1.0,"Retail":0.9,"Wholesale":0.85}[ch]
            for rg in DEFAULT_REGIONS:
                regm = {"AMER":1.15,"EMEA":1.0,"APAC":0.95}[rg]
                segment = np.random.choice(DEFAULT_SEGMENTS, p=[0.25,0.30,0.30,0.15])
                product = np.random.choice(DEFAULT_PRODUCTS)
                base_rev = 20000 if segment=="Enterprise" else 12000 if segment=="Mid-Market" else 6000 if segment=="SMB" else 3000
                revenue = base_rev * week_mult * season * base * regm * np.random.normal(1, 0.18)
                revenue = max(0.0, revenue)
                customers = max(1, int(np.random.poisson(lam=max(1, revenue/1800))))
                rows.append({
                    "date": d, "region": rg, "channel": ch, "segment": segment, "product": product,
                    "revenue": float(revenue), "customers": int(customers),
                })
    df = pd.DataFrame(rows)
    # inject a few anomaly days (sharp drops)
    anom_days = np.random.choice(df["date"].dt.normalize().unique(), size=8, replace=False)
    for ad in anom_days:
        m = df["date"].dt.normalize().eq(ad)
        df.loc[m, "revenue"] *= np.random.uniform(0.18, 0.35)
        df.loc[m, "customers"] = np.maximum(1, (df.loc[m, "customers"] * np.random.uniform(0.3,0.6)).round()).astype(int)
    return df

# =============================================================================
# CSV loader (flexible headers)
# =============================================================================
def coerce_numeric(s): return pd.to_numeric(s, errors="coerce")

def load_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    cols = {c.lower().strip(): c for c in df.columns}
    # date
    for cand in ["date","order_date","created_at","day"]:
        if cand in cols:
            df["date"] = pd.to_datetime(df[cols[cand]], errors="coerce"); break
    else:
        raise ValueError("CSV must include 'date' (or order_date/created_at/day).")
    # dims
    df["region"]  = df[cols["region"]].astype(str)  if "region"  in cols else "Unknown"
    df["channel"] = df[cols["channel"]].astype(str) if "channel" in cols else "Unknown"
    df["segment"] = df[cols["segment"]].astype(str) if "segment" in cols else "All"
    df["product"] = df[cols["product"]].astype(str) if "product" in cols else "All"
    # revenue
    if "revenue" in cols:
        df["revenue"] = coerce_numeric(df[cols["revenue"]])
    elif "price" in cols and "quantity" in cols:
        df["revenue"] = coerce_numeric(df[cols["price"]]) * coerce_numeric(df[cols["quantity"]])
    else:
        raise ValueError("CSV must include 'revenue' or 'price' and 'quantity'.")
    # customers
    if "customers" in cols:
        df["customers"] = coerce_numeric(df[cols["customers"]]).fillna(1).astype(int)
    else:
        df["customers"] = np.maximum(1, (df["revenue"] / np.maximum(1.0, df["revenue"].median()/5)).round()).astype(int)
    df = df.dropna(subset=["date","revenue"]).copy()
    df["revenue"] = df["revenue"].clip(lower=0.0)
    return df[["date","region","channel","segment","product","revenue","customers"]]

# =============================================================================
# ML: anomalies + forecast (trend + weekly seasonality)
# =============================================================================
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

def make_forecast_ensemble(daily_df: pd.DataFrame, days=30) -> pd.DataFrame:
    if daily_df.empty: return pd.DataFrame(columns=["date","value","type"])
    s = daily_df.set_index("date")["revenue"].asfreq("D").fillna(method="ffill")
    df = s.reset_index().rename(columns={"revenue":"revenue"})
    df["day_num"] = (df["date"] - df["date"].min()).dt.days
    # trend
    if df["day_num"].nunique() >= 2:
        lr = LinearRegression().fit(df[["day_num"]].values, df["revenue"].values)
        future_days = np.arange(df["day_num"].max()+1, df["day_num"].max()+days+1).reshape(-1,1)
        trend = lr.predict(future_days)
    else:
        trend = np.repeat(df["revenue"].iloc[-1], days)
    # weekly seasonality
    dow_mean = df.groupby(df["date"].dt.weekday)["revenue"].mean()
    future_dates = pd.date_range(df["date"].max()+timedelta(days=1), periods=days)
    seasonal = np.array([dow_mean.get(d.weekday(), df["revenue"].mean()) for d in future_dates])
    pred = np.maximum(0.0, (trend + seasonal) / 2.0)
    hist = df[["date","revenue"]].rename(columns={"revenue":"value"}); hist["type"]="Historical"
    fc  = pd.DataFrame({"date": future_dates, "value": pred, "type":"Forecast"})
    return pd.concat([hist, fc])

def money(x): 
    try:
        return f"${float(x):,.0f}"
    except Exception:
        return "$0"

# =============================================================================
# Hero
# =============================================================================
st.markdown(
    '<div class="hero"><span class="pill">AI Revenue Recovery</span>'
    '<h2 style="margin:.25rem 0 0 0;">Recover $500K+ in Lost Revenue</h2>'
    '<p style="margin:.3rem 0 0 0;">Upload your CSV and instantly see anomalies to recover, upsell potential, and a 30-day forecast.</p></div>',
    unsafe_allow_html=True,
)

# =============================================================================
# Upload / Sample / Actions
# =============================================================================
with st.expander("Upload your CSV (or use sample data)"):
    up = st.file_uploader("CSV with columns like: date, region, channel, revenue, customers…", type=["csv"])
    sample_csv = make_sample_data().to_csv(index=False)
    st.download_button("⬇️ Download sample CSV", sample_csv, file_name="sample_revenue_data.csv", mime="text/csv")
    if st.button("Use sample data"):
        st.session_state["_use_sample"] = True

# Load data
with st.spinner("Analyzing your data..."):
    if up is not None and not getattr(st.session_state, "_use_sample", False):
        try:
            df = load_csv(up)
            st.success("Data loaded from your CSV ✅")
            ga_event_server("csv_loaded", rows=int(len(df)))
        except Exception as e:
            st.error(f"CSV error: {e}")
            df = make_sample_data()
            st.info("Using sample data instead.")
            ga_event_server("csv_load_failed", error=str(e))
    else:
        df = make_sample_data()
        st.info("Using sample data. Upload your CSV to analyze your own revenue.")
        ga_event_server("using_sample_data", rows=int(len(df)))

# =============================================================================
# Filters (URL-synced)
# =============================================================================
qp = read_query_params()

def split_or_all(s, all_vals, demo_defaults=None):
    if isinstance(s, str) and s:
        vals = [x.strip() for x in s.split(",")]
    elif demo_defaults:
        vals = [v for v in demo_defaults if v in all_vals]
    else:
        vals = all_vals
    return [v for v in vals if v in all_vals] or all_vals

all_regions  = sorted(df["region"].dropna().unique().tolist())
all_channels = sorted(df["channel"].dropna().unique().tolist())
all_segments = sorted(df["segment"].dropna().unique().tolist())
all_products = sorted(df["product"].dropna().unique().tolist())

sel_regions  = split_or_all(qp.get("region",""),  all_regions,  demo_defaults=["AMER"])
sel_channels = split_or_all(qp.get("channel",""), all_channels, demo_defaults=["Online"])
sel_segments = split_or_all(qp.get("segment",""), all_segments)
sel_products = split_or_all(qp.get("product",""), all_products)

# Date range
min_date, max_date = pd.to_datetime(df["date"]).min().date(), pd.to_datetime(df["date"]).max().date()
start_q, end_q = qp.get("start",""), qp.get("end","")
try:
    start_default = pd.to_datetime(start_q).date() if start_q else max(min_date, max_date - timedelta(days=59))
except Exception:
    start_default = max(min_date, max_date - timedelta(days=59))
try:
    end_default = pd.to_datetime(end_q).date() if end_q else max_date
except Exception:
    end_default = max_date

with st.sidebar:
    st.header("Filters")
    regions_sel   = st.multiselect("🌍 Region",   all_regions,  default=sel_regions)
    channels_sel  = st.multiselect("📊 Channel",  all_channels, default=sel_channels)
    segments_sel  = st.multiselect("👥 Segment",  all_segments, default=sel_segments)
    products_sel  = st.multiselect("📦 Product",  all_products, default=sel_products)
    date_sel = st.date_input("📅 Date Range", value=(start_default, end_default), min_value=min_date, max_value=max_date)

if isinstance(date_sel, tuple) and len(date_sel) == 2:
    start_date, end_date = pd.to_datetime(date_sel[0]), pd.to_datetime(date_sel[1])
else:
    start_date = end_date = pd.to_datetime(date_sel)

# Sync to URL
set_query_params(
    region=",".join(regions_sel)   if regions_sel  else None,
    channel=",".join(channels_sel) if channels_sel else None,
    segment=",".join(segments_sel) if segments_sel else None,
    product=",".join(products_sel) if products_sel else None,
    start=start_date.date().isoformat() if start_date is not None else None,
    end=end_date.date().isoformat()     if end_date   is not None else None,
)

# Filtered frame
df_f = df[
    (df["region"].isin(regions_sel)) &
    (df["channel"].isin(channels_sel)) &
    (df["segment"].isin(segments_sel)) &
    (df["product"].isin(products_sel)) &
    (df["date"].between(start_date, end_date))
].copy()
if df_f.empty:
    st.warning("No data for the selected filters. Showing all data.")
    df_f = df.copy()

# =============================================================================
# KPIs
# =============================================================================
daily = df_f.groupby("date", as_index=False)["revenue"].sum().sort_values("date")
anoms = detect_anomalies(daily)
avg_day_rev = float(daily["revenue"].mean()) if not daily.empty else 0.0
potential_loss = float(max(0.0, (avg_day_rev * len(anoms) - anoms["revenue"].sum()) if not anoms.empty else 0.0))

by_channel = df_f.groupby("channel", as_index=False).agg(revenue=("revenue","sum"), customers=("customers","sum"))
by_channel["avg_deal_size"] = by_channel["revenue"] / by_channel["customers"].clip(lower=1)
target_ads = float(by_channel["avg_deal_size"].quantile(0.75)) if not by_channel.empty else 0.0
upsell_potential = float(((target_ads - by_channel["avg_deal_size"]).clip(lower=0) * by_channel["customers"]).sum()) if target_ads>0 else 0.0

fc = make_forecast_ensemble(daily, 30)
future_sum = float(fc[fc["type"]=="Forecast"]["value"].sum()) if not fc.empty else 0.0
baseline_mean = float(daily["revenue"].tail(30).mean() if len(daily) >= 30 else (daily["revenue"].mean() if not daily.empty else 0.0))
forecast_uplift = float(max(0.0, future_sum - baseline_mean * 30))

# Executive strip
today = daily["date"].max() if not daily.empty else pd.Timestamp.today().normalize()
last_30_start = today - pd.Timedelta(days=29)
prev_30_start = today - pd.Timedelta(days=59)
prev_30_end   = today - pd.Timedelta(days=30)
last30 = daily[(daily["date"] >= last_30_start) & (daily["date"] <= today)]
prev30 = daily[(daily["date"] >= prev_30_start) & (daily["date"] <= prev_30_end)]
def pct_delta(cur, prev): return 0.0 if prev in (None, 0, np.nan) else 100.0 * (cur - prev) / prev

total_rev_30 = float(last30["revenue"].sum()) if not last30.empty else float(daily["revenue"].sum() if not daily.empty else 0.0)
total_rev_prev = float(prev30["revenue"].sum()) if not prev30.empty else None
delta_total_rev = pct_delta(total_rev_30, total_rev_prev)
avg_rev_30 = float(last30["revenue"].mean()) if not last30.empty else float(daily["revenue"].mean() if not daily.empty else 0.0)
avg_rev_prev = float(prev30["revenue"].mean()) if not prev30.empty else None
delta_avg_rev = pct_delta(avg_rev_30, avg_rev_prev)
anomaly_days = int(len(anoms))
anomaly_pct = (anomaly_days / max(1, len(daily))) * 100.0
top_ch = by_channel.sort_values("revenue", ascending=False).head(1)
top_channel_name = (top_ch["channel"].iloc[0] if not top_ch.empty else "—")
top_channel_share = float(top_ch["revenue"].iloc[0] / max(1.0, by_channel["revenue"].sum()) * 100.0) if not top_ch.empty else 0.0

k1, k2, k3, k4 = st.columns(4)
with k1: st.metric("Total Revenue (30d)", money(total_rev_30), f"{delta_total_rev:+.1f}% vs prev 30d" if total_rev_prev is not None else None)
with k2: st.metric("Avg Daily Revenue (30d)", money(avg_rev_30), f"{delta_avg_rev:+.1f}% vs prev 30d" if avg_rev_prev is not None else None)
with k3: st.metric("Anomaly Days", f"{anomaly_days} days", f"{anomaly_pct:.1f}% of days")
with k4: st.metric("Top Channel", top_channel_name, f"{top_channel_share:.1f}% share")

c1, c2, c3 = st.columns(3)
with c1: st.markdown(f"<div class='card'><div class='kpi'>{money(potential_loss)}</div><div class='kpi-sub'>Recoverable (Anomalies)</div></div>", unsafe_allow_html=True)
with c2: st.markdown(f"<div class='card'><div class='kpi'>{money(upsell_potential)}</div><div class='kpi-sub'>Upsell Potential</div></div>", unsafe_allow_html=True)
with c3: st.markdown(f"<div class='card'><div class='kpi'>{money(forecast_uplift)}</div><div class='kpi-sub'>30-day Forecast Uplift</div></div>", unsafe_allow_html=True)

# =============================================================================
# Top recovery moves
# =============================================================================
st.subheader("💡 Top 3 Recovery Moves")
moves = []
if potential_loss > 0: moves.append(f"Plug anomaly days → recover ~{money(potential_loss)} (pricing/promos, billing, ops).")
if upsell_potential > 0 and not by_channel.empty:
    worst = by_channel.nsmallest(1, "avg_deal_size")
    if not worst.empty: moves.append(f"Lift {worst['channel'].iloc[0]} avg deal size to 75th pct → unlock ~{money(upsell_potential)}.")
if forecast_uplift > 0: moves.append(f"Prep capacity & promos for next 30 days → capture ~{money(forecast_uplift)}.")
if not moves: moves = ["🎉 No major gaps detected — focus on targeted retention & upsell."]
for i, m in enumerate(moves, 1): st.markdown(f"- **{i}. {m}**")

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# =============================================================================
# Charts
# =============================================================================
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
            st.info("🎉 No revenue anomalies detected in the selected period.")
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
    if fc[fc["type"]=="Forecast"].empty:
        st.info("Not enough history to forecast yet. Add more data to unlock forecasts.")
    chart3 = alt.Chart(fc).mark_line(point=True).encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y("value:Q", title="Revenue ($)", axis=alt.Axis(format="~s")),
        color="type:N",
        tooltip=["type:N","date:T",alt.Tooltip("value:Q", format=",.0f")]
    ).properties(height=360)
    st.altair_chart(chart3, use_container_width=True)

# =============================================================================
# Downloads & share link
# =============================================================================
st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
st.subheader("⬇️ Export & Share")

kpi_rows = [
    {"metric":"Total Revenue (30d)","value": total_rev_30},
    {"metric":"Avg Daily Revenue (30d)","value": avg_rev_30},
    {"metric":"Anomaly Days","value": anomaly_days},
    {"metric":"Top Channel","value": top_channel_name},
    {"metric":"Recoverable (Anomalies)","value": potential_loss},
    {"metric":"Upsell Potential","value": upsell_potential},
    {"metric":"30-day Forecast Uplift","value": forecast_uplift},
]
kpi_df = pd.DataFrame(kpi_rows)

colA, colB, colC, colD = st.columns([1,1,1,2])
with colA:
    st.download_button("Filtered data CSV", df_f.to_csv(index=False), "filtered_data.csv", "text/csv")
with colB:
    st.download_button("KPI CSV", kpi_df.to_csv(index=False), "kpis.csv", "text/csv")
with colC:
    top_moves_txt = "\n".join(f"{i}. {m}" for i, m in enumerate(moves, 1))
    st.download_button("Top moves TXT", top_moves_txt, "top_moves.txt")

cur_qp = read_query_params()
share_url = "?" + "&".join([f"{k}={','.join(v) if isinstance(v,list) else v}" for k,v in cur_qp.items()])
with colD:
    st.text_input("Share this exact view (copy URL)", value=share_url)

# =============================================================================
# Assistant bot (lightweight, no external calls unless OPENAI_API_KEY is set)
# =============================================================================
st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
with st.expander("🤖 Assistant (beta) — ask how to recover more revenue", expanded=False):
    if "chat" not in st.session_state:
        st.session_state.chat = [
            {"role":"assistant","content":"Hi! I analyze your KPIs and suggest specific revenue recovery moves. Ask me anything."}
        ]

    # Local knowledge & live data fusion
    context = {
        "total_rev_30": money(total_rev_30),
        "avg_rev_30": money(avg_rev_30),
        "anomaly_days": anomaly_days,
        "recoverable": money(potential_loss),
        "upsell": money(upsell_potential),
        "uplift": money(forecast_uplift),
        "top_channel": top_channel_name,
        "top_channel_share": f"{top_channel_share:.1f}%",
        "weakest_channel": (by_channel.sort_values("avg_deal_size").head(1)["channel"].iloc[0] if not by_channel.empty else "—")
    }

    def local_assistant_reply(prompt: str) -> str:
        p = prompt.lower()
        if "anomal" in p or "recover" in p:
            return (f"You have **{context['anomaly_days']} anomaly days**. "
                    f"Estimated recoverable revenue is **{context['recoverable']}**. "
                    f"Start by auditing pricing/promotions and billing on those dates; "
                    f"stand up a playbook to prevent repeats.")
        if "upsell" in p or "deal size" in p:
            return (f"Target avg deal size 75th percentile. Estimated **upsell potential {context['upsell']}**. "
                    f"Focus first on **{context['weakest_channel']}** with add-ons & bundles.")
        if "forecast" in p or "next 30" in p:
            return (f"Projected 30-day uplift is **{context['uplift']}** vs baseline. "
                    f"Plan inventory & promos to capture peak days; stagger campaigns mid-week.")
        if "channel" in p:
            return (f"Top channel is **{context['top_channel']}** ({context['top_channel_share']} share). "
                    f"Shift budget from low-ROI channels, A/B pricing on {context['top_channel']}.")
        # default summary
        return (f"Summary: 30-day revenue **{context['total_rev_30']}**, avg/day **{context['avg_rev_30']}**. "
                f"Recoverable **{context['recoverable']}**, upsell **{context['upsell']}**, "
                f"forecast uplift **{context['uplift']}**. Ask me about anomalies, upsell, or forecasts.")

    # Optional OpenAI integration (if key provided & package installed)
    def openai_reply(prompt: str) -> str:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            return local_assistant_reply(prompt)
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            sys = ("You are a revenue recovery copilot. Use the provided KPIs to give crisp, "
                   "actionable, numbers-first guidance. Avoid fluff.")
            tools_blob = json.dumps(context)
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role":"system","content":sys},
                    {"role":"user","content":f"Live KPIs: {tools_blob}"},
                    {"role":"user","content":prompt}
                ],
                temperature=0.2,
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            return local_assistant_reply(prompt)

    # Chat UI
    for m in st.session_state.chat:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
    user_msg = st.chat_input("Ask about anomalies, upsell, forecast, channels…")
    if user_msg:
        st.session_state.chat.append({"role":"user","content":user_msg})
        with st.chat_message("user"): st.markdown(user_msg)
        reply = openai_reply(user_msg)
        st.session_state.chat.append({"role":"assistant","content":reply})
        with st.chat_message("assistant"): st.markdown(reply)

# =============================================================================
# Footer
# =============================================================================
st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
st.markdown("<div class='cta'><b>Want a private pilot?</b> — Upload your latest CSV and we’ll surface your top recovery moves in minutes.</div>", unsafe_allow_html=True)
