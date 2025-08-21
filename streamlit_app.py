# AI Revenue Recovery – Final Demo (fixed, URL-sync, GA server-fallback)
# Run: streamlit run streamlit_app.py

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
import requests, uuid

st.set_page_config(page_title="AI Revenue Recovery", page_icon="💰", layout="wide")

# =============================================================================
# GA4 MEASUREMENT PROTOCOL (server-side fallback)
# =============================================================================
# Replace these with your GA4 values:
GA_MEASUREMENT_ID = "G-XXXXXXXXXX"   # e.g., G-ABCD123456
GA_API_SECRET     = "YOUR_API_SECRET"  # Admin → Data Streams → Measurement Protocol API secret

def ga_event_server(name: str, client_id=None, **params):
    """
    Fire a GA4 Measurement Protocol event from the server.
    Works even when client-side JS is blocked.
    """
    if not client_id:
        client_id = st.session_state.get("client_id")
        if not client_id:
            client_id = str(uuid.uuid4())
            st.session_state.client_id = client_id

    payload = {
        "client_id": client_id,
        "events": [{"name": name, "params": params}],
    }
    try:
        r = requests.post(
            f"https://www.google-analytics.com/mp/collect"
            f"?measurement_id={GA_MEASUREMENT_ID}&api_secret={GA_API_SECRET}",
            json=payload,
            timeout=3,
        )
        # GA returns 204 No Content on success
        if r.status_code != 204:
            st.warning(f"GA server event failed: {r.text}")
    except Exception as e:
        st.error(f"GA server event error: {e}")

# Optional: Tiny client-side helper (if you also add GA JS tag yourself)
def ga_event_client(name: str, **params):
    components.html(
        f"""<script>
            window.dataLayer = window.dataLayer || [];
            function gtag(){{ dataLayer.push(arguments); }}
            window.gaEvent = function(n,p){{ try{{gtag('event', n, p||{{}});}}catch(e){{}} }}
            window.gaEvent({name!r}, {params!r});
        </script>""",
        height=0,
    )

# =============================================================================
# STYLING
# =============================================================================
st.markdown(
    """
    <style>
      .hero {padding:1.25rem 1.5rem;border-radius:16px;
             background:linear-gradient(135deg,#6366F1,#EC4899);color:white;}
      .card {background:white;border-radius:12px;padding:1rem;box-shadow:0 2px 12px rgba(0,0,0,.06);}
      .kpi  {font-size:1.8rem;font-weight:800;margin:0;}
      .kpi-sub {opacity:.8;margin-top:.25rem;}
      .cta  {padding:1rem;border-radius:12px;background:#111827;color:#fff;}
    </style>
    """,
    unsafe_allow_html=True,
)

try:
    alt.data_transformers.disable_max_rows()
except Exception:
    pass

# =============================================================================
# QUERY PARAM HELPERS
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
                if k in qp:
                    del qp[k]
            else:
                qp[k] = v
    except Exception:
        st.experimental_set_query_params(**{k: v for k, v in kwargs.items() if v is not None})

# =============================================================================
# SAMPLE DATA
# =============================================================================
DEFAULT_CHANNELS = ["Direct Sales","Partner","Online","Retail","Wholesale"]
DEFAULT_REGIONS  = ["AMER","EMEA","APAC"]
DEFAULT_SEGMENTS = ["Enterprise","Mid-Market","SMB","Startup"]

@st.cache_data
def make_sample_data(seed=17, days=120) -> pd.DataFrame:
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
                customers = max(1, int(np.random.poisson(lam=max(1, revenue/1800))))
                rows.append({
                    "date": d, "region": rg, "channel": ch, "segment": segment, "product": product,
                    "revenue": max(0, float(revenue)), "customers": customers
                })
    return pd.DataFrame(rows)

# =============================================================================
# CSV LOADER
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
        raise ValueError("CSV must include a 'date' column (or order_date/created_at/day).")
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
    return df.dropna(subset=["date","revenue"])[["date","region","channel","segment","product","revenue","customers"]]

# =============================================================================
# ML HELPERS
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
    return dd[dd["anomaly"] == -1]

def make_forecast_ensemble(daily_df: pd.DataFrame, days=30) -> pd.DataFrame:
    if daily_df.empty: return pd.DataFrame(columns=["date","value","type"])
    s = daily_df.set_index("date")["revenue"].asfreq("D").fillna(method="ffill")
    df = s.reset_index().rename(columns={"revenue":"revenue"})
    df["day_num"] = (df["date"] - df["date"].min()).dt.days
    if df["day_num"].nunique() >= 2:
        lr = LinearRegression().fit(df[["day_num"]].values, df["revenue"].values)
        future_days = np.arange(df["day_num"].max()+1, df["day_num"].max()+days+1).reshape(-1,1)
        pred_trend = lr.predict(future_days)
    else:
        pred_trend = np.repeat(df["revenue"].iloc[-1], days)
    dow_mean = df.groupby(df["date"].dt.weekday)["revenue"].mean()
    future_dates = pd.date_range(df["date"].max()+timedelta(days=1), periods=days)
    pred_seasonal = np.array([dow_mean.get(d.weekday(), df["revenue"].mean()) for d in future_dates])
    pred_ens = np.maximum(0.0, (pred_trend + pred_seasonal)/2)
    hist = df[["date","revenue"]].rename(columns={"revenue":"value"}); hist["type"] = "Historical"
    fcst = pd.DataFrame({"date":future_dates,"value":pred_ens,"type":"Forecast"})
    return pd.concat([hist, fcst])

def money(x): return f"${x:,.0f}"

# =============================================================================
# HERO
# =============================================================================
st.markdown(
    '<div class="hero"><h2>Recover $500K in Lost Revenue with AI-Powered Insights</h2>'
    '<p>Upload your sales data and instantly see how much revenue you can recover, upsell, and forecast with AI.</p></div>',
    unsafe_allow_html=True
)

# =============================================================================
# UPLOAD
# =============================================================================
with st.expander("Upload your CSV (or use sample data)"):
    up = st.file_uploader("Upload CSV", type=["csv"])
    if st.download_button(
        "Download sample CSV",
        make_sample_data().to_csv(index=False),
        file_name="sample_revenue_data.csv",
        mime="text/csv",
    ):
        ga_event_server("sample_download", file_name="sample_revenue_data.csv")

# =============================================================================
# LOAD
# =============================================================================
with st.spinner("Analyzing your data..."):
    if up is not None:
        try:
            df = load_csv(up)
            st.success("Data loaded ✅")
            ga_event_server("csv_loaded", file_name=getattr(up, "name", "uploaded.csv"), rows=int(len(df)))
        except Exception as e:
            st.error(f"CSV error: {e}")
            df = make_sample_data()
            ga_event_server("csv_load_failed", error=str(e))
    else:
        df = make_sample_data()
        st.info("Using sample data.")
        ga_event_server("using_sample_data", rows=int(len(df)))

# =============================================================================
# FILTERS + URL SYNC (region, channel, segment, product, date)
# =============================================================================
qp = read_query_params()

def split_or_all(s, all_vals, demo_defaults=None):
    """Parse comma string into list; use demo_defaults if provided and URL param is blank."""
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

# Preselect AMER + Online if no URL param (nice demo defaults)
sel_regions  = split_or_all(qp.get("region",""),  all_regions,  demo_defaults=["AMER"])
sel_channels = split_or_all(qp.get("channel",""), all_channels, demo_defaults=["Online"])
sel_segments = split_or_all(qp.get("segment",""), all_segments)
sel_products = split_or_all(qp.get("product",""), all_products)

# Date range (URL synced)
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

st.sidebar.header("Filters")
regions_sel   = st.sidebar.multiselect("🌍 Region",   all_regions,  default=sel_regions)
channels_sel  = st.sidebar.multiselect("📊 Channel",  all_channels, default=sel_channels)
segments_sel  = st.sidebar.multiselect("👥 Segment",  all_segments, default=sel_segments)
products_sel  = st.sidebar.multiselect("📦 Product",  all_products, default=sel_products)
date_sel = st.sidebar.date_input(
    "📅 Date Range",
    value=(start_default, end_default),
    min_value=min_date, max_value=max_date
)
if isinstance(date_sel, tuple) and len(date_sel) == 2:
    start_date, end_date = pd.to_datetime(date_sel[0]), pd.to_datetime(date_sel[1])
else:
    start_date, end_date = pd.to_datetime(date_sel), pd.to_datetime(date_sel)

# Sync to URL
set_query_params(
    region=",".join(regions_sel)   if regions_sel  else None,
    channel=",".join(channels_sel) if channels_sel else None,
    segment=",".join(segments_sel) if segments_sel else None,
    product=",".join(products_sel) if products_sel else None,
    start=start_date.date().isoformat() if start_date else None,
    end=end_date.date().isoformat()     if end_date   else None,
)

# Fire a server-side event if filters or dates changed
curr_filters = {
    "region": ",".join(regions_sel),
    "channel": ",".join(channels_sel),
    "segment": ",".join(segments_sel),
    "product": ",".join(products_sel),
    "start": start_date.date().isoformat(),
    "end": end_date.date().isoformat(),
}
if st.session_state.get("prev_filters") != curr_filters:
    ga_event_server("filters_change", **curr_filters)
    st.session_state.prev_filters = curr_filters

# Apply filters
df_f = df[
    (df["region"].isin(regions_sel)) &
    (df["channel"].isin(channels_sel)) &
    (df["segment"].isin(segments_sel)) &
    (df["product"].isin(products_sel)) &
    (df["date"].between(start_date, end_date))
].copy()

if df_f.empty:
    st.warning("No data for these filters. Showing all data.")
    df_f = df.copy()

# =============================================================================
# KPIs (money-first + executive strip)
# =============================================================================
daily = df_f.groupby("date", as_index=False)["revenue"].sum().sort_values("date")
avg_day_rev = daily["revenue"].mean() if not daily.empty else 0.0
anoms = detect_anomalies(daily)
potential_loss = max(0.0, (avg_day_rev * len(anoms) - anoms["revenue"].sum()) if not anoms.empty else 0.0)

by_channel = df_f.groupby("channel", as_index=False).agg(revenue=("revenue","sum"), customers=("customers","sum"))
by_channel["avg_deal_size"] = by_channel["revenue"] / by_channel["customers"].clip(lower=1)
target_ads = by_channel["avg_deal_size"].quantile(0.75) if not by_channel.empty else 0.0
upsell_potential = ((target_ads - by_channel["avg_deal_size"]).clip(lower=0) * by_channel["customers"]).sum() if target_ads > 0 else 0.0

# Ensemble forecast + fair uplift
fc = make_forecast_ensemble(daily, 30)
future_sum = fc[fc["type"] == "Forecast"]["value"].sum() if not fc.empty else 0.0
baseline_mean = daily["revenue"].tail(30).mean() if len(daily) >= 30 else (daily["revenue"].mean() if not daily.empty else 0.0)
forecast_uplift = max(0.0, future_sum - baseline_mean * 30)

# Fire server event for forecast
ga_event_server("forecast_computed", uplift=float(forecast_uplift), future_sum=float(future_sum), baseline=float(baseline_mean * 30))

# Executive KPI row (Total 30d, Avg 30d, Anomaly days, Top channel share)
today = daily["date"].max() if not daily.empty else pd.Timestamp.today().normalize()
last_30_start = today - pd.Timedelta(days=29)
prev_30_start = today - pd.Timedelta(days=59)
prev_30_end   = today - pd.Timedelta(days=30)
last30 = daily[(daily["date"] >= last_30_start) & (daily["date"] <= today)]
prev30 = daily[(daily["date"] >= prev_30_start) & (daily["date"] <= prev_30_end)]
def pct_delta(cur, prev): return (0.0 if (prev is None or prev == 0) else 100.0 * (cur - prev) / prev)
total_rev_30 = float(last30["revenue"].sum()) if not last30.empty else float(daily["revenue"].sum())
total_rev_prev = float(prev30["revenue"].sum()) if not prev30.empty else None
delta_total_rev = pct_delta(total_rev_30, total_rev_prev)
avg_rev_30 = float(last30["revenue"].mean()) if not last30.empty else float(daily["revenue"].mean() if not daily.empty else 0.0)
avg_rev_prev = float(prev30["revenue"].mean()) if not prev30.empty else None
delta_avg_rev = pct_delta(avg_rev_30, avg_rev_prev)
anomaly_days = int(len(anoms))
anomaly_pct = (anomaly_days / max(1, len(daily))) * 100.0
top_ch = by_channel.sort_values("revenue", ascending=False).head(1)
if not top_ch.empty:
    top_channel_name = str(top_ch["channel"].iloc[0])
    top_channel_share = float(top_ch["revenue"].iloc[0] / max(1.0, by_channel["revenue"].sum()) * 100.0)
else:
    top_channel_name, top_channel_share = "—", 0.0

k1, k2, k3, k4 = st.columns(4)
with k1: st.metric("Total Revenue (30d)", money(total_rev_30), f"{delta_total_rev:+.1f}% vs prev 30d" if total_rev_prev is not None else None)
with k2: st.metric("Avg Daily Revenue (30d)", money(avg_rev_30), f"{delta_avg_rev:+.1f}% vs prev 30d" if avg_rev_prev is not None else None)
with k3: st.metric("Anomaly Days", f"{anomaly_days} days", f"{anomaly_pct:.1f}% of days")
with k4: st.metric("Top Channel", top_channel_name, f"{top_channel_share:.1f}% share")

# Money-first cards
c1,c2,c3 = st.columns(3)
for c,v,l in zip(
    (c1,c2,c3),
    (potential_loss, upsell_potential, forecast_uplift),
    ("Recoverable (Anomalies)", "Upsell Potential", "30d Forecast Uplift")
):
    with c:
        st.markdown(f"<div class='card'><div class='kpi'>{money(v)}</div><div class='kpi-sub'>{l}</div></div>", unsafe_allow_html=True)

# =============================================================================
# TOP MOVES
# =============================================================================
st.subheader("💡 Top 3 Recovery Moves")
moves=[]
if potential_loss>0: moves.append(f"Plug anomaly days → recover ~{money(potential_loss)} (pricing/promos, billing, ops).")
if upsell_potential>0 and not by_channel.empty:
    weakest = by_channel.sort_values("avg_deal_size").head(1)
    if not weakest.empty:
        moves.append(f"Lift {weakest['channel'].iloc[0]} avg deal size to 75th pct → unlock ~{money(upsell_potential)}.")
if forecast_uplift>0: moves.append(f"Prep capacity & promos for next 30 days → capture ~{money(forecast_uplift)}.")
if not moves: moves=["🎉 No major gaps detected — focus on targeted retention & upsell."]
for i,m in enumerate(moves,1): st.markdown(f"- **{i}. {m}**")

st.divider()

# =============================================================================
# CHARTS
# =============================================================================
if not df_f.empty:
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
# WAITLIST (with GA server fallback)
# =============================================================================
st.divider()
st.subheader("🚀 Join Early Access")
with st.form("waitlist"):
    col1, col2 = st.columns(2)
    with col1: name  = st.text_input("Full name")
    with col2: email = st.text_input("Work email")
    use_case = st.text_input("What do you want to recover or optimize?")
    submitted = st.form_submit_button("Request Access")
    if submitted:
        if not name or not email:
            st.error("Need name + email")
            ga_event_server("waitlist_submit_failed", reason="missing_fields")
        else:
            st.success("Thanks! We'll be in touch ✅")
            domain = email.split("@")[-1]
            ga_event_server("waitlist_submit", name=name, email_domain=domain, use_case=use_case)

# =============================================================================
# FOOTER / SHARE LINK
# =============================================================================
st.divider()
cur_qp = read_query_params()
demo_url = "?" + "&".join([f"{k}={','.join(v) if isinstance(v, list) else v}" for k, v in cur_qp.items()])
st.markdown(
    f"<div class='cta'><b>Want a private pilot?</b> Upload your CSV & get insights.<br/>"
    f"🔗 <a href='{demo_url}' target='_blank' "
    f"onclick=\"window.gaEvent && window.gaEvent('share_link_click', {{href: '{demo_url}'}})\">"
    f"Share this exact view ↗</a></div>",
    unsafe_allow_html=True
)
