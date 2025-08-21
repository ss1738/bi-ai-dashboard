# AI Revenue Recovery – Polished + RAG + KB Manager + URL Crawler + PDF Export
# Run: streamlit run streamlit_app.py
#
# Add to requirements.txt:
# streamlit>=1.28
# pandas>=2
# numpy>=1.24
# altair>=5
# scikit-learn>=1.3
# PyPDF2>=3.0
# sentence-transformers>=2.2   # optional; falls back to TF-IDF if missing
# requests>=2
# beautifulsoup4>=4.12
# reportlab>=3.6

import os, io, uuid, json, pickle, tempfile, pathlib, re
import requests
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch

# Optional SBERT (auto-fallback)
SBERT_OK = False
try:
    from sentence_transformers import SentenceTransformer
    SBERT_OK = True
except Exception:
    SBERT_OK = False

st.set_page_config(page_title="AI Revenue Recovery", page_icon="💰", layout="wide")

# ============= Styling =========================================================
st.markdown(
    """
    <style>
      :root { --primary:#6366F1; --secondary:#EC4899; --success:#10B981; --dark:#111827; }
      .hero {padding:1.2rem 1.4rem;border-radius:16px;background:linear-gradient(135deg,var(--primary),var(--secondary));color:#fff;box-shadow:0 10px 30px rgba(0,0,0,.12);}
      .card {background:#fff;border-radius:14px;padding:1rem;box-shadow:0 4px 16px rgba(0,0,0,.07);}
      .kpi {font-size:1.8rem;font-weight:800;margin:0;}
      .kpi-sub {opacity:.8;margin-top:.25rem;}
      .cta {padding:1rem;border-radius:12px;background:var(--dark);color:#fff;}
      .divider {height:1px;background:linear-gradient(90deg,#fff,#e5e7eb,#fff);margin:1rem 0;}
      .muted { color:#6b7280; }
      .pill {display:inline-block;padding:.15rem .6rem;border-radius:999px;background:#EEF2FF;color:#4338CA;font-weight:600;}
      .chatbox { border-radius:16px; background:#fff; box-shadow:0 6px 24px rgba(0,0,0,.08); border:1px solid #eee; }
      .kb-table td, .kb-table th { padding: .3rem .5rem; border-bottom: 1px solid #eee; }
    </style>
    """,
    unsafe_allow_html=True,
)
try:
    alt.data_transformers.disable_max_rows()
except Exception:
    pass

# ============= Query params helpers ==========================================
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

# ============= Sample Data / CSV Loader ======================================
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
                rows.append({"date": d,"region": rg,"channel": ch,"segment": segment,"product": product,"revenue": float(revenue),"customers": customers})
    df = pd.DataFrame(rows)
    # Inject anomaly days
    anom_days = np.random.choice(df["date"].dt.normalize().unique(), size=8, replace=False)
    for ad in anom_days:
        m = df["date"].dt.normalize().eq(ad)
        df.loc[m, "revenue"]  *= np.random.uniform(0.18, 0.35)
        df.loc[m, "customers"] = np.maximum(1, (df.loc[m, "customers"] * np.random.uniform(0.3,0.6)).round()).astype(int)
    return df

def coerce_numeric(s): return pd.to_numeric(s, errors="coerce")

def load_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    cols = {c.lower().strip(): c for c in df.columns}
    for cand in ["date","order_date","created_at","day"]:
        if cand in cols:
            df["date"] = pd.to_datetime(df[cols[cand]], errors="coerce"); break
    else:
        raise ValueError("CSV must include 'date' (or order_date/created_at/day).")
    df["region"]  = df[cols["region"]].astype(str)  if "region"  in cols else "Unknown"
    df["channel"] = df[cols["channel"]].astype(str) if "channel" in cols else "Unknown"
    df["segment"] = df[cols["segment"]].astype(str) if "segment" in cols else "All"
    df["product"] = df[cols["product"]].astype(str) if "product" in cols else "All"
    if "revenue" in cols:
        df["revenue"] = coerce_numeric(df[cols["revenue"]])
    elif "price" in cols and "quantity" in cols:
        df["revenue"] = coerce_numeric(df[cols["price"]]) * coerce_numeric(df[cols["quantity"]])
    else:
        raise ValueError("CSV must include 'revenue' or 'price' and 'quantity'.")
    df["customers"] = coerce_numeric(df[cols["customers"]]).fillna(1).astype(int) if "customers" in cols \
                      else np.maximum(1,(df["revenue"]/np.maximum(1.0,df["revenue"].median()/5)).round()).astype(int)
    df = df.dropna(subset=["date","revenue"]).copy()
    df["revenue"] = df["revenue"].clip(lower=0.0)
    return df[["date","region","channel","segment","product","revenue","customers"]]

# ============= ML: Anomalies + Forecast ======================================
def detect_anomalies(daily_df: pd.DataFrame) -> pd.DataFrame:
    dd = daily_df.sort_values("date").copy()
    dd["day_of_week"] = dd["date"].dt.dayofweek
    dd["month"] = dd["date"].dt.month
    dd["revenue_lag1"] = dd["revenue"].shift(1)
    dd["revenue_lag7"] = dd["revenue"].shift(7)
    dd = dd.dropna()
    if dd.empty: return dd
    iso = IsolationForest(contamination=0.1, random_state=42)
    feats = dd[["revenue","day_of_week","month","revenue_lag1","revenue_lag7"]].values
    dd["anomaly"] = iso.fit_predict(feats)
    dd["anomaly_score"] = iso.score_samples(feats)
    return dd[dd["anomaly"] == -1].copy()

def make_forecast_ensemble(daily_df: pd.DataFrame, days=30) -> pd.DataFrame:
    if daily_df.empty: return pd.DataFrame(columns=["date","value","type"])
    s = daily_df.set_index("date")["revenue"].asfreq("D").fillna(method="ffill")
    df = s.reset_index().rename(columns={"revenue":"revenue"})
    df["day_num"] = (df["date"] - df["date"].min()).dt.days
    if df["day_num"].nunique() >= 2:
        lr = LinearRegression().fit(df[["day_num"]].values, df["revenue"].values)
        future_days = np.arange(df["day_num"].max()+1, df["day_num"].max()+days+1).reshape(-1,1)
        trend = lr.predict(future_days)
    else:
        trend = np.repeat(df["revenue"].iloc[-1], days)
    dow_mean = df.groupby(df["date"].dt.weekday)["revenue"].mean()
    future_dates = pd.date_range(df["date"].max()+timedelta(days=1), periods=days)
    seasonal = np.array([dow_mean.get(d.weekday(), df["revenue"].mean()) for d in future_dates])
    pred = np.maximum(0.0, (trend + seasonal) / 2.0)
    hist = df[["date","revenue"]].rename(columns={"revenue":"value"}); hist["type"]="Historical"
    fc  = pd.DataFrame({"date": future_dates, "value": pred, "type":"Forecast"})
    return pd.concat([hist, fc])

def money(x):
    try: return f"${float(x):,.0f}"
    except Exception: return "$0"

# ============= HERO ============================================================
st.markdown(
    '<div class="hero"><span class="pill">AI Revenue Recovery</span>'
    '<h2 style="margin:.25rem 0 0 0;">Recover $500K+ in Lost Revenue</h2>'
    '<p style="margin:.3rem 0 0 0;">Upload your CSV and instantly see anomalies to recover, upsell potential, and a 30-day forecast — now with a RAG Knowledge Base.</p></div>',
    unsafe_allow_html=True,
)

# ============= Upload / Sample ===============================================
with st.expander("Upload your CSV (or use sample data)"):
    up = st.file_uploader("CSV with columns like: date, region, channel, revenue, customers…", type=["csv"])
    st.download_button("⬇️ Download sample CSV", make_sample_data().to_csv(index=False),
                       file_name="sample_revenue_data.csv", mime="text/csv")
    if st.button("Use sample data"):
        st.session_state["_use_sample"] = True

with st.spinner("Analyzing your data..."):
    if up is not None and not st.session_state.get("_use_sample"):
        try:
            df = load_csv(up); st.success("Data loaded from your CSV ✅")
        except Exception as e:
            st.error(f"CSV error: {e}"); df = make_sample_data(); st.info("Using sample data instead.")
    else:
        df = make_sample_data(); st.info("Using sample data. Upload your CSV to analyze your own revenue.")

# ============= Filters (URL Sync) ============================================
qp = read_query_params()
def split_or_all(s, all_vals, demo_defaults=None):
    if isinstance(s, str) and s: vals = [x.strip() for x in s.split(",")]
    elif demo_defaults: vals = [v for v in demo_defaults if v in all_vals]
    else: vals = all_vals
    return [v for v in vals if v in all_vals] or all_vals

all_regions  = sorted(df["region"].unique());  all_channels = sorted(df["channel"].unique())
all_segments = sorted(df["segment"].unique()); all_products = sorted(df["product"].unique())

sel_regions  = split_or_all(qp.get("region",""),  all_regions,  ["AMER"])
sel_channels = split_or_all(qp.get("channel",""), all_channels, ["Online"])
sel_segments = split_or_all(qp.get("segment",""), all_segments)
sel_products = split_or_all(qp.get("product",""), all_products)

min_date, max_date = pd.to_datetime(df["date"]).min().date(), pd.to_datetime(df["date"]).max().date()
def to_date(s, default):
    try: return pd.to_datetime(s).date()
    except Exception: return default
start_default = to_date(qp.get("start",""), max(min_date, max_date - timedelta(days=59)))
end_default   = to_date(qp.get("end",""),   max_date)

with st.sidebar:
    st.header("Filters")
    regions_sel   = st.multiselect("🌍 Region",   all_regions,  default=sel_regions)
    channels_sel  = st.multiselect("📊 Channel",  all_channels, default=sel_channels)
    segments_sel  = st.multiselect("👥 Segment",  all_segments, default=sel_segments)
    products_sel  = st.multiselect("📦 Product",  all_products, default=sel_products)
    date_sel = st.date_input("📅 Date Range", value=(start_default, end_default), min_value=min_date, max_value=max_date)

start_date, end_date = pd.to_datetime(date_sel[0]), pd.to_datetime(date_sel[1])
set_query_params(
    region=",".join(regions_sel), channel=",".join(channels_sel),
    segment=",".join(segments_sel), product=",".join(products_sel),
    start=start_date.date().isoformat(), end=end_date.date().isoformat()
)

df_f = df[
    (df["region"].isin(regions_sel)) &
    (df["channel"].isin(channels_sel)) &
    (df["segment"].isin(segments_sel)) &
    (df["product"].isin(products_sel)) &
    (df["date"].between(start_date, end_date))
].copy()
if df_f.empty: st.warning("No data for the selected filters. Showing all data."); df_f = df.copy()

# ============= KPIs ===========================================================
daily = df_f.groupby("date", as_index=False)["revenue"].sum().sort_values("date")
anoms = detect_anomalies(daily)
avg_day_rev = float(daily["revenue"].mean()) if not daily.empty else 0.0
potential_loss = float(max(0.0, (avg_day_rev*len(anoms) - anoms["revenue"].sum()) if not anoms.empty else 0.0))

by_channel = df_f.groupby("channel", as_index=False).agg(revenue=("revenue","sum"), customers=("customers","sum"))
by_channel["avg_deal_size"] = by_channel["revenue"] / by_channel["customers"].clip(lower=1)
target_ads = float(by_channel["avg_deal_size"].quantile(0.75)) if not by_channel.empty else 0.0
upsell_potential = float(((target_ads - by_channel["avg_deal_size"]).clip(lower=0) * by_channel["customers"]).sum()) if target_ads>0 else 0.0

fc = make_forecast_ensemble(daily, 30)
future_sum = float(fc[fc["type"]=="Forecast"]["value"].sum()) if not fc.empty else 0.0
baseline_mean = float(daily["revenue"].tail(30).mean() if len(daily)>=30 else (daily["revenue"].mean() if not daily.empty else 0.0))
forecast_uplift = float(max(0.0, future_sum - baseline_mean*30))

today = daily["date"].max() if not daily.empty else pd.Timestamp.today().normalize()
last_30_start = today - pd.Timedelta(days=29)
prev_30_start = today - pd.Timedelta(days=59)
prev_30_end   = today - pd.Timedelta(days=30)
last30 = daily[(daily["date"] >= last_30_start) & (daily["date"] <= today)]
prev30 = daily[(daily["date"] >= prev_30_start) & (daily["date"] <= prev_30_end)]
def pct_delta(cur, prev): return 0.0 if (prev is None or prev==0 or np.isnan(prev)) else 100.0*(cur-prev)/prev

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

k1,k2,k3,k4 = st.columns(4)
with k1: st.metric("Total Revenue (30d)", money(total_rev_30), f"{delta_total_rev:+.1f}% vs prev 30d" if total_rev_prev is not None else None)
with k2: st.metric("Avg Daily Revenue (30d)", money(avg_rev_30), f"{pct_delta(avg_rev_30, avg_rev_prev):+.1f}% vs prev 30d" if avg_rev_prev is not None else None)
with k3: st.metric("Anomaly Days", f"{anomaly_days} days", f"{anomaly_pct:.1f}% of days")
with k4: st.metric("Top Channel", top_channel_name, f"{top_channel_share:.1f}% share")

c1,c2,c3 = st.columns(3)
with c1: st.markdown(f"<div class='card'><div class='kpi'>{money(potential_loss)}</div><div class='kpi-sub'>Recoverable (Anomalies)</div></div>", unsafe_allow_html=True)
with c2: st.markdown(f"<div class='card'><div class='kpi'>{money(upsell_potential)}</div><div class='kpi-sub'>Upsell Potential</div></div>", unsafe_allow_html=True)
with c3: st.markdown(f"<div class='card'><div class='kpi'>{money(forecast_uplift)}</div><div class='kpi-sub'>30-day Forecast Uplift</div></div>", unsafe_allow_html=True)

# ============= Top Moves ======================================================
st.subheader("💡 Top 3 Recovery Moves")
moves=[]
if potential_loss>0: moves.append(f"Plug anomaly days → recover ~{money(potential_loss)} (pricing/promos, billing, ops).")
if upsell_potential>0 and not by_channel.empty:
    worst = by_channel.nsmallest(1, "avg_deal_size")
    if not worst.empty: moves.append(f"Lift {worst['channel'].iloc[0]} avg deal size to 75th pct → unlock ~{money(upsell_potential)}.")
if forecast_uplift>0: moves.append(f"Prep capacity & promos for next 30 days → capture ~{money(forecast_uplift)}.")
if not moves: moves=["🎉 No major gaps detected — focus on targeted retention & upsell."]
for i,m in enumerate(moves,1): st.markdown(f"- **{i}. {m}**")

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# ============= Charts =========================================================
if not df_f.empty:
    a,b = st.columns([2,1])
    with a:
        st.subheader("Revenue Trend & Anomalies")
        base = alt.Chart(daily).encode(x=alt.X("date:T", title="Date"))
        line = base.mark_line().encode(y=alt.Y("revenue:Q", title="Revenue ($)", axis=alt.Axis(format="~s")))
        if not anoms.empty:
            pts = alt.Chart(anoms).mark_point(size=80, filled=True, color="#EF4444").encode(
                x="date:T", y="revenue:Q", tooltip=["date:T", alt.Tooltip("revenue:Q", format=",.0f")])
            st.altair_chart((line+pts).properties(height=340), use_container_width=True)
        else:
            st.altair_chart(line.properties(height=340), use_container_width=True)
    with b:
        st.subheader("Revenue by Channel")
        by_ch = df_f.groupby("channel", as_index=False)["revenue"].sum().sort_values("revenue", ascending=False)
        st.altair_chart(alt.Chart(by_ch).mark_bar().encode(
            x=alt.X("revenue:Q", title="Revenue ($)", axis=alt.Axis(format="~s")),
            y=alt.Y("channel:N", sort="-x"),
            tooltip=["channel:N", alt.Tooltip("revenue:Q", format=",.0f")]).properties(height=340), use_container_width=True)

    st.subheader("Revenue by Region")
    by_rg = df_f.groupby("region", as_index=False)["revenue"].sum().sort_values("revenue", ascending=False)
    st.altair_chart(alt.Chart(by_rg).mark_bar().encode(
        x=alt.X("region:N", title="Region"),
        y=alt.Y("revenue:Q", title="Revenue ($)", axis=alt.Axis(format="~s")),
        tooltip=["region:N", alt.Tooltip("revenue:Q", format=",.0f")]).properties(height=300), use_container_width=True)

    st.subheader("30-Day Revenue Forecast")
    st.altair_chart(alt.Chart(fc).mark_line(point=True).encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y("value:Q", title="Revenue ($)", axis=alt.Axis(format="~s")),
        color="type:N",
        tooltip=["type:N","date:T",alt.Tooltip("value:Q", format=",.0f")]).properties(height=360), use_container_width=True)

# ============= Downloads & Share + PDF =======================================
st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
st.subheader("⬇️ Export & Share")

kpi_df = pd.DataFrame([
    {"metric":"Total Revenue (30d)","value": total_rev_30},
    {"metric":"Avg Daily Revenue (30d)","value": avg_rev_30},
    {"metric":"Anomaly Days","value": anomaly_days},
    {"metric":"Top Channel","value": top_channel_name},
    {"metric":"Recoverable (Anomalies)","value": potential_loss},
    {"metric":"Upsell Potential","value": upsell_potential},
    {"metric":"30-day Forecast Uplift","value": forecast_uplift},
])

colA,colB,colC,colD = st.columns([1,1,1,2])
with colA: st.download_button("Filtered data CSV", df_f.to_csv(index=False), "filtered_data.csv", "text/csv")
with colB: st.download_button("KPI CSV", kpi_df.to_csv(index=False), "kpis.csv", "text/csv")
with colC:
    top_moves_txt = "\n".join(f"{i}. {m}" for i,m in enumerate(moves,1))
    st.download_button("Top moves TXT", top_moves_txt, "top_moves.txt")
with colD:
    cur_qp = read_query_params()
    share_url = "?" + "&".join([f"{k}={','.join(v) if isinstance(v,list) else v}" for k,v in cur_qp.items()])
    st.text_input("Share this exact view (copy URL)", value=share_url)

def build_kpi_pdf_bytes():
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    w, h = A4
    y = h - 1.1*inch
    c.setFont("Helvetica-Bold", 16); c.drawString(1*inch, y, "AI Revenue Recovery – KPI Report"); y -= 0.3*inch
    c.setFont("Helvetica", 10); c.drawString(1*inch, y, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}"); y -= 0.3*inch
    def line(txt): 
        nonlocal y
        c.setFont("Helvetica", 12); c.drawString(1*inch, y, txt); y -= 0.25*inch
    line(f"Total Revenue (30d): {money(total_rev_30)}")
    line(f"Avg Daily Revenue (30d): {money(avg_rev_30)}")
    line(f"Anomaly Days: {anomaly_days}  ({anomaly_pct:.1f}% of days)")
    line(f"Top Channel: {top_channel_name} ({top_channel_share:.1f}% share)")
    line(f"Recoverable (Anomalies): {money(potential_loss)}")
    line(f"Upsell Potential: {money(upsell_potential)}")
    line(f"30-day Forecast Uplift: {money(forecast_uplift)}")
    y -= 0.15*inch
    c.setFont("Helvetica-Bold", 13); c.drawString(1*inch, y, "Top Recovery Moves"); y -= 0.25*inch
    c.setFont("Helvetica", 12)
    for i, m in enumerate(moves, 1):
        for chunk in re.findall('.{1,90}(?:\\s+|$)', m):
            c.drawString(1*inch, y, f"{i}. {chunk.strip() if chunk!=m else m}"); y -= 0.22*inch
    c.showPage(); c.save()
    buf.seek(0)
    return buf.getvalue()

st.download_button("📄 Download KPI PDF", data=build_kpi_pdf_bytes(), file_name="kpi_report.pdf", mime="application/pdf")

# ============= RAG – KB Store (memory + optional temp disk) ===================
if "kb" not in st.session_state:
    st.session_state.kb = {"docs": [], "index": None}  # docs: [{id,name,source,chunks}], index: see build_index

# optional temp persistence (no permission error)
try:
    TMP_DIR = pathlib.Path(tempfile.gettempdir()) / "ai_rev_kb"
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    KB_PICKLE = TMP_DIR / "kb_store.pkl"
    DISK_OK = True
except Exception:
    KB_PICKLE = None
    DISK_OK = False

def save_kb_to_disk():
    if not DISK_OK: return
    try:
        with open(KB_PICKLE, "wb") as f: pickle.dump(st.session_state.kb, f)
    except Exception:
        pass

def load_kb_from_disk():
    if not (DISK_OK and KB_PICKLE.exists()): return
    try:
        with open(KB_PICKLE, "rb") as f: st.session_state.kb = pickle.load(f)
    except Exception:
        pass

load_kb_from_disk()

# ---- text extraction ----
def extract_text(file_bytes: bytes, filename: str) -> str:
    name = filename.lower()
    if name.endswith(".pdf"):
        from PyPDF2 import PdfReader
        reader = PdfReader(io.BytesIO(file_bytes))
        return "\n".join([p.extract_text() or "" for p in reader.pages])
    elif name.endswith(".csv"):
        try:
            df = pd.read_csv(io.BytesIO(file_bytes))
            return df.to_csv(index=False)
        except Exception:
            return file_bytes.decode("utf-8", errors="ignore")
    elif name.endswith(".md") or name.endswith(".txt"):
        return file_bytes.decode("utf-8", errors="ignore")
    else:
        return file_bytes.decode("utf-8", errors="ignore")

def chunk_text(text: str, chunk_size=900, overlap=150):
    text = " ".join(text.split())
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        start = max(0, end - overlap)
        if end == len(text): break
    return chunks

def build_index_from_docs(docs):
    # flatten chunks and track doc mapping
    chunks, mapping = [], []
    for d in docs:
        for ch in d["chunks"]:
            chunks.append(ch)
            mapping.append(d["id"])
    if not chunks:
        return None
    if SBERT_OK:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        emb = model.encode(chunks, normalize_embeddings=True)
        return {"kind":"sbert","chunks":chunks,"emb":emb,"chunk_doc_ids":mapping,"model":"all-MiniLM-L6-v2"}
    else:
        vec = TfidfVectorizer(max_features=20000)
        X = vec.fit_transform(chunks)
        return {"kind":"tfidf","chunks":chunks,"emb":X,"chunk_doc_ids":mapping,"vectorizer":vec}

def add_files_to_kb(uploaded_files):
    added = 0
    for f in uploaded_files:
        text = extract_text(f.read(), f.name)
        chs = chunk_text(text)
        st.session_state.kb["docs"].append({"id": str(uuid.uuid4()), "name": f.name, "source": "file", "chunks": chs})
        added += 1
    st.session_state.kb["index"] = build_index_from_docs(st.session_state.kb["docs"])
    save_kb_to_disk()
    return added

def delete_doc(doc_id):
    st.session_state.kb["docs"] = [d for d in st.session_state.kb["docs"] if d["id"] != doc_id]
    st.session_state.kb["index"] = build_index_from_docs(st.session_state.kb["docs"])
    save_kb_to_disk()

def import_kb_pickle(file):
    try:
        kb = pickle.loads(file.read())
        if isinstance(kb, dict) and "docs" in kb:
            st.session_state.kb = kb
            save_kb_to_disk()
            return True
    except Exception:
        return False
    return False

def export_kb_pickle_bytes():
    return pickle.dumps(st.session_state.kb)

# ---- URL crawler (depth 0/1, same domain only for depth 1) ----
def normalize_url(u): return u.split("#")[0].strip()
def fetch_url_text(url):
    try:
        resp = requests.get(url, timeout=8, headers={"User-Agent":"Mozilla/5.0 (AI-RevBot)"})
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script","style","noscript"]): tag.extract()
        text = soup.get_text(separator=" ")
        text = re.sub(r"\s+", " ", text)
        return text[:500000]
    except Exception:
        return ""

def crawl_urls(seed_urls: list[str], depth: int = 0, max_pages: int = 8):
    seen, out_docs = set(), []
    seeds = [normalize_url(u) for u in seed_urls if u]
    queue = [(u,0) for u in seeds]
    domain = None
    if seeds:
        try: domain = requests.utils.urlparse(seeds[0]).netloc
        except Exception: domain = None
    while queue and len(out_docs) < max_pages:
        url, d = queue.pop(0)
        if url in seen: continue
        seen.add(url)
        txt = fetch_url_text(url)
        if txt:
            out_docs.append({"id": str(uuid.uuid4()), "name": url, "source": "url", "chunks": chunk_text(txt)})
        if depth >= 1 and d < depth:
            try:
                resp = requests.get(url, timeout=8, headers={"User-Agent":"Mozilla/5.0 (AI-RevBot)"})
                soup = BeautifulSoup(resp.text, "html.parser")
                links = [requests.compat.urljoin(url, a.get("href")) for a in soup.find_all("a", href=True)]
                for lk in links:
                    if not lk.startswith("http"): continue
                    if domain and requests.utils.urlparse(lk).netloc != domain: continue
                    lk = normalize_url(lk)
                    if lk not in seen:
                        queue.append((lk, d+1))
                        if len(queue) > max_pages*3: break
            except Exception:
                pass
    # merge
    st.session_state.kb["docs"].extend(out_docs)
    st.session_state.kb["index"] = build_index_from_docs(st.session_state.kb["docs"])
    save_kb_to_disk()
    return len(out_docs)

def kb_search(query: str, k=5):
    idx = st.session_state.kb.get("index")
    if not idx: return []
    chunks = idx["chunks"]; mapping = idx["chunk_doc_ids"]
    if idx["kind"] == "sbert":
        model = SentenceTransformer(idx.get("model","all-MiniLM-L6-v2"))
        q = model.encode([query], normalize_embeddings=True)
        sims = (idx["emb"] @ q.T).ravel()
    else:
        vec: TfidfVectorizer = idx["vectorizer"]
        sims = cosine_similarity(idx["emb"], vec.transform([query])).ravel()
    top = sims.argsort()[::-1][:k]
    out = []
    for i in top:
        doc_id = mapping[i]
        doc_name = next((d["name"] for d in st.session_state.kb["docs"] if d["id"]==doc_id), "KB")
        out.append({"text": chunks[i], "score": float(sims[i]), "doc_id": doc_id, "doc_name": doc_name})
    return out

# ============= KB Manager UI ==================================================
st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
st.subheader("📚 Knowledge Base (RAG)")

tab1, tab2, tab3 = st.tabs(["Upload Files", "Import/Export", "URL Crawler"])
with tab1:
    files = st.file_uploader("Add documents (PDF/TXT/MD/CSV)", type=["pdf","txt","md","csv"], accept_multiple_files=True, key="kb_files")
    if st.button("Add to KB", type="primary"):
        if files:
            n = add_files_to_kb(files)
            st.success(f"Added {n} file(s) to KB and updated index.")
        else:
            st.info("Select one or more files first.")
    st.write("### KB Documents")
    if st.session_state.kb["docs"]:
        for d in st.session_state.kb["docs"]:
            cols = st.columns([6,2,2])
            with cols[0]:
                st.markdown(f"**{d['name']}**  \n<span class='muted'>source: {d['source']}, chunks: {len(d['chunks'])}</span>", unsafe_allow_html=True)
            with cols[1]:
                st.button("Delete", key=f"del_{d['id']}", on_click=lambda i=d["id"]: delete_doc(i))
            with cols[2]:
                st.text("")  # spacer
    else:
        st.caption("No documents yet — add files above.")

with tab2:
    colx, coly = st.columns(2)
    with colx:
        st.markdown("**Export current KB**")
        st.download_button("Download KB (.pkl)", data=export_kb_pickle_bytes(), file_name="kb_store.pkl", mime="application/octet-stream")
    with coly:
        st.markdown("**Import KB**")
        imp = st.file_uploader("Upload .pkl exported earlier", type=["pkl"])
        if st.button("Import KB"):
            if imp and import_kb_pickle(imp):
                st.success("KB imported.")
            else:
                st.error("Invalid KB file.")

with tab3:
    urls_text = st.text_area("Enter one or more URLs (newline separated)", placeholder="https://yourdomain.com/help\nhttps://docs.yourdomain.com/faq")
    depth = st.selectbox("Crawl depth", [0,1], index=0, help="Depth 0 = just these URLs; Depth 1 = follow internal links from first URL.")
    max_pages = st.slider("Max pages", 1, 30, 8)
    if st.button("Crawl & Index"):
        seeds = [u.strip() for u in urls_text.splitlines() if u.strip()]
        if not seeds:
            st.info("Add at least one URL.")
        else:
            added = crawl_urls(seeds, depth=int(depth), max_pages=int(max_pages))
            st.success(f"Crawled and added {added} page(s) to KB.")

# ============= Assistant (RAG-aware) ==========================================
st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
st.subheader("🤖 Assistant (RAG-aware) — ask how to recover more revenue")

if "chat" not in st.session_state:
    st.session_state.chat = [{"role":"assistant","content":"Hi! I use your live KPIs and your Knowledge Base to give grounded answers. Ask me about anomalies, upsell, forecasting, or anything in your docs."}]

context_numbers = {
    "total_rev_30": money(total_rev_30), "avg_rev_30": money(avg_rev_30),
    "anomaly_days": anomaly_days, "recoverable": money(potential_loss),
    "upsell": money(upsell_potential), "uplift": money(forecast_uplift),
    "top_channel": top_channel_name, "top_channel_share": f"{top_channel_share:.1f}%"
}

def local_reply(prompt: str) -> str:
    hits = kb_search(prompt, k=5)
    kb_text = "\n".join([f"[{i+1}] {h['doc_name']}: {h['text'][:280]}…" for i,h in enumerate(hits)]) if hits else "—"
    p = prompt.lower()
    if "anomal" in p or "recover" in p:
        base = (f"You have **{context_numbers['anomaly_days']} anomaly days**. "
                f"Estimated recoverable revenue **{context_numbers['recoverable']}**. "
                f"Audit pricing/promotions & billing on those dates; implement prevention checks.")
    elif "upsell" in p or "deal size" in p:
        base = (f"Target 75th-pct deal size. Estimated **upsell potential {context_numbers['upsell']}**. "
                f"Start with weaker channels using bundles, add-ons, and annual incentives.")
    elif "forecast" in p or "next 30" in p:
        base = (f"Projected 30-day uplift **{context_numbers['uplift']}** vs baseline. "
                f"Align inventory, promo cadence, and capacity with weekly peaks.")
    else:
        base = (f"Summary — 30d revenue **{context_numbers['total_rev_30']}**, avg/day **{context_numbers['avg_rev_30']}**, "
                f"recoverable **{context_numbers['recoverable']}**, upsell **{context_numbers['upsell']}**, uplift **{context_numbers['uplift']}**.")
    if kb_text and kb_text != "—":
        base += "\n\n**Context from your Knowledge Base:**\n" + kb_text
    return base

def openai_reply(prompt: str) -> str:
    api_key = os.environ.get("OPENAI_API_KEY","")
    if not api_key: return local_reply(prompt)
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        hits = kb_search(prompt, k=6)
        kb_full = "\n\n".join([f"[{i+1}] {h['doc_name']}: {h['text']}" for i,h in enumerate(hits)])
        sys = ("You are a revenue recovery copilot. Ground answers in the provided KPIs and 'Context Docs'. "
               "Cite with bracket numbers like [1], [2]. Keep answers practical and concise.")
        kpi_blob = json.dumps(context_numbers)
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role":"system","content":sys},
                {"role":"user","content":f"KPIs: {kpi_blob}\n\nContext Docs:\n{kb_full}\n\nQuestion: {prompt}"}
            ],
            temperature=0.2,
        )
        return r.choices[0].message.content.strip()
    except Exception:
        return local_reply(prompt)

for m in st.session_state.chat:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
user_msg = st.chat_input("Ask about anomalies, upsell, forecast, or anything in your docs…")
if user_msg:
    st.session_state.chat.append({"role":"user","content":user_msg})
    with st.chat_message("user"): st.markdown(user_msg)
    reply = openai_reply(user_msg)
    st.session_state.chat.append({"role":"assistant","content":reply})
    with st.chat_message("assistant"): st.markdown(reply)

# ============= Footer =========================================================
st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
st.markdown("<div class='cta'><b>Want a private pilot?</b> — Upload your latest CSV and documents; we’ll surface your top recovery moves in minutes.</div>", unsafe_allow_html=True)
