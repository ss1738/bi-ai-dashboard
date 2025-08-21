# AI Revenue Recovery – Polished + RAG Assistant
# Run: streamlit run streamlit_app.py

import os, uuid, json, pickle, io
import requests
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Optional SBERT (used if available)
SBERT_OK = False
try:
    from sentence_transformers import SentenceTransformer
    SBERT_OK = True
except Exception:
    SBERT_OK = False

st.set_page_config(page_title="AI Revenue Recovery", page_icon="💰", layout="wide")

# ---------- Styling ----------
st.markdown("""
<style>
:root { --primary:#6366F1; --secondary:#EC4899; --success:#10B981; --dark:#111827; }
.hero {padding:1.25rem 1.5rem;border-radius:16px;background:linear-gradient(135deg,var(--primary),var(--secondary));
       color:white; box-shadow:0 10px 30px rgba(0,0,0,.12);}
.card {background:white;border-radius:14px;padding:1rem;box-shadow:0 4px 16px rgba(0,0,0,.07);}
.kpi{font-size:1.8rem;font-weight:800;margin:0;} .kpi-sub{opacity:.8;margin-top:.25rem;}
.cta{padding:1rem;border-radius:12px;background:var(--dark);color:#fff;} .divider{height:1px;background:#e5e7eb;margin:1rem 0;}
.chatbox{border-radius:16px;background:white;box-shadow:0 6px 24px rgba(0,0,0,.08);border:1px solid #eee;}
</style>
""", unsafe_allow_html=True)

try: alt.data_transformers.disable_max_rows()
except Exception: pass

# ---------- (Optional) GA4 Measurement Protocol ----------
GA_MEASUREMENT_ID = os.environ.get("GA_MEASUREMENT_ID", "")
GA_API_SECRET      = os.environ.get("GA_API_SECRET", "")

def ga_event(name: str, **params):
    if not (GA_MEASUREMENT_ID and GA_API_SECRET): return
    cid = st.session_state.get("client_id") or str(uuid.uuid4())
    st.session_state.client_id = cid
    try:
        requests.post(
            f"https://www.google-analytics.com/mp/collect?measurement_id={GA_MEASUREMENT_ID}&api_secret={GA_API_SECRET}",
            json={"client_id": cid, "events": [{"name": name, "params": params}]},
            timeout=3
        )
    except Exception:
        pass

# ---------- Query params helpers ----------
def read_query_params():
    try: return st.query_params
    except Exception: return st.experimental_get_query_params()

def set_query_params(**kwargs):
    try:
        qp = st.query_params
        for k, v in kwargs.items():
            if v is None:
                if k in qp: del qp[k]
            else: qp[k] = v
    except Exception:
        st.experimental_set_query_params(**{k: v for k, v in kwargs.items() if v is not None})

# ---------- Sample Data ----------
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
                rows.append({"date": d,"region": rg,"channel": ch,"segment": segment,"product": product,
                             "revenue": float(revenue),"customers": customers})
    df = pd.DataFrame(rows)
    # Inject anomaly days
    anom_days = np.random.choice(df["date"].dt.normalize().unique(), size=8, replace=False)
    for ad in anom_days:
        m = df["date"].dt.normalize().eq(ad)
        df.loc[m, "revenue"]  *= np.random.uniform(0.18, 0.35)
        df.loc[m, "customers"] = np.maximum(1, (df.loc[m, "customers"] * np.random.uniform(0.3,0.6)).round()).astype(int)
    return df

# ---------- CSV Loader ----------
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
    if "customers" in cols:
        df["customers"] = coerce_numeric(df[cols["customers"]]).fillna(1).astype(int)
    else:
        df["customers"] = np.maximum(1, (df["revenue"] / np.maximum(1.0, df["revenue"].median()/5)).round()).astype(int)
    df = df.dropna(subset=["date","revenue"]).copy()
    df["revenue"] = df["revenue"].clip(lower=0.0)
    return df[["date","region","channel","segment","product","revenue","customers"]]

# ---------- ML ----------
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

# ---------- Hero ----------
st.markdown(
    '<div class="hero"><h2 style="margin:0;">Recover $500K+ in Lost Revenue</h2>'
    '<p style="margin:.2rem 0 0 0;">Upload your CSV and instantly see anomalies to recover, upsell potential, and a 30-day forecast.</p></div>',
    unsafe_allow_html=True,
)

# ---------- Upload / Sample ----------
with st.expander("Upload your CSV (or use sample data)"):
    up = st.file_uploader("CSV with columns like: date, region, channel, revenue, customers…", type=["csv"])
    st.download_button("⬇️ Download sample CSV", make_sample_data().to_csv(index=False),
                       file_name="sample_revenue_data.csv", mime="text/csv")
    if st.button("Use sample data"):
        st.session_state["_use_sample"] = True

with st.spinner("Analyzing your data..."):
    if up is not None and not st.session_state.get("_use_sample"):
        try:
            df = load_csv(up); st.success("Data loaded from your CSV ✅"); ga_event("csv_loaded", rows=int(len(df)))
        except Exception as e:
            st.error(f"CSV error: {e}"); df = make_sample_data(); st.info("Using sample data instead."); ga_event("csv_load_failed")
    else:
        df = make_sample_data(); st.info("Using sample data. Upload your CSV to analyze your own revenue."); ga_event("using_sample_data")

# ---------- Filters (URL-sync) ----------
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

# ---------- KPIs ----------
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
delta_avg_rev = pct_delta(avg_rev_30, avg_rev_prev)
anomaly_days = int(len(anoms))
anomaly_pct = (anomaly_days / max(1, len(daily))) * 100.0
top_ch = by_channel.sort_values("revenue", ascending=False).head(1)
top_channel_name = (top_ch["channel"].iloc[0] if not top_ch.empty else "—")
top_channel_share = float(top_ch["revenue"].iloc[0] / max(1.0, by_channel["revenue"].sum()) * 100.0) if not top_ch.empty else 0.0

k1,k2,k3,k4 = st.columns(4)
with k1: st.metric("Total Revenue (30d)", money(total_rev_30), f"{delta_total_rev:+.1f}% vs prev 30d" if total_rev_prev is not None else None)
with k2: st.metric("Avg Daily Revenue (30d)", money(avg_rev_30), f"{delta_avg_rev:+.1f}% vs prev 30d" if avg_rev_prev is not None else None)
with k3: st.metric("Anomaly Days", f"{anomaly_days} days", f"{anomaly_pct:.1f}% of days")
with k4: st.metric("Top Channel", top_channel_name, f"{top_channel_share:.1f}% share")

c1,c2,c3 = st.columns(3)
with c1: st.markdown(f"<div class='card'><div class='kpi'>{money(potential_loss)}</div><div class='kpi-sub'>Recoverable (Anomalies)</div></div>", unsafe_allow_html=True)
with c2: st.markdown(f"<div class='card'><div class='kpi'>{money(upsell_potential)}</div><div class='kpi-sub'>Upsell Potential</div></div>", unsafe_allow_html=True)
with c3: st.markdown(f"<div class='card'><div class='kpi'>{money(forecast_uplift)}</div><div class='kpi-sub'>30-day Forecast Uplift</div></div>", unsafe_allow_html=True)

# ---------- Moves ----------
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

# ---------- Charts ----------
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
            x=alt.X("revenue:Q", title="Revenue ($)", axis=alt.Axis(format="~s")), y=alt.Y("channel:N", sort="-x"),
            tooltip=["channel:N", alt.Tooltip("revenue:Q", format=",.0f")]).properties(height=340), use_container_width=True)

    st.subheader("Revenue by Region")
    by_rg = df_f.groupby("region", as_index=False)["revenue"].sum().sort_values("revenue", ascending=False)
    st.altair_chart(alt.Chart(by_rg).mark_bar().encode(
        x=alt.X("region:N", title="Region"), y=alt.Y("revenue:Q", title="Revenue ($)", axis=alt.Axis(format="~s")),
        tooltip=["region:N", alt.Tooltip("revenue:Q", format=",.0f")]).properties(height=300), use_container_width=True)

    st.subheader("30-Day Revenue Forecast")
    st.altair_chart(alt.Chart(fc).mark_line(point=True).encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y("value:Q", title="Revenue ($)", axis=alt.Axis(format="~s")),
        color="type:N",
        tooltip=["type:N","date:T",alt.Tooltip("value:Q", format=",.0f")]).properties(height=360), use_container_width=True)

# ---------- Downloads & Share ----------
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

# =============================================================================
# RAG: Knowledge Base (PDF/TXT/MD/CSV)  ———  Simple local vector store
# =============================================================================
KB_DIR = os.path.join("/mnt/data", "kb_store")
os.makedirs(KB_DIR, exist_ok=True)
INDEX_PATH = os.path.join(KB_DIR, "kb_index.pkl")

def extract_text(file_bytes: bytes, filename: str) -> str:
    name = filename.lower()
    if name.endswith(".pdf"):
        from PyPDF2 import PdfReader
        reader = PdfReader(io.BytesIO(file_bytes))
        return "\n".join([p.extract_text() or "" for p in reader.pages])
    elif name.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(file_bytes))
        return df.to_csv(index=False)
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
        start = end - overlap
        if start < 0: start = 0
        if end == len(text): break
    return chunks

def build_index(docs: list[str]):
    # returns dict with 'emb', 'chunks', and encoder info
    if SBERT_OK:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        chunks = []
        for doc in docs: chunks.extend(chunk_text(doc))
        emb = model.encode(chunks, normalize_embeddings=True)
        return {"kind":"sbert","model":"all-MiniLM-L6-v2","chunks":chunks,"emb":emb}
    else:
        chunks = []
        for doc in docs: chunks.extend(chunk_text(doc))
        vec = TfidfVectorizer(max_features=20000)
        X = vec.fit_transform(chunks)
        return {"kind":"tfidf","vectorizer":vec,"chunks":chunks,"emb":X}

def load_index():
    if os.path.exists(INDEX_PATH):
        with open(INDEX_PATH, "rb") as f: return pickle.load(f)
    return None

def save_index(idx):
    with open(INDEX_PATH, "wb") as f: pickle.dump(idx, f)

def add_to_kb(files):
    docs = []
    for f in files:
        txt = extract_text(f.read(), f.name)
        docs.append(txt)
    idx = load_index()
    new_idx = build_index(docs)
    if idx:  # merge
        idx["chunks"].extend(new_idx["chunks"])
        if idx["kind"] == "sbert" and new_idx["kind"] == "sbert":
            idx["emb"] = np.vstack([idx["emb"], new_idx["emb"]])
        else:
            # rebuild TF-IDF on merged chunks for simplicity
            vec = TfidfVectorizer(max_features=20000)
            X = vec.fit_transform(idx["chunks"])
            idx = {"kind":"tfidf","vectorizer":vec,"chunks":idx["chunks"],"emb":X}
    else:
        idx = new_idx
    save_index(idx)
    return len(idx["chunks"])

def kb_search(query: str, k=5):
    idx = load_index()
    if not idx: return []
    chunks = idx["chunks"]
    if idx["kind"] == "sbert":
        model = SentenceTransformer(idx.get("model","all-MiniLM-L6-v2"))
        q = model.encode([query], normalize_embeddings=True)
        sims = (idx["emb"] @ q.T).ravel()
        top = sims.argsort()[::-1][:k]
        return [(chunks[i], float(sims[i])) for i in top]
    else:
        vec: TfidfVectorizer = idx["vectorizer"]
        Xq = vec.transform([query])
        sims = cosine_similarity(idx["emb"], Xq).ravel()
        top = sims.argsort()[::-1][:k]
        return [(chunks[i], float(sims[i])) for i in top]

with st.expander("📚 Knowledge Base (RAG) — upload PDFs/TXT/MD/CSV and ground the assistant", expanded=False):
    kb_files = st.file_uploader("Add documents", type=["pdf","txt","md","csv"], accept_multiple_files=True)
    colx, coly = st.columns([1,2])
    with colx:
        if st.button("Build / Update Index"):
            if kb_files:
                n = add_to_kb(kb_files)
                st.success(f"Index updated. Chunks in KB: {n}")
            else:
                st.info("Upload one or more files first.")
    with coly:
        if st.button("Clear KB"):
            if os.path.exists(INDEX_PATH): os.remove(INDEX_PATH); st.success("Knowledge Base cleared.")
            else: st.info("KB already empty.")
    if os.path.exists(INDEX_PATH):
        sz = os.path.getsize(INDEX_PATH)/1024
        st.caption(f"KB ready • index size ~{sz:.1f} KB")

# ---------- Assistant (RAG-aware) ----------
st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
with st.expander("🤖 Assistant (RAG-aware) — ask how to recover more revenue", expanded=False):
    if "chat" not in st.session_state:
        st.session_state.chat = [{"role":"assistant","content":"Hi! I’m grounded in your KPIs and any documents you add to the Knowledge Base."}]
    context_numbers = {
        "total_rev_30": money(total_rev_30), "avg_rev_30": money(avg_rev_30),
        "anomaly_days": anomaly_days, "recoverable": money(potential_loss),
        "upsell": money(upsell_potential), "uplift": money(forecast_uplift),
        "top_channel": top_channel_name, "top_channel_share": f"{top_channel_share:.1f}%"
    }

    def local_reply(prompt: str) -> str:
        # retrieve KB
        hits = kb_search(prompt, k=4)
        kb_snips = "\n\n".join([f"[{i+1}] {t[0][:500]}" for i,t in enumerate(hits)]) if hits else ""
        # simple intent rules on top of KPIs
        p = prompt.lower()
        if "anomal" in p or "recover" in p:
            base = (f"You have **{context_numbers['anomaly_days']} anomaly days**. "
                    f"Estimated recoverable revenue **{context_numbers['recoverable']}**. "
                    f"Audit pricing/promotions and billing on those dates, then codify a prevention playbook.")
        elif "upsell" in p or "deal size" in p:
            base = (f"Target the 75th percentile deal size. Estimated **upsell potential {context_numbers['upsell']}**. "
                    f"Use bundles/add-ons on weaker channels first.")
        elif "forecast" in p or "next 30" in p:
            base = (f"Projected 30-day uplift **{context_numbers['uplift']}** vs baseline. "
                    f"Plan inventory and promos to capture weekly peaks.")
        else:
            base = (f"Summary: 30-day revenue **{context_numbers['total_rev_30']}**, "
                    f"avg/day **{context_numbers['avg_rev_30']}**, recoverable **{context_numbers['recoverable']}**, "
                    f"upsell **{context_numbers['upsell']}**, uplift **{context_numbers['uplift']}**.")
        if kb_snips:
            base += "\n\n**Context from your Knowledge Base:**\n" + kb_snips
        return base

    def openai_reply(prompt: str) -> str:
        api_key = os.environ.get("OPENAI_API_KEY","")
        if not api_key: return local_reply(prompt)
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            hits = kb_search(prompt, k=6)
            kb_text = "\n\n".join([f"[{i+1}] {t[0]}" for i,t in enumerate(hits)]) if hits else ""
            sys = ("You are a revenue recovery copilot. Ground every answer in the provided KPIs and the 'Context Docs'. "
                   "Cite snippets by bracket numbers like [1], [2]. Be concise and actionable.")
            kpi_blob = json.dumps(context_numbers)
            user_prompt = f"KPIs: {kpi_blob}\n\nContext Docs:\n{kb_text}\n\nQuestion: {prompt}"
            r = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role":"system","content":sys},{"role":"user","content":user_prompt}],
                temperature=0.2,
            )
            return r.choices[0].message.content.strip()
        except Exception:
            return local_reply(prompt)

    for m in st.session_state.chat:
        with st.chat_message(m["role"]): st.markdown(m["content"])
    user_msg = st.chat_input("Ask about anomalies, upsell, forecast, or anything in your docs…")
    if user_msg:
        st.session_state.chat.append({"role":"user","content":user_msg})
        with st.chat_message("user"): st.markdown(user_msg)
        answer = openai_reply(user_msg)
        st.session_state.chat.append({"role":"assistant","content":answer})
        with st.chat_message("assistant"): st.markdown(answer)

# ---------- Footer ----------
st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
st.markdown("<div class='cta'><b>Want a private pilot?</b> — Upload your latest CSV and documents; we’ll surface your top recovery moves in minutes.</div>", unsafe_allow_html=True)
