# AI Revenue Recovery – Polished UI Edition (keeps all features)
# Paste into streamlit_app.py

import os, io, re, json, uuid, sqlite3
from datetime import datetime, timedelta
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.ensemble import IsolationForest, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Optional libs with graceful fallbacks
PROPHET_OK = XGB_OK = LGB_OK = TORCH_OK = TF_OK = BS4_OK = PDF_READER_OK = True
try:
    from prophet import Prophet
except Exception:
    PROPHET_OK = False
try:
    import xgboost as xgb
except Exception:
    XGB_OK = False
try:
    import lightgbm as lgb
except Exception:
    LGB_OK = False
try:
    import torch, torch.nn as nn, torch.optim as optim
except Exception:
    TORCH_OK = False
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
except Exception:
    TF_OK = False
try:
    import requests
    from bs4 import BeautifulSoup
except Exception:
    BS4_OK = False
try:
    from PyPDF2 import PdfReader
except Exception:
    PDF_READER_OK = False
try:
    from fpdf import FPDF
except Exception:
    pass

try:
    alt.data_transformers.disable_max_rows()
except Exception:
    pass

# ---------------------------------------------------------
# Page + Theming
# ---------------------------------------------------------
st.set_page_config(page_title="AI Revenue Recovery", page_icon="💰", layout="wide")

if "theme" not in st.session_state:
    st.session_state.theme = "dark"  # default

DARK = st.session_state.theme == "dark"

PALETTE = {
    "primary": "#7C3AED" if DARK else "#6366F1",
    "secondary": "#EC4899",
    "bg": "#0B1020" if DARK else "#F6F7FB",
    "card": "rgba(255,255,255,0.06)" if DARK else "#FFFFFF",
    "text": "#E5E7EB" if DARK else "#0F172A",
    "muted": "#9CA3AF" if DARK else "#6B7280",
    "accent": "#22D3EE",
    "success": "#10B981",
    "danger": "#EF4444",
}

st.markdown(
    f"""
<style>
:root {{
  --bg: {PALETTE['bg']};
  --text: {PALETTE['text']};
  --muted: {PALETTE['muted']};
  --card: {PALETTE['card']};
  --primary: {PALETTE['primary']};
  --secondary: {PALETTE['secondary']};
  --accent: {PALETTE['accent']};
  --success: {PALETTE['success']};
  --danger: {PALETTE['danger']};
}}
html, body, .stApp {{
  background: var(--bg) !important;
  color: var(--text) !important;
}}
/* Hide default chrome */
#MainMenu, footer, .stDeployButton {{ display: none !important; }}
/* Top bar */
.navbar {{
  position: sticky; top: 0; z-index: 1000;
  backdrop-filter: saturate(180%) blur(8px);
  background: rgba(0,0,0,{0.35 if DARK else 0.05});
  border-bottom: 1px solid rgba(255,255,255,0.08);
  padding: .6rem 1rem; margin: -1rem -1rem 1rem -1rem;
}}
.brand {{ display:flex; align-items:center; gap:.6rem; font-weight:800; }}
.badge {{
  padding:.15rem .5rem;border-radius:999px;
  background: linear-gradient(135deg, var(--primary), var(--secondary));
  color:#fff;font-size:.78rem; font-weight:700;
}}
.theme-toggle button {{ border:1px solid rgba(255,255,255,.15); }}
/* Hero */
.hero {{
  padding: 1.1rem 1.3rem; border-radius: 16px;
  background: linear-gradient(135deg, var(--primary), var(--secondary));
  color:#fff; box-shadow: 0 12px 40px rgba(0,0,0,.25);
}}
/* Cards */
.card {{
  background: var(--card);
  border-radius: 16px;
  padding: 1rem 1.1rem;
  border: 1px solid rgba(255,255,255,{0.06 if DARK else 0.08});
  box-shadow: { '0 10px 30px rgba(0,0,0,.35)' if DARK else '0 6px 18px rgba(0,0,0,.07)' };
}}
.kpi-title {{ color: var(--muted); font-weight:600; font-size:.86rem; }}
.kpi-value {{ font-size: 1.6rem; font-weight: 800; margin:.1rem 0; }}
.kpi-delta {{
  font-size:.78rem; font-weight:700; color:#fff;
  display:inline-block; padding:.12rem .5rem; border-radius:999px;
}}
.kpi-delta.pos {{ background: rgba(16,185,129,.9); }}
.kpi-delta.neg {{ background: rgba(239,68,68,.9); }}
.section-title {{ font-size:1.05rem; font-weight:800; margin:.2rem 0 .6rem; }}
.muted {{ color: var(--muted); }}
.cta {{ background: #0F172A; color:#fff; padding: .9rem 1rem; border-radius: 14px; }}
hr.div {{ border:none; height:1px; background:linear-gradient(90deg,transparent,rgba(255,255,255,.18),transparent); margin:1rem 0; }}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Utils
# ---------------------------------------------------------
DB_PATH = "sales.db"

def money(x):
    try:
        return f"${float(x):,.0f}"
    except Exception:
        return "$0"

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

# ---------------------------------------------------------
# DB
# ---------------------------------------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
      CREATE TABLE IF NOT EXISTS sales_data (
        date TEXT, region TEXT, channel TEXT, segment TEXT, product TEXT,
        revenue REAL, customers INTEGER
      )
    """)
    conn.close()

def save_to_db(df):
    conn = sqlite3.connect(DB_PATH)
    df.to_sql("sales_data", conn, if_exists="replace", index=False)
    conn.close()

def load_from_db():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM sales_data", conn)
    conn.close()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()
    return df

# ---------------------------------------------------------
# CSV Loader
# ---------------------------------------------------------
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
    df["region"]  = df[cols["region"]].astype(str)  if "region"  in cols else "Unknown"
    df["channel"] = df[cols["channel"]].astype(str) if "channel" in cols else "Unknown"
    df["segment"] = df[cols["segment"]].astype(str) if "segment" in cols else "All"
    df["product"] = df[cols["product"]].astype(str) if "product" in cols else "All"
    if "revenue" in cols:
        df["revenue"] = coerce_numeric(df[cols["revenue"]])
    elif "price" in cols and "quantity" in cols:
        df["revenue"] = coerce_numeric(df[cols["price"]]) * coerce_numeric(df[cols["quantity"]])
    else:
        raise ValueError("CSV must include 'revenue' or ('price' and 'quantity').")
    if "customers" in cols:
        df["customers"] = coerce_numeric(df[cols["customers"]]).fillna(1).astype(int)
    else:
        df["customers"] = np.maximum(1, (df["revenue"] / np.maximum(1.0, df["revenue"].median()/5)).round()).astype(int)
    df = df.dropna(subset=["date","revenue"]).copy()
    df["revenue"] = df["revenue"].clip(lower=0.0)
    return df[["date","region","channel","segment","product","revenue","customers"]]

# ---------------------------------------------------------
# ML – Anomalies + Forecast (Prophet fallback)
# ---------------------------------------------------------
def detect_anomalies(daily_df: pd.DataFrame) -> pd.DataFrame:
    dd = daily_df.sort_values("date").copy()
    if dd.empty or dd["revenue"].isna().all():
        return dd.head(0).copy()
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

def forecast_prophet(daily: pd.DataFrame, days=30) -> pd.DataFrame:
    if not PROPHET_OK or daily.empty: return pd.DataFrame()
    df = daily.rename(columns={"date":"ds","revenue":"y"})
    try:
        m = Prophet()
        m.fit(df)
        future = m.make_future_dataframe(periods=days)
        fc = m.predict(future)
        out = fc[["ds","yhat","yhat_lower","yhat_upper"]].copy()
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
    if df["day_num"].nunique() >= 2:
        X = df["day_num"].values; y = df["revenue"].values
        A = np.vstack([X, np.ones_like(X)]).T
        coef, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
        fut = np.arange(df["day_num"].max()+1, df["day_num"].max()+days+1)
        trend = coef * fut + intercept
    else:
        trend = np.repeat(df["revenue"].iloc[-1], days)
    dow_mean = df.groupby(df["date"].dt.weekday)["revenue"].mean()
    future_dates = pd.date_range(df["date"].max()+timedelta(days=1), periods=days)
    seas = np.array([dow_mean.get(d.weekday(), df["revenue"].mean()) for d in future_dates])
    pred = np.maximum(0.0, (trend + seas)/2.0)
    hist = df[["date","revenue"]].rename(columns={"revenue":"value"}); hist["type"]="Historical"
    fc = pd.DataFrame({"date":future_dates,"value":pred,"type":"Forecast"})
    return pd.concat([hist, fc], ignore_index=True)

def train_churn_model(df):
    base = df.copy()
    base["avg_rev_per_cust"] = base["revenue"] / base["customers"].replace(0,1)
    base["churn_flag"] = (base["revenue"] < base["revenue"].median()).astype(int)
    X = base[["revenue","customers","avg_rev_per_cust"]]; y = base["churn_flag"]
    if len(base) < 10 or y.nunique() < 2: return None, None
    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,random_state=42)
    if XGB_OK:
        try:
            model = xgb.XGBClassifier(eval_metric="logloss", use_label_encoder=False).fit(Xtr,ytr)
        except Exception:
            model = GradientBoostingClassifier().fit(Xtr,ytr)
    else:
        model = GradientBoostingClassifier().fit(Xtr,ytr)
    acc = accuracy_score(yte, model.predict(Xte))
    return model, acc

def train_upsell_model(df):
    base = df.copy()
    base["avg_rev_per_cust"] = base["revenue"] / base["customers"].replace(0,1)
    base["upsell_flag"] = (base["avg_rev_per_cust"] > base["avg_rev_per_cust"].median()).astype(int)
    X = base[["revenue","customers","avg_rev_per_cust"]]; y = base["upsell_flag"]
    if len(base) < 10 or y.nunique() < 2: return None, None
    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,random_state=42)
    if LGB_OK:
        try:
            model = lgb.LGBMClassifier().fit(Xtr,ytr)
        except Exception:
            model = GradientBoostingClassifier().fit(Xtr,ytr)
    else:
        model = GradientBoostingClassifier().fit(Xtr,ytr)
    acc = accuracy_score(yte, model.predict(Xte))
    return model, acc

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=2, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out,_ = self.lstm(x)
        return self.fc(out[:, -1, :])

def train_lstm_pytorch(series, epochs=3):
    if not TORCH_OK or len(series) < 5: return None, None
    data = torch.tensor(series.values, dtype=torch.float32).view(-1,1,1)
    X, y = data[:-1], data[1:]
    model = LSTMModel()
    criterion = nn.MSELoss(); opt = optim.Adam(model.parameters(), lr=0.01)
    for _ in range(epochs):
        opt.zero_grad(); out = model(X); loss = criterion(out, y); loss.backward(); opt.step()
    return model, float(loss.item())

def train_tf_dense(df):
    if not TF_OK or len(df) < 10: return None, None
    X = df[["revenue","customers"]].values
    y = (df["revenue"] > df["revenue"].median()).astype(int).values
    model = keras.Sequential([
        layers.Dense(32, activation="relu", input_shape=(X.shape[1],)),
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(X, y, epochs=5, verbose=0)
    _, acc = model.evaluate(X, y, verbose=0)
    return model, float(acc)

def export_kpi_pdf(kpis, insights):
    pdf = FPDF(); pdf.add_page()
    pdf.set_font("Arial", "B", 16); pdf.cell(200,10,"AI Revenue Recovery Report",ln=True,align="C")
    pdf.set_font("Arial","",12); pdf.ln(6); pdf.cell(200,8,"KPIs",ln=True)
    for k,v in kpis.items(): pdf.cell(200,8,f"{k}: {v}",ln=True)
    pdf.ln(6); pdf.cell(200,8,"Insights",ln=True)
    for ins in insights: pdf.multi_cell(0,8,f"- {ins}")
    return pdf.output(dest="S").encode("latin-1")

# ---------------------------------------------------------
# NAV / THEME
# ---------------------------------------------------------
st.markdown(
    f"""
<div class="navbar">
  <div style="display:flex;justify-content:space-between;align-items:center;">
    <div class="brand">
      <span style="font-size:1.35rem;">💰 AI Revenue Recovery</span>
      <span class="badge">v1.0</span>
    </div>
    <div class="theme-toggle">
      <form action="#" method="post">
      </form>
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)
tcol1, tcol2, tcol3 = st.columns([0.7,0.15,0.15])
with tcol2:
    if st.button(("🌙 Dark" if not DARK else "🌞 Light"), use_container_width=True):
        st.session_state.theme = "dark" if not DARK else "light"
        st.rerun()
with tcol3:
    st.markdown(f"<div class='card' style='text-align:center;font-weight:700;'>Theme: {'Dark' if DARK else 'Light'}</div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Hero
# ---------------------------------------------------------
st.markdown(
    "<div class='hero'><h3 style='margin:0;'>Recover $500K+ in Lost Revenue with AI</h3>"
    "<p style='margin:.3rem 0 0 0;'>Upload your sales CSV • Filter by region/channel/segment/product/date • See KPIs, anomalies, forecast, ML/DL, export PDF, and search your KB.</p></div>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Data load / upload
# ---------------------------------------------------------
init_db()
with st.expander("📂 Upload CSV (or keep previous data)", expanded=False):
    up = st.file_uploader("CSV columns: date, region, channel, segment, product, revenue, customers", type=["csv"])
    c1, c2 = st.columns(2)
    with c1:
        if up is not None:
            try:
                df_u = load_csv(up)
                save_to_db(df_u)
                st.success("✅ Data uploaded & saved")
            except Exception as e:
                st.error(f"CSV error: {e}")
    with c2:
        if st.button("Clear saved data"):
            if os.path.exists(DB_PATH): os.remove(DB_PATH)
            st.success("Cleared SQLite DB. Upload again to proceed.")
            st.stop()

if os.path.exists(DB_PATH):
    df = load_from_db()
else:
    st.warning("No dataset found. Please upload a CSV.")
    st.stop()

# ---------------------------------------------------------
# Filters (+ URL sync) in sidebar
# ---------------------------------------------------------
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"]).copy()
if df.empty:
    st.warning("No valid dates after cleaning."); st.stop()

min_ts, max_ts = pd.to_datetime(df["date"].min()), pd.to_datetime(df["date"].max())
default_end = max_ts.date()
default_start = max((min_ts + pd.Timedelta(days=0)).date(), (max_ts - pd.Timedelta(days=60)).date())

qp = read_query_params()
def parse_csv_list(s):
    if not s: return []
    if isinstance(s, list): s = s[0] if s else ""
    return [x.strip() for x in str(s).split(",") if x.strip()]

all_regions  = sorted(df["region"].dropna().unique().tolist())
all_channels = sorted(df["channel"].dropna().unique().tolist())
all_segments = sorted(df.get("segment", pd.Series(["All"])).dropna().unique().tolist())
all_products = sorted(df.get("product", pd.Series(["All"])).dropna().unique().tolist())

sel_regions  = parse_csv_list(qp.get("region",""))  or all_regions
sel_channels = parse_csv_list(qp.get("channel","")) or all_channels
sel_segments = parse_csv_list(qp.get("segment","")) or all_segments
sel_products = parse_csv_list(qp.get("product","")) or all_products
def _to_date(s, default):
    try: return pd.to_datetime(s).date()
    except Exception: return default
start_default = _to_date(qp.get("start",""), default_start)
end_default   = _to_date(qp.get("end",""),   default_end)

with st.sidebar:
    st.header("Filters")
    with st.expander("Dimensions", True):
        regions_sel  = st.multiselect("🌍 Region",  all_regions,  default=sel_regions)
        channels_sel = st.multiselect("📊 Channel", all_channels, default=sel_channels)
        segments_sel = st.multiselect("👥 Segment", all_segments, default=sel_segments)
        products_sel = st.multiselect("📦 Product", all_products, default=sel_products)
    with st.expander("Date Range", True):
        date_range = st.date_input("📅", value=(start_default, end_default),
                                   min_value=min_ts.date(), max_value=max_ts.date())
    with st.expander("Actions"):
        st.caption("Download and sharing options at the bottom of the page.")

if isinstance(date_range, (list, tuple)) and len(date_range)==2:
    start_date, end_date = map(pd.to_datetime, date_range)
else:
    start_date = pd.to_datetime(date_range); end_date = pd.to_datetime(date_range)

set_query_params(region=",".join(regions_sel), channel=",".join(channels_sel),
                 segment=",".join(segments_sel), product=",".join(products_sel),
                 start=start_date.date().isoformat(), end=end_date.date().isoformat())

mask = (
    df["region"].isin(regions_sel)
    & df["channel"].isin(channels_sel)
    & (df.get("segment","All").isin(segments_sel) if "segment" in df.columns else True)
    & (df.get("product","All").isin(products_sel) if "product" in df.columns else True)
    & df["date"].between(start_date, end_date)
)
filtered = df.loc[mask].copy()
if filtered.empty:
    st.warning("No rows match these filters. Showing all data for context.")
    filtered = df.copy()

# ---------------------------------------------------------
# KPIs (glass cards)
# ---------------------------------------------------------
daily = filtered.groupby("date", as_index=False)["revenue"].sum().sort_values("date")
anoms = detect_anomalies(daily)
avg_day_rev = float(daily["revenue"].mean()) if not daily.empty else 0.0
potential_loss = float(max(0.0, (avg_day_rev*len(anoms) - anoms["revenue"].sum()) if not anoms.empty else 0.0))

by_channel = filtered.groupby("channel", as_index=False).agg(revenue=("revenue","sum"), customers=("customers","sum"))
by_channel["avg_deal_size"] = by_channel["revenue"] / by_channel["customers"].clip(lower=1)
target_ads = float(by_channel["avg_deal_size"].quantile(0.75)) if not by_channel.empty else 0.0
upsell_potential = float(((target_ads - by_channel["avg_deal_size"]).clip(lower=0) * by_channel["customers"]).sum()) if target_ads>0 else 0.0

fc_prophet = forecast_prophet(daily, 30) if PROPHET_OK else pd.DataFrame()
fc_fallback = forecast_fallback(daily, 30) if fc_prophet.empty else pd.DataFrame()
fc = fc_prophet if not fc_prophet.empty else fc_fallback
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
anomaly_days = int(len(anoms)); anomaly_pct = (anomaly_days / max(1, len(daily))) * 100.0
top_ch = by_channel.sort_values("revenue", ascending=False).head(1)
top_channel_name = (top_ch["channel"].iloc[0] if not top_ch.empty else "—")
top_channel_share = float(top_ch["revenue"].iloc[0] / max(1.0, by_channel["revenue"].sum()) * 100.0) if not top_ch.empty else 0.0

g1, g2, g3, g4 = st.columns(4)
with g1:
    st.markdown(f"<div class='card'><div class='kpi-title'>Total Revenue (30d)</div><div class='kpi-value'>{money(total_rev_30)}</div><span class='kpi-delta {'pos' if (delta_total_rev or 0)>=0 else 'neg'}'>{(delta_total_rev or 0):+.1f}% vs prev</span></div>", unsafe_allow_html=True)
with g2:
    delta_avg = pct_delta(avg_rev_30, avg_rev_prev or avg_rev_30)
    st.markdown(f"<div class='card'><div class='kpi-title'>Avg Daily Revenue (30d)</div><div class='kpi-value'>{money(avg_rev_30)}</div><span class='kpi-delta {'pos' if (delta_avg or 0)>=0 else 'neg'}'>{(delta_avg or 0):+.1f}% vs prev</span></div>", unsafe_allow_html=True)
with g3:
    st.markdown(f"<div class='card'><div class='kpi-title'>Anomaly Days</div><div class='kpi-value'>{anomaly_days}</div><div class='muted'>{anomaly_pct:.1f}% of days</div></div>", unsafe_allow_html=True)
with g4:
    st.markdown(f"<div class='card'><div class='kpi-title'>Top Channel</div><div class='kpi-value'>{top_channel_name}</div><div class='muted'>{top_channel_share:.1f}% share</div></div>", unsafe_allow_html=True)

k1,k2,k3 = st.columns(3)
with k1: st.markdown(f"<div class='card'><div class='kpi-title'>Recoverable (Anomalies)</div><div class='kpi-value'>{money(potential_loss)}</div></div>", unsafe_allow_html=True)
with k2: st.markdown(f"<div class='card'><div class='kpi-title'>Upsell Potential</div><div class='kpi-value'>{money(upsell_potential)}</div></div>", unsafe_allow_html=True)
with k3: st.markdown(f"<div class='card'><div class='kpi-title'>30d Forecast Uplift</div><div class='kpi-value'>{money(forecast_uplift)}</div></div>", unsafe_allow_html=True)

st.markdown("<hr class='div'/>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Recovery Moves
# ---------------------------------------------------------
st.subheader("💡 Top Recovery Moves")
moves=[]
if potential_loss>0: moves.append(f"Plug anomaly days → recover ~{money(potential_loss)} (pricing/promos, billing, ops).")
if upsell_potential>0 and not by_channel.empty:
    worst = by_channel.nsmallest(1, "avg_deal_size")
    if not worst.empty: moves.append(f"Lift {worst['channel'].iloc[0]} avg deal size to 75th pct → unlock ~{money(upsell_potential)}.")
if forecast_uplift>0: moves.append(f"Prep capacity & promos for next 30 days → capture ~{money(forecast_uplift)}.")
if not moves: moves=["🎉 No major gaps detected — focus on targeted retention & upsell."]
st.markdown("\n".join([f"- **{i}. {m}**" for i,m in enumerate(moves,1)]))

st.markdown("<hr class='div'/>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Charts (Tabbed)
# ---------------------------------------------------------
tab_trend, tab_dist, tab_forecast = st.tabs(["📈 Trend & Anomalies","📊 Distribution","🔮 Forecast"])
with tab_trend:
    base = alt.Chart(daily).encode(x=alt.X("date:T", title="Date"))
    line = base.mark_line(color=PALETTE["primary"]).encode(
        y=alt.Y("revenue:Q", title="Revenue ($)", axis=alt.Axis(format="~s")),
        tooltip=[alt.Tooltip("date:T"), alt.Tooltip("revenue:Q", format=",.0f")]
    )
    if not anoms.empty:
        pts = alt.Chart(anoms).mark_point(size=85, filled=True, color=PALETTE["danger"]).encode(
            x="date:T", y="revenue:Q", tooltip=["date:T", alt.Tooltip("revenue:Q", format=",.0f")]
        )
        st.altair_chart((line+pts).properties(height=380), use_container_width=True)
    else:
        st.altair_chart(line.properties(height=380), use_container_width=True)

with tab_dist:
    c1, c2 = st.columns(2)
    with c1:
        by_ch = filtered.groupby("channel", as_index=False)["revenue"].sum().sort_values("revenue", ascending=False)
        st.markdown("<div class='section-title'>Revenue by Channel</div>", unsafe_allow_html=True)
        st.altair_chart(alt.Chart(by_ch).mark_bar().encode(
            x=alt.X("revenue:Q", title="Revenue ($)", axis=alt.Axis(format="~s")),
            y=alt.Y("channel:N", sort="-x"),
            tooltip=["channel:N", alt.Tooltip("revenue:Q", format=",.0f")]
        ).properties(height=300), use_container_width=True)
    with c2:
        by_rg = filtered.groupby("region", as_index=False)["revenue"].sum().sort_values("revenue", ascending=False)
        st.markdown("<div class='section-title'>Revenue by Region</div>", unsafe_allow_html=True)
        st.altair_chart(alt.Chart(by_rg).mark_bar().encode(
            x=alt.X("region:N"), y=alt.Y("revenue:Q", axis=alt.Axis(format="~s")),
            tooltip=["region:N", alt.Tooltip("revenue:Q", format=",.0f")]
        ).properties(height=300), use_container_width=True)

with tab_forecast:
    st.markdown("<div class='section-title'>30-Day Revenue Forecast</div>", unsafe_allow_html=True)
    if fc.empty:
        st.info("Not enough history to forecast yet.")
    else:
        st.altair_chart(alt.Chart(fc).mark_line(point=True).encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("value:Q", title="Revenue ($)", axis=alt.Axis(format="~s")),
            color="type:N",
            tooltip=["type:N", "date:T", alt.Tooltip("value:Q", format=",.0f")]
        ).properties(height=380), use_container_width=True)

st.markdown("<hr class='div'/>", unsafe_allow_html=True)

# ---------------------------------------------------------
# ML & DL (cards)
# ---------------------------------------------------------
ml1, ml2 = st.columns(2)
with ml1:
    model_churn, acc_xgb = train_churn_model(filtered)
    st.markdown("<div class='card'><div class='section-title'>🤖 Churn Model</div>", unsafe_allow_html=True)
    if model_churn is None:
        st.markdown("<span class='muted'>Not enough rows/variance to train.</span>", unsafe_allow_html=True)
    else:
        st.markdown(f"<b>Accuracy:</b> {acc_xgb:.2f} {'(XGBoost)' if XGB_OK else '(GBClassifier)'}", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with ml2:
    model_upsell, acc_lgb = train_upsell_model(filtered)
    st.markdown("<div class='card'><div class='section-title'>📈 Upsell Model</div>", unsafe_allow_html=True)
    if model_upsell is None:
        st.markdown("<span class='muted'>Not enough rows/variance to train.</span>", unsafe_allow_html=True)
    else:
        st.markdown(f"<b>Accuracy:</b> {acc_lgb:.2f} {'(LightGBM)' if LGB_OK else '(GBClassifier)'}", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

dl1, dl2 = st.columns(2)
with dl1:
    series = filtered.groupby("date")["revenue"].sum().sort_index()
    m_lstm, lstm_loss = train_lstm_pytorch(series)
    st.markdown("<div class='card'><div class='section-title'>🧠 PyTorch LSTM (demo)</div>", unsafe_allow_html=True)
    if m_lstm is None:
        st.markdown("<span class='muted'>Unavailable (library missing or too few points).</span>", unsafe_allow_html=True)
    else:
        st.markdown(f"<b>Final loss:</b> {lstm_loss:.4f}", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with dl2:
    m_tf, tf_acc = train_tf_dense(filtered)
    st.markdown("<div class='card'><div class='section-title'>🧪 TensorFlow Dense (demo)</div>", unsafe_allow_html=True)
    if m_tf is None:
        st.markdown("<span class='muted'>Unavailable (library missing or too few rows).</span>", unsafe_allow_html=True)
    else:
        st.markdown(f"<b>Accuracy:</b> {tf_acc:.2f}", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.subheader("🚨 Anomaly Days")
if anoms.empty:
    st.caption("No anomalies detected under current filters.")
else:
    st.dataframe(anoms[["date","revenue","anomaly_score"]].sort_values("date", ascending=False), use_container_width=True)

st.markdown("<hr class='div'/>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Export & Share
# ---------------------------------------------------------
st.subheader("⬇️ Export & Share")
kpi_df = pd.DataFrame([
    {"metric":"Total Revenue (30d)","value": total_rev_30},
    {"metric":"Avg Daily Revenue (30d)","value": avg_rev_30},
    {"metric":"Anomaly Days","value": anomaly_days},
    {"metric":"Top Channel","value": top_channel_name},
    {"metric":"Recoverable (Anomalies)","value": potential_loss},
    {"metric":"Upsell Potential","value": upsell_potential},
    {"metric":"30d Forecast Uplift","value": forecast_uplift},
])

cA, cB, cC, cD = st.columns([1,1,1,2])
with cA: st.download_button("Filtered CSV", filtered.to_csv(index=False), "filtered_data.csv", "text/csv")
with cB: st.download_button("KPIs CSV", kpi_df.to_csv(index=False), "kpis.csv", "text/csv")
with cC:
    top_moves_txt = "\n".join(f"{i}. {m}" for i,m in enumerate(moves,1))
    st.download_button("Top Moves TXT", top_moves_txt, "top_moves.txt")
with cD:
    cur_qp = read_query_params()
    share_url = "?" + "&".join([f"{k}={','.join(v) if isinstance(v,list) else v}" for k,v in cur_qp.items()])
    st.text_input("Share this exact view (copy URL)", value=share_url)

if st.button("📄 Download KPI PDF"):
    pdf_bytes = export_kpi_pdf(
        {
            "Total Revenue (30d)": money(total_rev_30),
            "Avg Daily Revenue (30d)": money(avg_rev_30),
            "Anomaly Days": f"{anomaly_days} ({(anomaly_pct):.1f}% of days)",
            "Top Channel": f"{top_channel_name} ({top_channel_share:.1f}% share)",
            "Recoverable (Anomalies)": money(potential_loss),
            "Upsell Potential": money(upsell_potential),
            "30-day Forecast Uplift": money(forecast_uplift),
        }, moves
    )
    st.download_button("Download KPI PDF", data=pdf_bytes, file_name="kpi_report.pdf", mime="application/pdf")

st.markdown("<hr class='div'/>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Simple KB + Assistant (unchanged logic, cleaner UI)
# ---------------------------------------------------------
if "kb" not in st.session_state:
    st.session_state.kb = {"docs": [], "chunks": [], "doc_map": []}

def extract_text_from_file(uploaded):
    name = uploaded.name.lower(); data = uploaded.read()
    if name.endswith(".pdf") and PDF_READER_OK:
        try:
            reader = PdfReader(io.BytesIO(data))
            return "\n".join([p.extract_text() or "" for p in reader.pages])
        except Exception:
            return data.decode("utf-8", errors="ignore")
    try:
        return data.decode("utf-8", errors="ignore")
    except Exception:
        return ""

def chunk_text(txt, size=900, overlap=150):
    s = " ".join(txt.split()); out, i = [], 0
    while i < len(s):
        out.append(s[i:i+size])
        if i+size >= len(s): break
        i = i + size - overlap
    return out

def add_docs_to_kb(files):
    for f in files:
        txt = extract_text_from_file(f); chs = chunk_text(txt); doc_id = str(uuid.uuid4())
        st.session_state.kb["docs"].append({"id": doc_id, "name": f.name, "chunks": chs})
        st.session_state.kb["chunks"].extend(chs); st.session_state.kb["doc_map"].extend([doc_id]*len(chs))

def kb_search(query, top_k=5):
    chunks = st.session_state.kb["chunks"]
    if not chunks: return []
    tokens = [t for t in re.findall(r"\w+", query.lower()) if t]
    scores = []
    for i, ch in enumerate(chunks):
        text = ch.lower()
        s = sum(text.count(t) for t in tokens) if tokens else 0
        scores.append((i, s))
    scores.sort(key=lambda x: x[1], reverse=True)
    results = []
    for i, s in scores[:top_k]:
        doc_id = st.session_state.kb["doc_map"][i]
        name = next((d["name"] for d in st.session_state.kb["docs"] if d["id"]==doc_id), "KB")
        results.append({"chunk": chunks[i][:300], "score": int(s), "doc": name})
    return results

def crawl_and_add(urls, max_pages=5):
    if not BS4_OK: return 0
    added = 0
    for u in urls[:max_pages]:
        try:
            r = requests.get(u, timeout=8, headers={"User-Agent":"AI-RevBot"})
            soup = BeautifulSoup(r.text, "html.parser")
            for tag in soup(["script","style","noscript"]): tag.extract()
            text = soup.get_text(" ")
            chs = chunk_text(text); doc_id = str(uuid.uuid4())
            st.session_state.kb["docs"].append({"id":doc_id,"name":u,"chunks":chs})
            st.session_state.kb["chunks"].extend(chs); st.session_state.kb["doc_map"].extend([doc_id]*len(chs))
            added += 1
        except Exception:
            pass
    return added

st.subheader("📚 Knowledge Base")
kb_tab1, kb_tab2 = st.tabs(["Upload Docs", "URL Crawler"])
with kb_tab1:
    files = st.file_uploader("Add PDF/TXT/CSV/MD", type=["pdf","txt","csv","md"], accept_multiple_files=True)
    if st.button("Add to KB"):
        if files: add_docs_to_kb(files); st.success(f"Added {len(files)} file(s).")
        else: st.info("Select file(s) first.")
    if st.session_state.kb["docs"]:
        st.caption("KB Files:")
        for d in st.session_state.kb["docs"]:
            st.caption(f"• {d['name']} ({len(d['chunks'])} chunks)")
    q = st.text_input("Search your KB")
    if q:
        hits = kb_search(q, top_k=5)
        if not hits: st.info("No matches.")
        for h in hits:
            st.markdown(f"**{h['doc']}** — score {h['score']}  \n{h['chunk']}…")

with kb_tab2:
    if not BS4_OK:
        st.info("Install beautifulsoup4 + requests to enable crawling.")
    else:
        urls_text = st.text_area("Enter URLs (one per line)")
        max_pages = st.slider("Max pages", 1, 20, 5)
        if st.button("Crawl & Add"):
            seeds = [u.strip() for u in urls_text.splitlines() if u.strip()]
            if not seeds: st.info("Add at least one URL.")
            else:
                added = crawl_and_add(seeds, max_pages=max_pages)
                st.success(f"Added {added} page(s).")

st.subheader("🤖 Assistant")
if "chat" not in st.session_state:
    st.session_state.chat = [{"role":"assistant","content":"Hi! Ask about anomalies, upsell, forecast, or your KB."}]
for m in st.session_state.chat:
    with st.chat_message(m["role"]): st.markdown(m["content"])
msg = st.chat_input("Type your question…")
if msg:
    st.session_state.chat.append({"role":"user","content":msg})
    with st.chat_message("user"): st.markdown(msg)
    # Compose summary + KB
    # (Fast, local response – replace with real RAG later)
    daily = filtered.groupby("date", as_index=False)["revenue"].sum().sort_values("date")
    anoms = detect_anomalies(daily)
    avg_day_rev = float(daily["revenue"].mean()) if not daily.empty else 0.0
    potential_loss = float(max(0.0, (avg_day_rev*len(anoms) - anoms["revenue"].sum()) if not anoms.empty else 0.0))
    by_channel = filtered.groupby("channel", as_index=False).agg(revenue=("revenue","sum"), customers=("customers","sum"))
    by_channel["avg_deal_size"] = by_channel["revenue"] / by_channel["customers"].clip(lower=1)
    target_ads = float(by_channel["avg_deal_size"].quantile(0.75)) if not by_channel.empty else 0.0
    upsell_potential = float(((target_ads - by_channel["avg_deal_size"]).clip(lower=0) * by_channel["customers"]).sum()) if target_ads>0 else 0.0
    fc_prophet = forecast_prophet(daily, 30) if PROPHET_OK else pd.DataFrame()
    fc_fallback = forecast_fallback(daily, 30) if fc_prophet.empty else pd.DataFrame()
    fc = fc_prophet if not fc_prophet.empty else fc_fallback
    future_sum = float(fc[fc["type"]=="Forecast"]["value"].sum()) if not fc.empty else 0.0
    baseline_mean = float(daily["revenue"].tail(30).mean() if len(daily)>=30 else (daily["revenue"].mean() if not daily.empty else 0.0))
    forecast_uplift = float(max(0.0, future_sum - baseline_mean*30))
    summary = f"30d revenue {money(daily.tail(30)['revenue'].sum()) if not daily.empty else '$0'}, avg/day {money(avg_day_rev)}, recoverable {money(potential_loss)}, upsell {money(upsell_potential)}, uplift {money(forecast_uplift)}."
    hits = kb_search(msg, top_k=3)
    kb_txt = "\n".join([f"[{i+1}] {h['doc']}: {h['chunk']}…" for i,h in enumerate(hits)]) if hits else ""
    reply = f"**Answer (data-aware):** {summary}\n" + (f"\n**KB context:**\n{kb_txt}" if kb_txt else "")
    st.session_state.chat.append({"role":"assistant","content":reply})
    with st.chat_message("assistant"): st.markdown(reply)

st.markdown("<hr class='div'/>", unsafe_allow_html=True)
st.markdown("<div class='cta'><b>Want a private pilot?</b> Upload your latest CSV and docs — we’ll surface your top recovery moves in minutes.</div>", unsafe_allow_html=True)
