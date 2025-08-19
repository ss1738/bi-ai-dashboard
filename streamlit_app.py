import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.arima.model import ARIMA

# ------------------ Page config ------------------
st.set_page_config(page_title="AI BI Dashboard", layout="wide")
st.title("📊 AI BI Dashboard — MVP")

st.caption(
    "Upload a CSV or use the demo. Map your columns below. "
    "The app handles messy dates, missing values, and odd column names."
)

# ------------------ Data load ------------------
with st.sidebar:
    st.header("📁 Dataset")
    up = st.file_uploader("Upload CSV (optional)", type=["csv"], key="u_csv")
    use_demo = st.checkbox("Use demo data", value=(up is None), key="use_demo")

if use_demo:
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=40, freq="D"),
        "category": (["Electronics","Fashion","Groceries","Home","Sports"] * 8),
        "sales":   [1200,900,600,800,700]*8,
        "profit":  [ 200,150, 80,120,100]*8,
    })
else:
    try:
        df = pd.read_csv(up)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()

# Basic hygiene
raw_rows = len(df)
df = df.copy()
df.columns = [str(c).strip() for c in df.columns]

# ------------------ Column mapping (real-world friendly) ------------------
st.subheader("🧭 Column Mapping")
cols = df.columns.tolist()

# Heuristics for good defaults
def guess(colnames, candidates):
    cand = [c for c in candidates if c in colnames]
    return cand[0] if cand else None

date_guess    = guess([c.lower() for c in cols], ["date","order_date","invoice_date","timestamp"])
cat_guess     = guess([c.lower() for c in cols], ["category","segment","product","dept"])
sales_guess   = guess([c.lower() for c in cols], ["sales","revenue","amount","net_sales"])
profit_guess  = guess([c.lower() for c in cols], ["profit","margin","gross_profit"])

map_cols = st.columns(4)
with map_cols[0]:
    date_col   = st.selectbox("Date/time column (optional)", [None] + cols, index=(cols.index(date_guess) + 1) if date_guess and date_guess in [c.lower() for c in cols] else 0, key="m_date")
with map_cols[1]:
    cat_col    = st.selectbox("Category column (optional)", [None] + cols, index=(cols.index(next((c for c in cols if c.lower()==cat_guess), cols[0]))+1) if cat_guess else 0, key="m_cat")
with map_cols[2]:
    sales_col  = st.selectbox("Sales/Value column (required)", cols, index=cols.index(next((c for c in cols if c.lower()==sales_guess), cols[0])), key="m_sales")
with map_cols[3]:
    profit_col = st.selectbox("Profit column (optional)", [None] + cols, index=(cols.index(next((c for c in cols if c.lower()==profit_guess), cols[0]))+1) if profit_guess else 0, key="m_profit")

# Type coercion & cleaning
def to_datetime_safe(s):
    try:
        return pd.to_datetime(s, errors="coerce")
    except Exception:
        return pd.to_datetime(pd.Series(["1900-01-01"]).repeat(len(s)), errors="coerce")

def to_numeric_safe(s):
    return pd.to_numeric(s, errors="coerce")

if date_col:
    df[date_col] = to_datetime_safe(df[date_col])
if sales_col:
    df[sales_col] = to_numeric_safe(df[sales_col])
if profit_col:
    df[profit_col] = to_numeric_safe(df[profit_col]) if profit_col else None

# Drop obviously broken rows for core metrics
keep_cols = [c for c in [date_col, cat_col, sales_col, profit_col] if c]
clean = df.copy()
if sales_col:
    clean = clean[~clean[sales_col].isna()]
drop_cnt = raw_rows - len(clean)

with st.expander("🔍 Data preview & cleaning summary", expanded=False):
    st.write(f"Rows: {raw_rows} → after cleaning: {len(clean)} (dropped {drop_cnt})")
    st.dataframe(clean.head(25), use_container_width=True)

# ------------------ Sidebar filters ------------------
with st.sidebar:
    st.header("🔎 Filters")
    if cat_col:
        cats = sorted(clean[cat_col].dropna().astype(str).unique())
        sel_cats = st.multiselect("Categories", cats, default=cats, key="f_cats")
    else:
        sel_cats = []
    if date_col:
        dmin, dmax = pd.Timestamp(clean[date_col].min()), pd.Timestamp(clean[date_col].max())
        d1, d2 = st.date_input("Date range", [dmin.date(), dmax.date()], key="f_dates")
    else:
        d1 = d2 = None

fdf = clean.copy()
if cat_col and sel_cats:
    fdf = fdf[fdf[cat_col].astype(str).isin(sel_cats)]
if date_col and d1 and d2:
    fdf = fdf[(fdf[date_col] >= pd.to_datetime(d1)) & (fdf[date_col] <= pd.to_datetime(d2))]

# ------------------ Tabs ------------------
t_dash, t_seg, t_an, t_fc, t_ai = st.tabs(
    ["📊 Dashboard", "📌 Segmentation", "⚠️ Anomalies", "🔮 Forecast", "🤖 AI Insights"]
)

# ------------------ Dashboard ------------------
with t_dash:
    st.subheader("📈 Key Metrics")
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Rows", f"{len(fdf):,}")
    c2.metric("Columns", f"{fdf.shape[1]}")
    c3.metric("Total Sales", f"{fdf[sales_col].sum():,.0f}" if sales_col in fdf else "—")
    c4.metric("Avg Sales", f"{fdf[sales_col].mean():,.2f}" if sales_col in fdf else "—")
    if profit_col:
        c5.metric("Total Profit", f"{fdf[profit_col].sum():,.0f}")
    else:
        c5.metric("Total Profit", "—")

    st.markdown("### 📊 Charts")
    if date_col and sales_col in fdf:
        fig_ts = px.line(
            fdf.sort_values(date_col),
            x=date_col, y=sales_col,
            color=cat_col if cat_col else None,
            markers=True, title=f"{sales_col} over time"
        )
        st.plotly_chart(fig_ts, use_container_width=True, key="chart_ts")

    if cat_col and sales_col in fdf:
        gp = fdf.groupby(cat_col, as_index=False)[sales_col].sum().sort_values(sales_col, ascending=False)
        fig_bar = px.bar(gp, x=cat_col, y=sales_col, title=f"{sales_col} by {cat_col}")
        st.plotly_chart(fig_bar, use_container_width=True, key="chart_bar")

    if profit_col and cat_col and profit_col in fdf:
        gp2 = fdf.groupby(cat_col, as_index=False)[profit_col].sum()
        fig_pie = px.pie(gp2, names=cat_col, values=profit_col, title=f"{profit_col} share by {cat_col}")
        st.plotly_chart(fig_pie, use_container_width=True, key="chart_pie")

    st.markdown("### ⬇️ Export")
    st.download_button(
        "Download filtered data (CSV)",
        fdf.to_csv(index=False).encode("utf-8"),
        file_name="filtered_data.csv",
        mime="text/csv",
        key="dl_csv_main",
    )

# ------------------ Segmentation ------------------
with t_seg:
    st.subheader("📌 Segmentation (KMeans)")
    num_candidates = [c for c in fdf.columns if pd.api.types.is_numeric_dtype(fdf[c])]
    if len(num_candidates) < 2:
        st.info("Need at least 2 numeric columns (e.g., sales, profit).")
    else:
        feats = st.multiselect("Numeric features", num_candidates, default=[c for c in [sales_col, profit_col] if c] or num_candidates[:2], key="seg_feats")
        k = st.slider("k (clusters)", 2, 8, 3, key="seg_k")
        if feats and len(fdf.dropna(subset=feats)) >= k:
            X = fdf[feats].dropna().copy()
            kmeans = KMeans(n_clusters=k, n_init="auto", random_state=42).fit(X)
            fdf["_segment"] = pd.NA
            fdf.loc[X.index, "_segment"] = kmeans.labels_
            fig_seg = px.scatter(X, x=feats[0], y=feats[1] if len(feats) > 1 else feats[0],
                                 color=kmeans.labels_.astype(str),
                                 title="Cluster visualization")
            st.plotly_chart(fig_seg, use_container_width=True, key="seg_plot")
            out = fdf.loc[X.index, [*feats]].copy()
            out["_segment"] = kmeans.labels_
            st.download_button("Download clusters (CSV)", out.to_csv(index=False).encode(), "clusters.csv", "text/csv", key="dl_clusters")
        else:
            st.warning("Pick features and ensure enough rows (rows ≥ k).")

# ------------------ Anomalies ------------------
with t_an:
    st.subheader("⚠️ Anomaly Detection (IsolationForest)")
    feats2 = st.multiselect(
        "Numeric features to score",
        [c for c in fdf.columns if pd.api.types.is_numeric_dtype(fdf[c])],
        default=[c for c in [sales_col, profit_col] if c],
        key="anom_feats"
    )
    cont = st.slider("Contamination (outlier %)", 0.01, 0.20, 0.05, key="anom_cont")
    if feats2:
        X2 = fdf.dropna(subset=feats2).copy()
        if len(X2) > 10:
            iso = IsolationForest(contamination=cont, random_state=42).fit(X2[feats2])
            scores = -iso.score_samples(X2[feats2])
            report = X2.copy()
            report["_anomaly_score"] = scores
            st.dataframe(report.sort_values("_anomaly_score", ascending=False).head(20), use_container_width=True)
            st.plotly_chart(px.histogram(report, x="_anomaly_score", nbins=30, title="Anomaly score distribution"), use_container_width=True, key="anom_hist")
            st.download_button("Download anomaly report (CSV)", report.to_csv(index=False).encode(), "anomalies.csv", "text/csv", key="dl_anom")
        else:
            st.warning("Need >10 rows after filtering.")
    else:
        st.info("Select at least one numeric feature.")

# ------------------ Forecast ------------------
with t_fc:
    st.subheader("🔮 Forecast (ARIMA, auto-aggregated)")
    if not date_col or sales_col not in fdf:
        st.info("Select a date and a numeric target (sales) to forecast.")
    else:
        freq = st.selectbox("Resample frequency", ["D","W","M"], index=1, key="fc_freq")
        horizon = st.slider("Horizon (periods)", 4, 26, 12, key="fc_h")
        ts = fdf[[date_col, sales_col]].dropna()
        ts = ts.groupby(date_col, as_index=False)[sales_col].sum()
        ts.index = pd.to_datetime(ts[date_col])
        y = ts[[sales_col]].resample(freq).sum().dropna()[sales_col]
        if len(y) < 8:
            st.warning("Need at least 8 periods after resampling.")
        else:
            try:
                model = ARIMA(y, order=(1,1,1))
                fit = model.fit()
                fc = fit.forecast(steps=horizon)
                hist = pd.DataFrame({"date": y.index, "value": y.values})
                fc_df = pd.DataFrame({"date": fc.index, "forecast": fc.values})
                fig_fc = px.line(hist, x="date", y="value", title=f"{sales_col} — history & forecast")
                fig_fc.add_scatter(x=fc_df["date"], y=fc_df["forecast"], mode="lines+markers", name="forecast")
                st.plotly_chart(fig_fc, use_container_width=True, key="fc_plot")
                st.download_button("Download forecast (CSV)", fc_df.to_csv(index=False).encode(), "forecast.csv", "text/csv", key="dl_fc")
            except Exception as e:
                st.error(f"Forecast failed: {e}")

# ------------------ AI Insights (simple, rule-based) ------------------
with t_ai:
    st.subheader("🤖 AI-style Insights (rule-based MVP)")
    bullets = []
    if sales_col in fdf:
        total = fdf[sales_col].sum()
        mean = fdf[sales_col].mean()
        bullets.append(f"Total {sales_col}: **{total:,.0f}**; average per row: **{mean:,.2f}**.")
    if cat_col and sales_col in fdf:
        gp = fdf.groupby(cat_col, as_index=False)[sales_col].sum().sort_values(sales_col, ascending=False)
        if len(gp):
            top = gp.iloc[0]
            share = 100 * top[sales_col] / (fdf[sales_col].sum() or 1)
            bullets.append(f"Top {cat_col}: **{top[cat_col]}** with **{top[sales_col]:,.0f}** ({share:.1f}% of total).")
    if date_col and sales_col in fdf:
        # Week-over-week change heuristic
        by_week = fdf.set_index(date_col).resample("W")[sales_col].sum()
        if len(by_week) >= 2:
            delta = (by_week.iloc[-1] - by_week.iloc[-2]) / (by_week.iloc[-2] or 1)
            bullets.append(f"Last week vs prior: **{delta*100:+.1f}%** change.")
    if profit_col and profit_col in fdf and sales_col in fdf and fdf[sales_col].sum() > 0:
        margin = 100 * fdf[profit_col].sum() / fdf[sales_col].sum()
        bullets.append(f"Overall margin: **{margin:.1f}%**.")
    if not bullets:
        bullets.append("Provide at least a value column (e.g., sales) for insights.")
    for b in bullets:
        st.markdown(f"- {b}")

    st.caption("Upgrade idea: swap this with an LLM to generate natural-language summaries from the same stats.")
