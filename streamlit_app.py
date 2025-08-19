import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="BI Dashboard + AI Insights", layout="wide")
st.title("📊 Business Intelligence Dashboard + 🤖 AI Insights")

# -------- Demo data (always available) --------
demo_df = pd.DataFrame({
    "date": pd.date_range("2025-01-01", periods=36, freq="W"),
    "category": (["Electronics","Clothing","Home","Grocery","Sports","Beauty"] * 6)[:36],
    "sales":  [2000,1500,1800,2200,2100,1600]*6,
    "profit": [ 300, 200, 260, 340, 320, 230]*6,
    "units":  [ 400, 300, 350, 450, 420, 310]*6,
})

def coerce_dates(d: pd.DataFrame) -> pd.DataFrame:
    for c in d.columns:
        if d[c].dtype == "object":
            try: d[c] = pd.to_datetime(d[c])
            except Exception: pass
    return d

# -------- Sidebar: upload + filters --------
with st.sidebar:
    st.header("Dataset")
    f = st.file_uploader("Upload CSV (optional)", type=["csv"])
    use_demo = st.checkbox("Use demo data", value=True)

if use_demo or not f:
    df = demo_df.copy()
else:
    try:
        df = coerce_dates(pd.read_csv(f))
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        st.stop()

# Ensure a date column if present
time_candidates = [c for c in df.columns if "date" in c.lower()] + \
                  [c for c in df.columns if str(df[c].dtype).startswith("datetime64")]
date_col = time_candidates[0] if time_candidates else None

num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
cat_cols = [c for c in df.columns if df[c].dtype == "object"]

# --- Filters (new) ---
with st.sidebar:
    st.header("Filters")
    # Category filter
    cat_for_filter = None
    if cat_cols:
        cat_for_filter = st.selectbox("Category column", options=cat_cols, index=0)
        selected_cats = st.multiselect("Include categories", options=sorted(df[cat_for_filter].astype(str).unique()))
    else:
        selected_cats = []

    # Date range filter
    if date_col is not None:
        dmin, dmax = pd.to_datetime(df[date_col].min()), pd.to_datetime(df[date_col].max())
        drange = st.date_input("Date range", value=(dmin.date(), dmax.date()))
        if isinstance(drange, tuple) and len(drange) == 2:
            start_date, end_date = pd.to_datetime(drange[0]), pd.to_datetime(drange[1])
        else:
            start_date, end_date = dmin, dmax
    else:
        start_date = end_date = None

# Apply filters
fdf = df.copy()
if cat_cols and cat_for_filter and len(selected_cats) > 0:
    fdf = fdf[fdf[cat_for_filter].astype(str).isin(selected_cats)]
if date_col is not None and start_date is not None and end_date is not None:
    fdf = fdf[(pd.to_datetime(fdf[date_col]) >= start_date) & (pd.to_datetime(fdf[date_col]) <= end_date)]

# Helpers recalculated on filtered data
num_cols_f = [c for c in fdf.columns if pd.api.types.is_numeric_dtype(fdf[c])]
cat_cols_f = [c for c in fdf.columns if fdf[c].dtype == "object"]
time_cols_f = [c for c in fdf.columns if "date" in c.lower()] + \
              [c for c in fdf.columns if str(fdf[c].dtype).startswith("datetime64")]

# -------- Tabs --------
tab_ov, tab_seg, tab_an, tab_ai = st.tabs(
    ["Overview", "Segmentation (KMeans)", "Anomalies (IsolationForest)", "AI Q&A"]
)

with tab_ov:
    st.subheader("Data Preview (filtered)")
    st.caption(f"Rows after filters: {len(fdf):,}")
    st.dataframe(fdf.head(50), use_container_width=True)

    st.markdown("### KPIs")
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Rows", f"{len(fdf):,}")
    c2.metric("Columns", f"{fdf.shape[1]}")
    if "sales" in fdf.columns:
        c3.metric("Total Sales", f"{fdf['sales'].sum():,.0f}")
        c4.metric("Avg Sales", f"{fdf['sales'].mean():,.2f}")
    if "profit" in fdf.columns and "sales" in fdf.columns and fdf["sales"].sum() != 0:
        margin = 100 * fdf["profit"].sum() / fdf["sales"].sum()
        c5.metric("Profit Margin %", f"{margin:.1f}%")

    st.markdown("### Charts")
    tcol = st.selectbox("Time column", options=time_cols_f or [None], key="tcol")
    # Prefer sales/profit/units in that order
    preferred = [c for c in ["sales","profit","units"] if c in fdf.columns]
    y_options = preferred + [c for c in num_cols_f if c not in preferred]
    ycol = st.selectbox("Value", options=y_options or [None], index=0 if y_options else 0, key="ycol")
    ccol = st.selectbox("Category", options=cat_cols_f or [None], index=0 if cat_cols_f else 0, key="ccol")

    if tcol and ycol and tcol in fdf and ycol in fdf:
        st.plotly_chart(
            px.line(fdf.sort_values(tcol), x=tcol, y=ycol, title=f"{ycol} over time"),
            use_container_width=True
        )
    if ccol and ycol and ccol in fdf and ycol in fdf:
        gp = fdf.groupby(ccol, as_index=False)[ycol].sum().sort_values(ycol, ascending=False)
        st.plotly_chart(
            px.bar(gp, x=ccol, y=ycol, title=f"{ycol} by {ccol}"),
            use_container_width=True
        )
    # Profit share pie (new)
    if "profit" in fdf.columns and ccol and ccol in fdf:
        gp2 = fdf.groupby(ccol, as_index=False)["profit"].sum()
        st.plotly_chart(px.pie(gp2, names=ccol, values="profit", title="Profit share by category"),
                        use_container_width=True)

with tab_seg:
    st.subheader("Customer/Product Segmentation (KMeans)")
    if len(num_cols_f) < 2:
        st.info("Need at least 2 numeric columns.")
    else:
        feats = st.multiselect("Numeric features", num_cols_f, default=num_cols_f[:2])
        k = st.slider("Clusters (k)", 2, 8, 4)
        if feats:
            X = fdf[feats].dropna().copy()
            if len(X) >= k:
                Xs = StandardScaler().fit_transform(X)
                km = KMeans(n_clusters=k, n_init="auto", random_state=42).fit(Xs)
                labels = km.labels_
                viz = pd.DataFrame(Xs[:, :2], columns=["feat1","feat2"], index=X.index)
                viz["_cluster"] = labels.astype(str)
                st.plotly_chart(px.scatter(viz, x="feat1", y="feat2", color="_cluster",
                                           title="Cluster visualization"),
                                use_container_width=True)
                out = fdf.loc[X.index].copy()
                out["_cluster"] = labels
                st.download_button("Download clusters CSV", out.to_csv(index=False).encode(),
                                   file_name="clusters.csv", mime="text/csv")
            else:
                st.warning("Not enough rows after filtering to form clusters.")

with tab_an:
    st.subheader("Anomaly Detection (IsolationForest)")
    if not num_cols_f:
        st.info("Need at least 1 numeric column.")
    else:
        feats2 = st.multiselect("Numeric features", num_cols_f, default=num_cols_f[: min(3, len(num_cols_f))])
        cont = st.slider("Contamination (outlier %)", 0.01, 0.20, 0.05)
        if feats2:
            X2 = fdf[feats2].dropna().copy()
            if len(X2) > 10:
                X2s = StandardScaler().fit_transform(X2)
                iso = IsolationForest(contamination=cont, random_state=42).fit(X2s)
                scores = -iso.score_samples(X2s)
                out2 = fdf.loc[X2.index].copy()
                out2["_anomaly_score"] = scores
                st.dataframe(out2.sort_values("_anomaly_score", ascending=False).head(20),
                             use_container_width=True)
                st.plotly_chart(px.histogram(out2, x="_anomaly_score", nbins=30,
                                             title="Anomaly scores"),
                                use_container_width=True)
                st.download_button("Download anomaly report", out2.to_csv(index=False).encode(),
                                   file_name="anomalies.csv", mime="text/csv")
            else:
                st.warning("Need >10 rows after filtering for stable anomaly scoring.")

with tab_ai:
    st.subheader("Ask questions (rules-based MVP)")
    st.caption("Try: *Which category has highest sales?*")
    q = st.text_input("Your question")
    if q:
        ans = "I need a numeric and a category column."
        if num_cols_f and cat_cols_f:
            num = "sales" if "sales" in num_cols_f else num_cols_f[0]
            cat = cat_cols_f[0]
            gp = fdf.groupby(cat, as_index=False)[num].sum().sort_values(num, ascending=False)
            if len(gp):
                best = gp.iloc[0]
                ans = f"Top {cat} by total {num}: **{best[cat]}** ({best[num]:,.0f})."
        st.markdown(ans)
        if 'gp' in locals():
            st.dataframe(gp.head(5), use_container_width=True)

