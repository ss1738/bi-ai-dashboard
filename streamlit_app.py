import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="BI Dashboard + AI Insights", layout="wide")
st.title("📊 Business Intelligence Dashboard + 🤖 AI Insights")

# Demo data (no CSV needed)
df = pd.DataFrame({
    "date": pd.date_range("2025-01-01", periods=36, freq="W"),
    "category": (["Electronics","Clothing","Home","Grocery","Sports","Beauty"] * 6)[:36],
    "sales":  [2000,1500,1800,2200,2100,1600]*6,
    "profit": [ 300, 200, 260, 340, 320, 230]*6,
    "units":  [ 400, 300, 350, 450, 420, 310]*6,
})

# Try date parsing for user CSVs
def coerce_dates(d):
    for c in d.columns:
        if d[c].dtype == "object":
            try: d[c] = pd.to_datetime(d[c])
            except Exception: pass
    return d

with st.sidebar:
    st.header("Dataset")
    f = st.file_uploader("Upload CSV (optional)", type=["csv"])
    use_demo = st.checkbox("Use demo data", value=True)

if not use_demo and f is not None:
    try:
        df = coerce_dates(pd.read_csv(f))
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        st.stop()

num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
cat_cols = [c for c in df.columns if df[c].dtype == "object"]
time_cols = [c for c in df.columns if "date" in c.lower()] + \
            [c for c in df.columns if str(df[c].dtype).startswith("datetime64")]

tab_ov, tab_seg, tab_an, tab_ai = st.tabs(
    ["Overview", "Segmentation", "Anomalies", "AI Q&A"]
)

with tab_ov:
    st.subheader("Data Preview")
    st.dataframe(df.head(50), use_container_width=True)

    st.markdown("### KPIs")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", f"{df.shape[1]}")
    if num_cols:
        c3.metric(f"Sum({num_cols[0]})", f"{df[num_cols[0]].sum():,.0f}")
        c4.metric(f"Avg({num_cols[0]})", f"{df[num_cols[0]].mean():,.2f}")

    st.markdown("### Charts")
    tcol = st.selectbox("Time column", options=time_cols or [None])
    ycol = st.selectbox("Value", options=num_cols or [None], index=0 if num_cols else 0)
    ccol = st.selectbox("Category", options=cat_cols or [None], index=0 if cat_cols else 0)
    if tcol and ycol and tcol in df and ycol in df:
        st.plotly_chart(px.line(df.sort_values(tcol), x=tcol, y=ycol, title=f"{ycol} over time"),
                        use_container_width=True)
    if ccol and ycol and ccol in df and ycol in df:
        gp = df.groupby(ccol, as_index=False)[ycol].sum().sort_values(ycol, ascending=False)
        st.plotly_chart(px.bar(gp, x=ccol, y=ycol, title=f"{ycol} by {ccol}"),
                        use_container_width=True)

with tab_seg:
    st.subheader("Segmentation (KMeans)")
    if len(num_cols) < 2:
        st.info("Need at least 2 numeric columns.")
    else:
        feats = st.multiselect("Numeric features", num_cols, default=num_cols[:2])
        k = st.slider("Clusters (k)", 2, 8, 4)
        if feats:
            X = df[feats].dropna().copy()
            Xs = StandardScaler().fit_transform(X)
            km = KMeans(n_clusters=k, n_init="auto", random_state=42).fit(Xs)
            labels = km.labels_
            viz = pd.DataFrame(Xs[:, :2], columns=["feat1","feat2"], index=X.index)
            viz["_cluster"] = labels.astype(str)
            st.plotly_chart(px.scatter(viz, x="feat1", y="feat2", color="_cluster",
                                       title="Cluster visualization"), use_container_width=True)
            out = df.loc[X.index].copy()
            out["_cluster"] = labels
            st.download_button("Download clusters CSV", out.to_csv(index=False).encode(),
                               file_name="clusters.csv", mime="text/csv")

with tab_an:
    st.subheader("Anomalies (IsolationForest)")
    if not num_cols:
        st.info("Need at least 1 numeric column.")
    else:
        feats2 = st.multiselect("Numeric features", num_cols, default=num_cols[: min(3, len(num_cols))])
        cont = st.slider("Contamination (outlier %)", 0.01, 0.20, 0.05)
        if feats2:
            X2 = df[feats2].dropna().copy()
            X2s = StandardScaler().fit_transform(X2)
            iso = IsolationForest(contamination=cont, random_state=42).fit(X2s)
            scores = -iso.score_samples(X2s)
            out2 = df.loc[X2.index].copy()
            out2["_anomaly_score"] = scores
            st.dataframe(out2.sort_values("_anomaly_score", ascending=False).head(20),
                         use_container_width=True)
            st.plotly_chart(px.histogram(out2, x="_anomaly_score", nbins=30, title="Anomaly scores"),
                            use_container_width=True)
            st.download_button("Download anomaly report", out2.to_csv(index=False).encode(),
                               file_name="anomalies.csv", mime="text/csv")

with tab_ai:
    st.subheader("Ask questions (rules-based MVP)")
    st.caption("Try: *Which category has highest sales?*")
    q = st.text_input("Your question")
    if q:
        ans = "I need a numeric and a category column."
        if num_cols and cat_cols:
            num = num_cols[0]; cat = cat_cols[0]
            gp = df.groupby(cat, as_index=False)[num].sum().sort_values(num, ascending=False)
            best = gp.iloc[0]
            ans = f"Top {cat} by total {num}: **{best[cat]}** ({best[num]:,.0f})."
        st.markdown(ans)
        if 'gp' in locals():
            st.dataframe(gp.head(5), use_container_width=True)
