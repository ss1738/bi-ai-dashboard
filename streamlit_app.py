import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import Ridge

# --------- Page setup ---------
st.set_page_config(page_title="BI + AI Dashboard", layout="wide")
st.title("📊 Interactive BI Dashboard + 🤖 AI Insights")

# --------- Data: upload OR built-in demo (always available) ---------
f = st.file_uploader("Upload CSV (optional)", type=["csv"])

if f is not None:
    df = pd.read_csv(f)
else:
    st.info("📂 No file uploaded — using built-in demo data.")
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=36, freq="W"),
        "category": (["Electronics","Fashion","Groceries","Home","Sports","Beauty"]*6)[:36],
        "sales":  [2000,1500, 800,1800,1600,1200]*6,
        "profit": [ 300, 200, 100, 260, 220, 180]*6,
        "units":  [ 400, 300, 220, 350, 320, 270]*6,
    })

# Safe date parsing
if "date" in df.columns:
    try:
        df["date"] = pd.to_datetime(df["date"])
    except Exception:
        pass

# --------- Sidebar filters ---------
st.sidebar.header("🔎 Filters")

# Category filter (if available)
cat_cols = [c for c in df.columns if df[c].dtype == "object"]
cat_col = cat_cols[0] if cat_cols else None
if cat_col:
    all_cats = sorted(df[cat_col].astype(str).unique())
    picked = st.sidebar.multiselect("Select categories", options=all_cats, default=all_cats)
else:
    picked = []

# Date range filter (if available)
date_col = None
for c in df.columns:
    if "date" in c.lower() or str(df[c].dtype).startswith("datetime64"):
        date_col = c
        break

if date_col:
    dmin, dmax = pd.to_datetime(df[date_col].min()), pd.to_datetime(df[date_col].max())
    start, end = st.sidebar.date_input("Date range", [dmin.date(), dmax.date()])
    if not isinstance(start, datetime): start = dmin
    if not isinstance(end, datetime):   end = dmax
else:
    start = end = None

# Apply filters
fdf = df.copy()
if cat_col and picked:
    fdf = fdf[fdf[cat_col].astype(str).isin(picked)]
if date_col and start is not None and end is not None:
    fdf = fdf[(pd.to_datetime(fdf[date_col]) >= pd.to_datetime(start)) &
              (pd.to_datetime(fdf[date_col]) <= pd.to_datetime(end))]

# Helpers
num_cols = [c for c in fdf.columns if pd.api.types.is_numeric_dtype(fdf[c])]
time_cols = [c for c in fdf.columns if "date" in c.lower()] + \
            [c for c in fdf.columns if str(fdf[c].dtype).startswith("datetime64")]

# --------- Tabs ---------
tab_ov, tab_seg, tab_an, tab_fc, tab_ai = st.tabs(
    ["Overview", "Segmentation (KMeans)", "Anomalies (IsolationForest)", "Forecast", "AI Q&A"]
)

# --------- Overview ---------
with tab_ov:
    st.subheader("Data Preview (filtered)")
    st.caption(f"Rows after filters: {len(fdf):,}")
    st.dataframe(fdf.head(50), use_container_width=True)

    # KPIs
    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    c1.metric("Rows", f"{len(fdf):,}")
    c2.metric("Columns", f"{fdf.shape[1]}")
    if "sales" in fdf:
        c3.metric("Total Sales", f"{fdf['sales'].sum():,.0f}")
        c6.metric("Avg Sales", f"{fdf['sales'].mean():,.2f}")
    if "profit" in fdf:
        c4.metric("Total Profit", f"{fdf['profit'].sum():,.0f}")
        c7.metric("Avg Profit", f"{fdf['profit'].mean():,.2f}")
    if "sales" in fdf and "profit" in fdf and fdf["sales"].sum() != 0:
        margin = 100 * fdf["profit"].sum() / fdf["sales"].sum()
        c5.metric("Profit Margin %", f"{margin:.1f}%")

    st.markdown("### Charts")
    # Pick sensible defaults
    y_pref = [c for c in ["sales","profit","units"] if c in fdf.columns] or num_cols
    tcol = st.selectbox("Time column", options=time_cols or [None], index=0 if time_cols else 0, key="tcol")
    ycol = st.selectbox("Value", options=y_pref or [None], index=0 if y_pref else 0, key="ycol")
    ccol = st.selectbox("Category", options([cat_col] if cat_col else []) if False else [cat_col] if cat_col else [None],
                        index=0, key="ccol")

    # Time series
    if tcol and ycol and tcol in fdf and ycol in fdf:
        st.plotly_chart(px.line(fdf.sort_values(tcol), x=tcol, y=ycol, color=ccol if ccol in fdf else None,
                                markers=True, title=f"{ycol} over time"), use_container_width=True)

    # Bar by category
    if ccol in fdf and ycol in fdf:
        gp = fdf.groupby(ccol, as_index=False)[ycol].sum().sort_values(ycol, ascending=False)
        st.plotly_chart(px.bar(gp, x=ccol, y=ycol, title=f"{ycol} by {ccol}"), use_container_width=True)

    # Profit share pie
    if ccol in fdf and "profit" in fdf:
        gp2 = fdf.groupby(ccol, as_index=False)["profit"].sum()
        st.plotly_chart(px.pie(gp2, names=ccol, values="profit", title="Profit share by category"),
                        use_container_width=True)

    # Download filtered
    st.download_button("⬇️ Download filtered data (CSV)",
                       fdf.to_csv(index=False).encode("utf-8"),
                       file_name="filtered_data.csv",
                       mime="text/csv")

# --------- Segmentation (KMeans) ---------
with tab_seg:
    st.subheader("Customer/Product Segmentation")
    if len(num_cols) < 2:
        st.info("Need at least 2 numeric columns (e.g., sales, profit, units).")
    else:
        feats = st.multiselect("Numeric features for clustering", options=num_cols, default=num_cols[:2])
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
                                           title="Cluster visualization"), use_container_width=True)
                out = fdf.loc[X.index].copy()
                out["_cluster"] = labels
                st.download_button("Download clusters", out.to_csv(index=False).encode(),
                                   file_name="clusters.csv", mime="text/csv")
            else:
                st.warning("Not enough rows after filtering to form clusters (rows ≥ k).")

# --------- Anomalies (IsolationForest) ---------
with tab_an:
    st.subheader("Anomaly Detection")
    if not num_cols:
        st.info("Need at least 1 numeric column.")
    else:
        feats2 = st.multiselect("Numeric features to score anomalies", options=num_cols,
                                default=num_cols[: min(3, len(num_cols))])
        cont = st.slider("Contamination (outlier %)", 0.01, 0.20, 0.05)
        if feats2:
            X2 = fdf[feats2].dropna().copy()
            if len(X2) > 10:
                X2s = StandardScaler().fit_transform(X2)
                iso = IsolationForest(contamination=cont, random_state=42).fit(X2s)
                scores = -iso.score_samples(X2s)  # higher = more anomalous
                out2 = fdf.loc[X2.index].copy()
                out2["_anomaly_score"] = scores
                st.dataframe(out2.sort_values("_anomaly_score", ascending=False).head(20),
                             use_container_width=True)
                st.plotly_chart(px.histogram(out2, x="_anomaly_score", nbins=30,
                                             title="Anomaly scores"), use_container_width=True)
                st.download_button("Download anomaly report", out2.to_csv(index=False).encode(),
                                   file_name="anomalies.csv", mime="text/csv")
            else:
                st.warning("Need > 10 rows after filtering for stable anomaly scoring.")

# --------- Forecast (Ridge regression) ---------
with tab_fc:
    st.subheader("Time-series Forecast (lightweight)")
    if not date_col:
        st.info("No date column detected. Use a dataset with a date/time column.")
    else:
        target_opts = [c for c in ["sales","profit","units"] if c in fdf.columns] or num_cols
        if not target_opts:
            st.info("Need a numeric target to forecast.")
        else:
            target = st.selectbox("Target to forecast", target_opts, index=0)
            freq = st.selectbox("Resample frequency", ["W","M"], index=0)
            horizon = st.slider("Forecast horizon", 4, 26, 12)

            ts = fdf[[date_col, target]].dropna()
            ts[date_col] = pd.to_datetime(ts[date_col])
            ts = ts.groupby(date_col, as_index=False)[target].sum()
            ts.index = pd.to_datetime(ts[date_col])
            # 🔥 FIX: keep only numeric columns when resampling
            ts = ts[[target]].resample(freq).sum().dropna()
            ts = ts.rename(columns={target: "y"})

            if len(ts) < 8:
                st.warning("Need at least 8 periods after resampling.")
            else:
                feat = ts.copy()
                feat["t"] = range(len(ts))
                feat["month"] = feat.index.month
                X = pd.get_dummies(feat[["t","month"]], columns=["month"], drop_first=True)
                y = feat["y"].values

                model = Ridge(alpha=1.0).fit(X, y)

                last_idx = feat.index[-1]
                future_idx = pd.date_range(last_idx, periods=horizon+1, freq=freq)[1:]
                fut = pd.DataFrame(index=future_idx)
                fut["t"] = range(len(feat), len(feat) + horizon)
                fut["month"] = fut.index.month
                Xf = pd.get_dummies(fut[["t","month"]], columns=["month"], drop_first=True)
                Xf = Xf.reindex(columns=X.columns, fill_value=0)

                yhat = model.predict(Xf)
                resid = y - model.predict(X)
                s = resid.std() if len(resid) > 1 else 0.0
                lo, hi = yhat - 1.96*s, yhat + 1.96*s

                fc = pd.DataFrame({"date": future_idx, "forecast": yhat, "lo": lo, "hi": hi})

                hist = ts.reset_index().rename(columns={date_col: "date", "y": target})
                fig = px.line(hist, x="date", y=target, title=f"{target}: history & forecast")
                fig.add_scatter(x=fc["date"], y=fc["forecast"], mode="lines", name="forecast")
                fig.add_scatter(x=fc["date"], y=fc["lo"], mode="lines", name="lower", line=dict(dash="dot"))
                fig.add_scatter(x=fc["date"], y=fc["hi"], mode="lines", name="upper", line=dict(dash="dot"))
                st.plotly_chart(fig, use_container_width=True)

                st.download_button("Download forecast CSV", fc.to_csv(index=False).encode(),
                                   file_name="forecast.csv", mime="text/csv")

# --------- AI Q&A (rules-based MVP) ---------
with tab_ai:
    st.subheader("Ask questions about your data")
    st.caption("Try: *Which category has highest sales?* or *highest profit?*")
    q = st.text_input("Your question")
    ans = ""
    if q:
        ql = q.lower()
        if cat_col and "highest" in ql and ("sale" in ql or "revenue" in ql):
            gp = fdf.groupby(cat_col, as_index=False)["sales"].sum() if "sales" in fdf else None
            if gp is not None and len(gp):
                top = gp.sort_values("sales", ascending=False).iloc[0]
                ans = f"Top {cat_col} by total sales: **{top[cat_col]}** ({top['sales']:,.0f})."
        elif cat_col and "highest" in ql and "profit" in ql and "profit" in fdf:
            gp = fdf.groupby(cat_col, as_index=False)["profit"].sum()
            if len(gp):
                top = gp.sort_values("profit", ascending=False).iloc[0]
                ans = f"Top {cat_col} by total profit: **{top[cat_col]}** ({top['profit']:,.0f})."
        else:
            ans = "I can answer questions like *Which category has highest sales/profit?*"

    if ans:
        st.success(ans)
