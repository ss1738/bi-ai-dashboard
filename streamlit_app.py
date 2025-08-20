import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.arima.model import ARIMA

# --- Page config ---
st.set_page_config(page_title="📊 AI BI Dashboard", layout="wide")
st.markdown("<a id='top'></a>", unsafe_allow_html=True)
st.title("📊 Interactive BI Dashboard + 🤖 AI Insights")

# --- File upload or demo ---
uploaded_file = st.file_uploader("Upload CSV", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    st.info("📂 No file uploaded. Using demo data…")
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=20, freq="D"),
        "category": ["Electronics","Fashion","Groceries","Electronics","Fashion",
                     "Groceries","Electronics","Fashion","Groceries","Electronics"]*2,
        "sales": [1200, 900, 600, 1500, 1100, 800, 1700, 950, 720, 1400,
                  1300, 920, 640, 1600, 1120, 850, 1750, 980, 750, 1450],
        "profit": [200, 150, 80, 300, 220, 120, 330, 180, 100, 260,
                   210, 160, 90, 320, 230, 130, 350, 190, 110, 270],
    })

# --- Sidebar filters + quick link to Download section ---
st.sidebar.header("🔎 Filters")
cat_col = "category" if "category" in df.columns else None
time_col = "date" if "date" in df.columns else None

if cat_col:
    categories = st.sidebar.multiselect(
        "Select categories", options=df[cat_col].unique(),
        default=list(df[cat_col].unique())
    )
    df = df[df[cat_col].isin(categories)]

if time_col:
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    if df[time_col].notna().any():
        dmin, dmax = df[time_col].min(), df[time_col].max()
        dr = st.sidebar.date_input("Select date range", [dmin, dmax])
        if isinstance(dr, (list, tuple)) and len(dr) == 2:
            df = df[(df[time_col] >= pd.to_datetime(dr[0])) & (df[time_col] <= pd.to_datetime(dr[1]))]

st.sidebar.markdown("---")
st.sidebar.markdown("[⬇️ Go to Download](#download-data)")

# --- Tabs ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📊 Dashboard", "📌 Segmentation", "⚠️ Anomalies", "🔮 Forecast", "🤖 AI Insights"]
)

# === Dashboard ===
with tab1:
    if df.empty:
        st.error("No data after filters.")
    else:
        st.subheader("📈 Key Metrics")
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", len(df))
        c2.metric("Total Sales", f"{df['sales'].sum():,.0f}" if "sales" in df else "—")
        c3.metric("Avg Sales", f"{df['sales'].mean():,.2f}" if "sales" in df else "—")

        st.subheader("📊 Charts")
        if time_col and "sales" in df and df[time_col].notna().any():
            fig_ts = px.line(df.sort_values(time_col), x=time_col, y="sales",
                             color=cat_col if cat_col else None,
                             markers=True, title="Sales Over Time")
            st.plotly_chart(fig_ts, use_container_width=True)
        if cat_col and "sales" in df:
            gp = df.groupby(cat_col, as_index=False)["sales"].sum().sort_values("sales", ascending=False)
            st.plotly_chart(px.bar(gp, x=cat_col, y="sales", title="Sales by Category"), use_container_width=True)
        if cat_col and "profit" in df:
            gp2 = df.groupby(cat_col, as_index=False)["profit"].sum()
            st.plotly_chart(px.pie(gp2, names=cat_col, values="profit", title="Profit Share by Category"),
                            use_container_width=True)

# === Segmentation ===
with tab2:
    st.subheader("📌 Customer Segmentation (KMeans)")
    if all(c in df.columns for c in ["sales", "profit"]) and len(df) >= 3:
        try:
            X = df[["sales", "profit"]].dropna()
            if len(X) >= 3:
                kmeans = KMeans(n_clusters=3, random_state=42).fit(X)
                tmp = df.loc[X.index].copy()
                tmp["segment"] = kmeans.labels_.astype(str)
                st.plotly_chart(px.scatter(tmp, x="sales", y="profit", color="segment", title="Customer Segments"),
                                use_container_width=True)
            else:
                st.info("Not enough rows after dropping NA to cluster.")
        except Exception as e:
            st.error(f"Segmentation failed: {e}")
    else:
        st.info("Need 'sales' and 'profit' columns with at least 3 rows.")

# === Anomalies ===
with tab3:
    st.subheader("⚠️ Anomaly Detection (IsolationForest)")
    if "sales" in df and len(df) > 10:
        try:
            X2 = df[["sales"]].dropna()
            if len(X2) > 10:
                model = IsolationForest(contamination=0.1, random_state=42).fit(X2)
                df_loc = df.loc[X2.index].copy()
                df_loc["anomaly"] = model.predict(X2)
                st.plotly_chart(
                    px.scatter(df_loc, x=(time_col if time_col else df_loc.index), y="sales", color="anomaly",
                               title="Anomalies in Sales (−1 = anomaly)"),
                    use_container_width=True
                )
                st.dataframe(df_loc[df_loc["anomaly"] == -1], use_container_width=True)
            else:
                st.info("Need > 10 usable rows for anomaly detection.")
        except Exception as e:
            st.error(f"Anomaly detection failed: {e}")
    else:
        st.info("Need 'sales' column and > 10 rows.")

# === Forecast ===
with tab4:
    st.subheader("🔮 Sales Forecast (ARIMA)")
    if time_col and "sales" in df and df[time_col].notna().any():
        try:
            ts = df.set_index(time_col)["sales"].dropna()
            if len(ts) >= 8:
                ts = ts.resample("D").sum().asfreq("D").fillna(method="ffill")
                model = ARIMA(ts, order=(1, 1, 1))
                fit = model.fit()
                fc = fit.forecast(steps=7)
                hist = pd.DataFrame({"date": ts.index, "value": ts.values})
                fc_df = pd.DataFrame({"date": fc.index, "forecast": fc.values})
                fig = px.line(hist, x="date", y="value", title="7-Day Sales Forecast")
                fig.add_scatter(x=fc_df["date"], y=fc_df["forecast"], mode="lines+markers", name="Forecast")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Need at least 8 periods to forecast.")
        except Exception as e:
            st.error(f"Forecasting failed: {e}")
    else:
        st.info("Need a valid date column and 'sales' to forecast.")

# === AI Insights (simple rule-based) ===
with tab5:
    st.subheader("🤖 AI Insights (MVP)")
    bullets = []
    if "sales" in df and len(df):
        bullets.append(f"Total sales: **{df['sales'].sum():,.0f}**; average per row: **{df['sales'].mean():,.2f}**.")
    if cat_col and "sales" in df and len(df):
        gp = df.groupby(cat_col, as_index=False)["sales"].sum().sort_values("sales", ascending=False)
        if len(gp):
            top = gp.iloc[0]
            share = 100 * top["sales"] / max(df["sales"].sum(), 1)
            bullets.append(f"Top {cat_col}: **{top[cat_col]}** ({top['sales']:,.0f}, {share:.1f}% of total).")
    if time_col and "sales" in df and df[time_col].notna().any():
        by_week = df.set_index(time_col)["sales"].resample("W").sum()
        if len(by_week) >= 2:
            delta = (by_week.iloc[-1] - by_week.iloc[-2]) / max(by_week.iloc[-2], 1)
            bullets.append(f"Last week vs prior: **{delta*100:+.1f}%**.")
    if "profit" in df and "sales" in df and df["sales"].sum() > 0:
        margin = 100 * df["profit"].sum() / df["sales"].sum()
        bullets.append(f"Overall margin: **{margin:.1f}%**.")
    if not bullets:
        bullets.append("Provide at least a numeric value column (e.g., sales) for insights.")
    for b in bullets:
        st.markdown(f"- {b}")

# --- Download section with anchor ---
st.markdown("<a id='download-data'></a>", unsafe_allow_html=True)
st.subheader("⬇️ Download Data")
if len(df):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, "data.csv", "text/csv")
else:
    st.warning("No data to download.")

st.markdown("[🔝 Back to Top](#top)")
