import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.arima.model import ARIMA

# --- Page Config ---
st.set_page_config(page_title="📊 AI BI Dashboard", layout="wide")

st.title("📊 Interactive BI Dashboard + 🤖 AI Insights")

# --- File Upload or Demo Data ---
uploaded_file = st.file_uploader("📂 Upload your CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")
else:
    st.info("📂 No file uploaded. Using demo data...")
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=20, freq="D"),
        "category": ["Electronics","Fashion","Groceries","Electronics","Fashion",
                     "Groceries","Electronics","Fashion","Groceries","Electronics"]*2,
        "sales": [1200, 900, 600, 1500, 1100, 800, 1700, 950, 720, 1400,
                  1300, 920, 640, 1600, 1120, 850, 1750, 980, 750, 1450],
        "profit": [200, 150, 80, 300, 220, 120, 330, 180, 100, 260,
                   210, 160, 90, 320, 230, 130, 350, 190, 110, 270],
    })

# --- Filters ---
st.sidebar.header("🔎 Filters")
cat_col = "category" if "category" in df.columns else None
time_col = "date" if "date" in df.columns else None

if cat_col:
    categories = st.sidebar.multiselect(
        "Select categories", options=df[cat_col].unique(),
        default=df[cat_col].unique(), key="filter_cat"
    )
    df = df[df[cat_col].isin(categories)]

if time_col:
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    min_d, max_d = df[time_col].min(), df[time_col].max()
    dr = st.sidebar.date_input("Select date range", [min_d, max_d], key="filter_date")
    if len(dr) == 2:
        df = df[(df[time_col] >= pd.to_datetime(dr[0])) & (df[time_col] <= pd.to_datetime(dr[1]))]

# --- Tabs ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Dashboard", "📌 Segmentation", "⚠️ Anomalies", "🔮 Forecast", "🤖 AI Insights"
])

# --- Dashboard ---
with tab1:
    st.subheader("📈 Key Metrics")
    kpi1, kpi2, kpi3 = st.columns(3)
    with kpi1:
        st.metric("Rows", len(df))
    with kpi2:
        st.metric("Total Sales", f"{df['sales'].sum():,.0f}" if "sales" in df else "N/A")
    with kpi3:
        st.metric("Avg Sales", f"{df['sales'].mean():,.2f}" if "sales" in df else "N/A")

    st.subheader("📊 Charts")
    if time_col and "sales" in df:
        fig_ts = px.line(df.sort_values(time_col), x=time_col, y="sales",
                         color=cat_col if cat_col else None,
                         markers=True, title="Sales Over Time")
        st.plotly_chart(fig_ts, use_container_width=True, key="line_chart")

    if cat_col and "sales" in df:
        gp = df.groupby(cat_col, as_index=False)["sales"].sum()
        fig_bar = px.bar(gp, x=cat_col, y="sales", title="Sales by Category")
        st.plotly_chart(fig_bar, use_container_width=True, key="bar_chart")

    if cat_col and "profit" in df:
        gp2 = df.groupby(cat_col, as_index=False)["profit"].sum()
        fig_pie = px.pie(gp2, names=cat_col, values="profit", title="Profit Share by Category")
        st.plotly_chart(fig_pie, use_container_width=True, key="pie_chart")

# --- Segmentation ---
with tab2:
    st.subheader("📌 Customer Segmentation (KMeans)")
    if "sales" in df and "profit" in df:
        try:
            X = df[["sales", "profit"]]
            kmeans = KMeans(n_clusters=3, random_state=42).fit(X)
            df["segment"] = kmeans.labels_
            fig_seg = px.scatter(df, x="sales", y="profit", color="segment", title="Customer Segments")
            st.plotly_chart(fig_seg, use_container_width=True, key="seg_chart")
        except Exception as e:
            st.error(f"Segmentation failed: {e}")
    else:
        st.warning("Need 'sales' and 'profit' columns for segmentation.")

# --- Anomalies ---
with tab3:
    st.subheader("⚠️ Anomaly Detection (IsolationForest)")
    if "sales" in df:
        try:
            model = IsolationForest(contamination=0.1, random_state=42)
            df["anomaly"] = model.fit_predict(df[["sales"]])
            anomalies = df[df["anomaly"] == -1]
            fig_anom = px.scatter(df, x=time_col, y="sales", color="anomaly",
                                  title="Anomalies in Sales (red = anomaly)")
            st.plotly_chart(fig_anom, use_container_width=True, key="anom_chart")
            st.write("Detected anomalies:", anomalies)
        except Exception as e:
            st.error(f"Anomaly detection failed: {e}")
    else:
        st.warning("Need 'sales' column for anomaly detection.")

# --- Forecast ---
with tab4:
    st.subheader("🔮 Sales Forecast (ARIMA)")
    if time_col and "sales" in df:
        try:
            ts = df.set_index(time_col)["sales"].resample("D").sum()
            ts = ts.asfreq("D").fillna(method="ffill")
            model = ARIMA(ts, order=(1, 1, 1))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=7)
            fc_df = pd.DataFrame({"date": forecast.index, "forecast": forecast.values})
            fig_fc = px.line(ts, x=ts.index, y=ts.values, title="7-Day Sales Forecast")
            fig_fc.add_scatter(x=fc_df["date"], y=fc_df["forecast"], mode="lines+markers", name="Forecast")
            st.plotly_chart(fig_fc, use_container_width=True, key="forecast_chart")
        except Exception as e:
            st.error(f"Forecasting failed: {e}")
    else:
        st.warning("Need 'date' and 'sales' columns for forecasting.")

# --- AI Insights ---
with tab5:
    st.subheader("🤖 AI Insights (Rule-based)")
    if "sales" in df and cat_col:
        top_cat = df.groupby(cat_col)["sales"].sum().idxmax()
        avg_sales = df["sales"].mean()
        weekend_sales = df[df[time_col].dt.dayofweek >= 5]["sales"].mean() if time_col else None

        st.write(f"📌 **Top Category:** {top_cat} drives the highest sales.")
        st.write(f"📌 **Average Sales:** {avg_sales:,.2f}")
        if weekend_sales:
            st.write(f"📌 **Weekend Sales Trend:** {weekend_sales:,.2f} (avg on weekends)")
    else:
        st.info("Insights will appear once you have `sales` and `category` data.")

# --- Download ---
st.subheader("⬇️ Download Data")
csv = df.to_csv(index=False).encode("utf-8")
st.download_button("Download CSV", csv, "data.csv", "text/csv", key="download_csv")
