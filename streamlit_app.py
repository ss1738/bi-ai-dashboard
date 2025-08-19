import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="📊 BI + AI Dashboard", layout="wide")

st.title("📊 Interactive BI Dashboard + 🤖 AI Insights")

# --- File uploader or demo data ---
f = st.file_uploader("Upload CSV", type=["csv"])

if f:
    df = pd.read_csv(f)
else:
    st.info("📂 No file uploaded. Using demo data...")
    demo_data = {
        "date": pd.date_range("2024-01-01", periods=7, freq="D"),
        "category": ["Electronics", "Fashion", "Electronics", "Groceries", "Fashion", "Groceries", "Electronics"],
        "sales": [2000, 1500, 3000, 800, 2200, 1200, 2500],
        "profit": [300, 200, 500, 100, 350, 150, 400]
    }
    df = pd.DataFrame(demo_data)

df["date"] = pd.to_datetime(df["date"])

# --- Filters ---
st.sidebar.header("🔎 Filters")
categories = st.sidebar.multiselect(
    "Select categories", options=df["category"].unique(), default=df["category"].unique()
)
date_range = st.sidebar.date_input(
    "Select date range",
    [df["date"].min(), df["date"].max()]
)

df_filtered = df[
    (df["category"].isin(categories)) &
    (df["date"].between(pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])))
]

# --- KPIs ---
st.subheader("📈 Key Metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Rows", len(df_filtered))
col2.metric("Columns", df_filtered.shape[1])
col3.metric("Total Sales", f"{df_filtered['sales'].sum():,.0f}")
col4.metric("Avg Sales", f"{df_filtered['sales'].mean():,.2f}")

# --- Charts ---
st.subheader("📊 Sales Trend")
fig = px.line(df_filtered, x="date", y="sales", color="category", markers=True)
st.plotly_chart(fig, use_container_width=True)

st.subheader("🛍️ Sales by Category")
fig2 = px.bar(df_filtered, x="category", y="sales", color="category", barmode="group")
st.plotly_chart(fig2, use_container_width=True)

# --- Download filtered data ---
st.subheader("⬇️ Download Data")
csv = df_filtered.to_csv(index=False).encode("utf-8")
st.download_button("Download CSV", csv, "filtered_data.csv", "text/csv")
