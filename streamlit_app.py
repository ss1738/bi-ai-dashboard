import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

st.set_page_config(page_title="BI + AI Dashboard", layout="wide")

# Load data (demo if no file uploaded)
f = st.file_uploader("Upload CSV", type=["csv"])
if f:
    df = pd.read_csv(f)
else:
    df = pd.read_csv("data/sample_sales.csv")
    st.info("📂 No file uploaded. Using demo data...")

# Convert date if available
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"])

# Sidebar Filters
st.sidebar.header("🔎 Filters")
if "category" in df.columns:
    categories = st.sidebar.multiselect(
        "Select categories", options=df["category"].unique(), default=df["category"].unique()
    )
    df = df[df["category"].isin(categories)]

if "date" in df.columns:
    min_date, max_date = df["date"].min(), df["date"].max()
    start, end = st.sidebar.date_input("Select date range", [min_date, max_date])
    if isinstance(start, datetime) and isinstance(end, datetime):
        df = df[(df["date"] >= pd.to_datetime(start)) & (df["date"] <= pd.to_datetime(end))]

# KPIs
st.title("📊 Interactive BI Dashboard + 🤖 AI Insights")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Rows", len(df))
col2.metric("Columns", len(df.columns))
col3.metric("Total Sales", f"{df['sales'].sum():,}")
col4.metric("Total Profit", f"{df['profit'].sum():,}")

col5, col6 = st.columns(2)
col5.metric("Avg Sales", f"{df['sales'].mean():.2f}")
col6.metric("Avg Profit", f"{df['profit'].mean():.2f}")

# Charts
st.subheader("📈 Sales Trend")
if "date" in df.columns:
    fig = px.line(df, x="date", y="sales", color="category", title="Sales over Time")
    st.plotly_chart(fig, use_container_width=True)

st.subheader("🛍️ Sales by Category")
fig2 = px.bar(df, x="category", y="sales", color="category", title="Sales by Category")
st.plotly_chart(fig2, use_container_width=True)

st.subheader("💰 Profit by Category")
fig3 = px.bar(df, x="category", y="profit", color="category", title="Profit by Category")
st.plotly_chart(fig3, use_container_width=True)

# Download Data
st.download_button(
    "⬇️ Download Data", df.to_csv(index=False), file_name="filtered_data.csv", mime="text/csv"
)
