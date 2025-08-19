import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

st.set_page_config(page_title="BI + AI Dashboard", layout="wide")
st.title("📊 Interactive BI Dashboard + 🤖 AI Insights")

# --- Upload or built-in demo (no files needed) ---
f = st.file_uploader("Upload CSV (optional)", type=["csv"])
if f is not None:
   # Use demo data if no upload
df = pd.DataFrame({
    "date": pd.date_range("2024-01-01", periods=14, freq="D"),
    "category": ["Electronics","Fashion","Groceries","Electronics","Fashion","Groceries","Electronics",
                 "Fashion","Groceries","Electronics","Fashion","Groceries","Electronics","Fashion"],
    "sales":  [2000,1500,800,3000,2200,1200,2500,1800,1100,2700,2100,900,2600,1900],
    "profit": [ 300, 200,100, 500, 350, 150, 400, 260, 130, 420, 320,120, 410, 280],
})

else:
    st.info("📂 No file uploaded — using demo data.")
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=14, freq="D"),
        "category": ["Electronics","Fashion","Groceries","Electronics","Fashion","Groceries","Electronics",
                     "Fashion","Groceries","Electronics","Fashion","Groceries","Electronics","Fashion"],
        "sales":  [2000,1500,800,3000,2200,1200,2500,1800,1100,2700,2100,900,2600,1900],
        "profit": [ 300, 200,100, 500, 350, 150, 400, 260, 130, 420, 320,120, 410, 280],
    })

# Parse dates if present
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"])

# ---- Sidebar filters ----
st.sidebar.header("🔎 Filters")
if "category" in df.columns:
    picked = st.sidebar.multiselect(
        "Select categories", options=sorted(df["category"].astype(str).unique()),
        default=sorted(df["category"].astype(str).unique())
    )
    df = df[df["category"].astype(str).isin(picked)]

if "date" in df.columns:
    dmin, dmax = pd.to_datetime(df["date"].min()), pd.to_datetime(df["date"].max())
    start, end = st.sidebar.date_input("Select date range", [dmin.date(), dmax.date()])
    if isinstance(start, datetime) and isinstance(end, datetime):
        df = df[(df["date"] >= pd.to_datetime(start)) & (df["date"] <= pd.to_datetime(end))]

# ---- KPIs ----
st.subheader("📈 Key Metrics")
c1,c2,c3,c4,c5,c6 = st.columns(6)
c1.metric("Rows", f"{len(df):,}")
c2.metric("Columns", f"{df.shape[1]}")
if "sales" in df:
    c3.metric("Total Sales", f"{df['sales'].sum():,}")
    c5.metric("Avg Sales", f"{df['sales'].mean():.2f}")
if "profit" in df:
    c4.metric("Total Profit", f"{df['profit'].sum():,}")
    c6.metric("Avg Profit", f"{df['profit'].mean():.2f}")

# ---- Charts ----
if "date" in df and "sales" in df:
    st.subheader("📊 Sales Trend")
    st.plotly_chart(px.line(df, x="date", y="sales", color="category", markers=True),
                    use_container_width=True)

if "category" in df and "sales" in df:
    st.subheader("🛍️ Sales by Category")
    gp_sales = df.groupby("category", as_index=False)["sales"].sum().sort_values("sales", ascending=False)
    st.plotly_chart(px.bar(gp_sales, x="category", y="sales"), use_container_width=True)

if "category" in df and "profit" in df:
    st.subheader("💰 Profit by Category")
    gp_profit = df.groupby("category", as_index=False)["profit"].sum().sort_values("profit", ascending=False)
    st.plotly_chart(px.bar(gp_profit, x="category", y="profit"), use_container_width=True)

# ---- Download filtered data ----
st.subheader("⬇️ Download Data")
st.download_button("Download filtered CSV",
                   df.to_csv(index=False).encode("utf-8"),
                   file_name="filtered_data.csv",
                   mime="text/csv")

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
