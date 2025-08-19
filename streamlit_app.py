import streamlit as st
import pandas as pd
import plotly.express as px
import io

st.set_page_config(page_title="📊 AI BI Dashboard", layout="wide")

st.title("📊 Interactive BI Dashboard + 🤖 AI Insights")

# --- File Upload or Demo Data ---
uploaded_file = st.file_uploader("Upload CSV", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    st.info("📂 No file uploaded. Using demo data...")
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=10, freq="D"),
        "category": ["Electronics","Fashion","Groceries","Electronics","Fashion",
                     "Groceries","Electronics","Fashion","Groceries","Electronics"],
        "sales": [1200, 900, 600, 1500, 1100, 800, 1700, 950, 720, 1400],
        "profit": [200, 150, 80, 300, 220, 120, 330, 180, 100, 260],
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

# --- KPIs ---
st.subheader("📈 Key Metrics")
kpi1, kpi2, kpi3 = st.columns(3)

with kpi1:
    st.metric("Rows", len(df))
with kpi2:
    st.metric("Total Sales", f"{df['sales'].sum():,.0f}" if "sales" in df else "N/A")
with kpi3:
    st.metric("Avg Sales", f"{df['sales'].mean():,.2f}" if "sales" in df else "N/A")

# --- Charts ---
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

# --- AI Insights Placeholder ---
st.subheader("🤖 AI Insights")
st.write("This is where AI-driven insights will appear (coming soon).")

# --- Data Download ---
st.subheader("⬇️ Download Data")
csv = df.to_csv(index=False).encode("utf-8")
st.download_button("Download CSV", csv, "data.csv", "text/csv", key="download_csv")
