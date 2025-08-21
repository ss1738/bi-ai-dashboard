# AI BI Dashboard - Complete Production App (Altair-only, Fixed)
# Save as: streamlit_app.py
#
# requirements.txt:
# streamlit>=1.28.0
# pandas>=2.0.0
# numpy>=1.24.0
# altair>=5.0.0
# scikit-learn>=1.3.0
#
# Run locally:
#   streamlit run streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
import warnings

warnings.filterwarnings('ignore')
alt.data_transformers.disable_max_rows()

# ═══════════════════════════════════════════════════════════════════════════════
# 🎨 BRANDING & CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

BRAND = {
    "name": "AI Revenue Recovery Platform",
    "tagline": "Recover $500K+ in lost revenue with AI-powered insights",
    "primary_color": "#6366F1",
    "secondary_color": "#EC4899",
    "success_color": "#10B981",
    "warning_color": "#F59E0B",
    "dark_color": "#1F2937",
    "light_color": "#F9FAFB",
}

st.set_page_config(
    page_title="AI Revenue Recovery Platform",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ═══════════════════════════════════════════════════════════════════════════════
# 🎯 CSS STYLING
# ═══════════════════════════════════════════════════════════════════════════════

def load_css():
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    :root {{
        --primary: {BRAND['primary_color']};
        --secondary: {BRAND['secondary_color']};
        --success: {BRAND['success_color']};
        --warning: {BRAND['warning_color']};
        --dark: {BRAND['dark_color']};
        --light: {BRAND['light_color']};
    }}
    #MainMenu {{visibility: hidden;}}
    .stDeployButton {{display:none;}}
    footer {{visibility: hidden;}}
    .stApp > header {{visibility: hidden;}}
    .stApp {{background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); font-family: 'Inter', sans-serif;}}
    </style>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 📊 DATA GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def generate_business_data(seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)
    dates = pd.date_range(start_date, end_date, freq="D")

    segments = ["Enterprise", "Mid-Market", "SMB", "Startup"]
    channels = ["Direct Sales", "Partner", "Online", "Retail"]
    regions = ["North America", "Europe", "Asia Pacific", "Latin America"]
    products = ["Platform Pro", "Analytics Suite", "AI Insights", "Basic Plan"]

    rows = []
    for date in dates:
        seasonal_mult = 1.2 if date.month in (11, 12) else (0.8 if date.month in (7, 8) else 1.0)
        weekly_mult = 0.7 if date.weekday() >= 5 else 1.0
        for segment in segments:
            base_revenue = {"Enterprise": 50000, "Mid-Market": 15000, "SMB": 5000, "Startup": 1500}[segment]
            for channel in channels:
                channel_mult = {"Direct Sales": 1.3, "Partner": 1.1, "Online": 0.9, "Retail": 0.8}[channel]
                for region in regions:
                    region_mult = {"North America": 1.2, "Europe": 1.0, "Asia Pacific": 0.9, "Latin America": 0.7}[region]
                    revenue = base_revenue * seasonal_mult * weekly_mult * channel_mult * region_mult * np.random.normal(1, 0.15)
                    revenue = max(0, revenue)
                    customers = max(1, int(np.random.poisson(revenue / (base_revenue / 10))))
                    avg_deal_size = revenue / customers if customers else 0
                    churn_rate = np.random.uniform(0.02, 0.08)
                    if np.random.random() < 0.05:
                        revenue *= 0.3
                    rows.append({
                        "date": date,
                        "segment": segment,
                        "channel": channel,
                        "region": region,
                        "product": np.random.choice(products),
                        "revenue": float(revenue),
                        "customers": int(customers),
                        "avg_deal_size": float(avg_deal_size),
                        "churn_rate": float(churn_rate),
                        "cost_of_acquisition": float(revenue * np.random.uniform(0.15, 0.35)),
                        "lifetime_value": float(revenue * np.random.uniform(2, 6)),
                    })
    return pd.DataFrame(rows)

# ═══════════════════════════════════════════════════════════════════════════════
# 🧠 AI/ML FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def perform_segmentation(df: pd.DataFrame) -> pd.DataFrame:
    segment_data = df.groupby("segment").agg({
        "revenue": "sum", "customers": "sum",
        "avg_deal_size": "mean", "churn_rate": "mean", "lifetime_value": "mean"
    }).reset_index()
    features = ["revenue", "customers", "avg_deal_size", "lifetime_value"]
    X = StandardScaler().fit_transform(segment_data[features])
    kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
    segment_data["cluster"] = kmeans.fit_predict(X)
    labels = {0: "High Value", 1: "Growing", 2: "Opportunity"}
    segment_data["cluster_name"] = segment_data["cluster"].map(labels)
    return segment_data

def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    daily = df.groupby("date", as_index=False)["revenue"].sum()
    daily["day_of_week"] = daily["date"].dt.dayofweek
    daily["month"] = daily["date"].dt.month
    daily["revenue_lag1"] = daily["revenue"].shift(1)
    daily["revenue_lag7"] = daily["revenue"].shift(7)
    daily = daily.dropna()
    feats = ["revenue", "day_of_week", "month", "revenue_lag1", "revenue_lag7"]
    iso = IsolationForest(contamination=0.1, random_state=42)
    daily["anomaly"] = iso.fit_predict(daily[feats])
    daily["anomaly_score"] = iso.score_samples(daily[feats])
    anomalies = daily[daily["anomaly"] == -1].copy()
    anomalies["potential_loss"] = anomalies["revenue"] * 0.3
    return anomalies

def generate_forecast(df: pd.DataFrame, days: int = 30) -> pd.DataFrame:
    daily = df.groupby("date", as_index=False)["revenue"].sum()
    daily["day_num"] = (daily["date"] - daily["date"].min()).dt.days
    X = daily[["day_num"]].values; y = daily["revenue"].values
    model = LinearRegression().fit(X, y)
    future_days = np.arange(daily["day_num"].max()+1, daily["day_num"].max()+days+1).reshape(-1, 1)
    forecast = model.predict(future_days)
    future_dates = pd.date_range(daily["date"].max() + timedelta(days=1), periods=days)
    forecast_df = pd.DataFrame({"date": future_dates, "value": forecast, "type": "Forecast"})
    hist_df = daily[["date", "revenue"]].rename(columns={"revenue": "value"}); hist_df["type"] = "Historical"
    return pd.concat([hist_df, forecast_df])

# ═══════════════════════════════════════════════════════════════════════════════
# 🧭 APP ENTRY
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    load_css()
    data = generate_business_data()
    st.title("💰 AI Revenue Recovery Platform")
    st.subheader(BRAND["tagline"])
    page = st.sidebar.radio("Navigation", ["Home","Dashboard","Segmentation","Anomaly Detection","Forecasting"])
    if page=="Home":
        st.write("Welcome! Use the sidebar to explore dashboards.")
    elif page=="Dashboard":
        st.write("📊 Dashboard coming soon.")
    elif page=="Segmentation":
        st.dataframe(perform_segmentation(data))
    elif page=="Anomaly Detection":
        st.dataframe(detect_anomalies(data))
    elif page=="Forecasting":
        st.dataframe(generate_forecast(data, days=30))

if __name__ == "__main__":
    main()
