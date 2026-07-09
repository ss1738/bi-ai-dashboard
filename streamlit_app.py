# --- Core Libraries ---
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import urllib.parse
import base64
import sqlite3
import io
import os

# --- AI insights (LLM) ---
from ai_insights import generate_insights, ask_data

# --- Plotting ---
import plotly.express as px
import plotly.graph_objects as go

# --- Machine Learning (with optional imports) ---
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Attempt to import optional heavy libraries
try:
    from prophet import Prophet
    PROPHET_INSTALLED = True
except ImportError:
    PROPHET_INSTALLED = False

try:
    from fpdf import FPDF
    FPDF_INSTALLED = True
except ImportError:
    FPDF_INSTALLED = False

#==============================================================================
# PAGE CONFIGURATION
#==============================================================================
st.set_page_config(
    page_title="AI Revenue Recovery Dashboard",
    page_icon="💡",
    layout="wide",
    initial_sidebar_state="expanded"
)

#==============================================================================
# HELPER FUNCTIONS
#==============================================================================

def generate_sample_data(domain="Sports Apparel"):
    """Generates a realistic sample DataFrame based on a business domain."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)
    dates = pd.to_datetime(pd.date_range(start=start_date, end=end_date, freq='D'))
    
    data = {'date': np.random.choice(dates, size=1500)}
    
    if domain == "Sports Apparel":
        regions = ['North America', 'Europe', 'Asia', 'South America']
        channels = ['Online', 'Retail', 'Outlet']
        products = ['Running Shoes', 'Yoga Pants', 'Team Jerseys', 'Fitness Trackers']
        data.update({
            'region': np.random.choice(regions, size=1500, p=[0.4, 0.3, 0.2, 0.1]),
            'channel': np.random.choice(channels, size=1500, p=[0.6, 0.3, 0.1]),
            'product': np.random.choice(products, size=1500),
            'revenue': np.random.uniform(50, 800, size=1500),
            'customers': np.random.randint(1, 50, size=1500)
        })
    elif domain == "Video Games":
        regions = ['NA', 'EU', 'APAC', 'LATAM']
        channels = ['Steam', 'PlayStation Store', 'Xbox Store', 'Direct']
        products = ['Action RPG', 'Strategy', 'Indie Puzzle', 'Subscription']
        data.update({
            'region': np.random.choice(regions, size=1500, p=[0.5, 0.3, 0.15, 0.05]),
            'channel': np.random.choice(channels, size=1500, p=[0.4, 0.25, 0.25, 0.1]),
            'product': np.random.choice(products, size=1500),
            'revenue': np.random.uniform(10, 200, size=1500),
            'customers': np.random.randint(10, 1000, size=1500)
        })
    
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    # Introduce some anomalies
    anomaly_indices = df.sample(frac=0.03).index
    df.loc[anomaly_indices, 'revenue'] *= np.random.choice([0.1, 0.2, 3, 5])
    return df

@st.cache_data
def load_data(uploaded_file):
    """Loads data from an uploaded CSV file."""
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            df['date'] = pd.to_datetime(df['date'])
            return df
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return None
    return None

def initialize_db(conn):
    """Initializes the SQLite database and table."""
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sales_data (
            date TEXT, region TEXT, channel TEXT, product TEXT,
            revenue REAL, customers INTEGER, upload_timestamp TEXT,
            UNIQUE(date, region, channel, product, revenue, customers)
        )
    """)
    conn.commit()

def insert_data(conn, df):
    """Inserts DataFrame data into the SQLite database, avoiding duplicates."""
    cursor = conn.cursor()
    df_to_insert = df.copy()
    df_to_insert['upload_timestamp'] = datetime.now().isoformat()
    tuples = [tuple(x) for x in df_to_insert.to_numpy()]
    cursor.executemany("""
        INSERT OR IGNORE INTO sales_data 
        (date, region, channel, product, revenue, customers, upload_timestamp) 
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, tuples)
    conn.commit()
    return cursor.rowcount

#==============================================================================
# CACHED ML & DATA PROCESSING FUNCTIONS
#==============================================================================

@st.cache_data
def get_anomalies(df, contamination=0.05):
    """Detects anomalies in revenue using Isolation Forest."""
    if 'revenue' not in df.columns or df.shape[0] < 2:
        return df.assign(anomaly=False)
        
    ts_data = df[['date', 'revenue']].set_index('date').resample('D').sum().reset_index()
    features = ts_data[['revenue']]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    model = IsolationForest(contamination=contamination, random_state=42)
    ts_data['anomaly_score'] = model.fit_predict(scaled_features)
    ts_data['anomaly'] = ts_data['anomaly_score'] == -1
    
    anomalous_dates = ts_data[ts_data['anomaly']]['date']
    # original rows carry a time-of-day; resampled dates are at midnight, match on the calendar day
    df['anomaly'] = df['date'].dt.normalize().isin(anomalous_dates)
    return df

@st.cache_data
def get_customer_segments(df):
    """Segments customers using K-Means clustering."""
    if 'revenue' not in df.columns or 'customers' not in df.columns or df.shape[0] < 3:
        return df.assign(segment='N/A')
        
    cust_data = df.groupby('customers').agg(
        total_revenue=('revenue', 'sum'),
        purchase_frequency=('date', 'count')
    ).reset_index()

    if cust_data.shape[0] < 3:
        return df.assign(segment='N/A')

    features = cust_data[['total_revenue', 'purchase_frequency']]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
    cust_data['segment_id'] = kmeans.fit_predict(scaled_features)
    
    centers = scaler.inverse_transform(kmeans.cluster_centers_)
    labels = {}
    high_value_idx = np.argmax(centers[:, 0] + centers[:, 1])
    low_value_idx = np.argmin(centers[:, 0] + centers[:, 1])
    
    for i in range(3):
        if i == high_value_idx: labels[i] = 'High Value'
        elif i == low_value_idx: labels[i] = 'Low Value'
        else: labels[i] = 'Mid Value'
            
    cust_data['segment'] = cust_data['segment_id'].map(labels)
    df = df.merge(cust_data[['customers', 'segment']], on='customers', how='left')
    return df

@st.cache_resource
def get_forecast(df, periods=90):
    """Generates a revenue forecast using available models."""
    results = {}
    ts_df = df.groupby('date')['revenue'].sum().reset_index()
    ts_df.columns = ['ds', 'y']
    
    if PROPHET_INSTALLED:
        try:
            m = Prophet()
            m.fit(ts_df)
            future = m.make_future_dataframe(periods=periods)
            forecast = m.predict(future)
            results['Prophet'] = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        except Exception as e:
            st.warning(f"Prophet forecast failed: {e}")
    
    if not results:
        last_date = ts_df['ds'].max()
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=periods)
        window = 30
        moving_avg = ts_df['y'].rolling(window=window).mean().iloc[-1]
        if pd.isna(moving_avg): moving_avg = ts_df['y'].mean()
        forecast_values = [moving_avg] * periods
        forecast_df = pd.DataFrame({
            'ds': future_dates, 'yhat': forecast_values,
            'yhat_lower': [v * 0.8 for v in forecast_values],
            'yhat_upper': [v * 1.2 for v in forecast_values],
        })
        results['Simple Moving Average'] = forecast_df
        
    return results

#==============================================================================
# UI RENDERING
#==============================================================================

def main():
    # auto-load a sample dataset for demos / screenshots via ?demo=1
    if st.query_params.get("demo") == "1" and "df" not in st.session_state:
        st.session_state.df = generate_sample_data("Sports Apparel")

    # --- Sidebar ---
    with st.sidebar:
        st.header("⚙️ Configuration")

        # --- AI engine (LLM) ---
        st.subheader("🤖 AI Engine")
        ai_provider = st.selectbox("LLM provider", ["Groq", "OpenAI", "xAI", "None (no key)"], key="ai_provider")
        ai_key = ""
        if ai_provider in ("Groq", "OpenAI", "xAI"):
            _envk = {"Groq": "GROQ_API_KEY", "OpenAI": "OPENAI_API_KEY", "xAI": "XAI_API_KEY"}[ai_provider]
            _default = os.environ.get(_envk, "")
            if not _default:
                try:
                    _default = st.secrets.get(_envk, "")
                except Exception:
                    _default = ""
            ai_key = st.text_input(f"{ai_provider} API key", type="password", value=_default, key="ai_key")
            st.caption("Groq (free): console.groq.com  |  xAI: console.x.ai")
        st.session_state["_ai_provider"] = ai_provider if ai_provider in ("Groq", "OpenAI", "xAI") else "None"
        st.session_state["_ai_key"] = ai_key
        st.markdown("---")

        uploaded_file = st.file_uploader("Upload your sales CSV", type="csv")
        st.markdown("**Or use a sample dataset:**")
        sample_domain = st.selectbox("Business Domain", ["Sports Apparel", "Video Games"])
        
        if st.button("Load Sample Data"):
            st.session_state.df = generate_sample_data(sample_domain)
            st.rerun()

        if 'df' not in st.session_state and uploaded_file:
            st.session_state.df = load_data(uploaded_file)
        
        if 'df' in st.session_state and st.session_state.df is not None:
            df = st.session_state.df
            st.header("📊 Filters")
            min_date, max_date = df['date'].min().date(), df['date'].max().date()
            date_range = st.date_input("Date Range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
            start_date, end_date = date_range if len(date_range) == 2 else (min_date, max_date)
            regions = sorted(df['region'].unique())
            selected_regions = st.multiselect('Region', regions, default=regions)
            channels = sorted(df['channel'].unique())
            selected_channels = st.multiselect('Channel', channels, default=channels)
            
            filtered_df = df[
                (df['date'].dt.date >= start_date) & (df['date'].dt.date <= end_date) &
                (df['region'].isin(selected_regions)) & (df['channel'].isin(selected_channels))
            ]
            
            filtered_df = get_anomalies(filtered_df)
            filtered_df = get_customer_segments(filtered_df)

            st.header("💾 Data Persistence")
            if st.toggle("Save data to local DB"):
                conn = sqlite3.connect("sales_data.db")
                initialize_db(conn)
                if st.button("Sync to Database"):
                    rows_added = insert_data(conn, df)
                    st.success(f"Synced to database. {rows_added} new records added.")
                conn.close()

            with st.expander("🛠️ Environment Debugger"):
                if st.button("Clear All Caches"):
                    st.cache_data.clear()
                    st.cache_resource.clear()
                    st.rerun()
                st.info(f"Prophet: {'✅ Installed' if PROPHET_INSTALLED else '❌ Not Installed'}")

    # --- Main Dashboard Area ---
    if 'df' not in st.session_state or st.session_state.df is None:
        st.info("👋 Welcome! Please upload a sales CSV or load a sample dataset to begin.")
        return

    st.title("💡 AI Revenue Recovery Dashboard")
    
    total_revenue = filtered_df['revenue'].sum()
    anomalies = filtered_df[filtered_df['anomaly']]
    recoverable_revenue = anomalies[anomalies['revenue'] < filtered_df['revenue'].mean()]['revenue'].sum()
    
    upsell_potential = 0.0
    if 'segment' in filtered_df.columns:
        upsell_potential = filtered_df[filtered_df['segment'] == 'Mid Value']['revenue'].sum() * 0.15

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Revenue", f"${total_revenue:,.2f}")
    col2.metric("Recoverable Revenue (Anomalies)", f"${recoverable_revenue:,.2f}")
    col3.metric("Upsell Potential", f"${upsell_potential:,.2f}")

    st.markdown("---")
    st.subheader("Revenue Trend & Anomalies")
    trend_df = filtered_df.groupby('date').agg({'revenue': 'sum', 'anomaly': 'max'}).reset_index()
    fig_trend = px.line(trend_df, x='date', y='revenue', title='Daily Revenue')
    anomaly_points = trend_df[trend_df['anomaly']]
    if not anomaly_points.empty:
        fig_trend.add_trace(go.Scatter(x=anomaly_points['date'], y=anomaly_points['revenue'], mode='markers', name='Anomaly', marker=dict(color='red', size=10)))
    st.plotly_chart(fig_trend, use_container_width=True)
    
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Revenue by Channel")
        channel_rev = filtered_df.groupby('channel')['revenue'].sum().reset_index()
        fig_channel = px.pie(channel_rev, names='channel', values='revenue', hole=0.3)
        st.plotly_chart(fig_channel, use_container_width=True)
    with c2:
        st.subheader("Revenue by Region")
        region_rev = filtered_df.groupby('region')['revenue'].sum().reset_index()
        fig_region = px.bar(region_rev, x='region', y='revenue', color='region')
        st.plotly_chart(fig_region, use_container_width=True)
    
    st.markdown("---")
    st.subheader("Revenue Forecast")
    if not filtered_df.empty:
        forecasts = get_forecast(filtered_df)
        model_choice = st.selectbox("Select Forecast Model", list(forecasts.keys()))
        forecast_df = forecasts[model_choice]
        fig_forecast = go.Figure()
        _actual = filtered_df.groupby('date')['revenue'].sum().sort_index().reset_index()
        fig_forecast.add_trace(go.Scatter(x=_actual['date'], y=_actual['revenue'], mode='lines', name='Actual Revenue'))
        fig_forecast.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat'], mode='lines', name='Forecast', line=dict(color='orange')))
        fig_forecast.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat_upper'], fill='tonexty', mode='lines', line_color='rgba(255,165,0,0.2)', name='Upper Bound'))
        fig_forecast.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat_lower'], fill='tonexty', mode='lines', line_color='rgba(255,165,0,0.2)', name='Lower Bound'))
        st.plotly_chart(fig_forecast, use_container_width=True)

    # --- CORRECTED ANOMALY DRILL-DOWN FEATURE ---
    st.markdown("---")
    st.subheader("Anomaly Investigation")
    if st.checkbox("Show Anomaly Details", key="show_anomaly_details"):
        anomaly_data = filtered_df[filtered_df['anomaly'] == True]
        if not anomaly_data.empty:
            st.write("Data points for all days flagged as anomalous within the selected filters:")
            st.dataframe(anomaly_data[['date', 'region', 'channel', 'revenue', 'customers']])
        else:
            st.info("No anomalies found within the selected date range and filters.")

    # --- AI Insights (real LLM) ---
    st.markdown("---")
    st.subheader("🤖 AI Insights")
    _prov = st.session_state.get("_ai_provider", "None")
    _key = st.session_state.get("_ai_key", "")
    st.caption("An LLM reads the filtered data above and writes specific, grounded recommendations.")
    if st.button("Generate AI insights", type="primary"):
        with st.spinner("Analysing the data..."):
            st.session_state["_insights"] = generate_insights(filtered_df, _prov, _key)
    if st.session_state.get("_insights"):
        st.markdown(st.session_state["_insights"])

    # --- Ask Your Data (conversational analytics) ---
    st.markdown("---")
    st.subheader("💬 Ask Your Data")
    st.caption("Ask questions in plain English, answers are grounded in your current data.")
    if "_chat" not in st.session_state:
        st.session_state["_chat"] = []
    for _role, _msg in st.session_state["_chat"]:
        with st.chat_message(_role):
            st.markdown(_msg)
    _q = st.chat_input("e.g. Which region is underperforming, and by how much?")
    if _q:
        st.session_state["_chat"].append(("user", _q))
        with st.chat_message("user"):
            st.markdown(_q)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                _a = ask_data(_q, filtered_df, _prov, _key)
            st.markdown(_a)
        st.session_state["_chat"].append(("assistant", _a))

    # --- Actions and Waitlist Form ---
    st.markdown("---")
    with st.form("waitlist_form"):
        st.subheader("Join the Waitlist for Advanced Features")
        email = st.text_input("Enter your email")
        submitted = st.form_submit_button("Join Now")
        if submitted and email:
            st.success(f"Thank you! {email} has been added to our waitlist.")

#==============================================================================
# SCRIPT EXECUTION
#==============================================================================
if __name__ == "__main__":
    main()