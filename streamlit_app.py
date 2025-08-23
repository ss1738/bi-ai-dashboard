# streamlit_app.py

# --- Core Libraries ---
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import urllib.parse
import base64
import sqlite3
import io

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
    import xgboost as xgb
    XGBOOST_INSTALLED = True
except ImportError:
    XGBOOST_INSTALLED = False
    
try:
    import lightgbm as lgb
    LIGHTGBM_INSTALLED = True
except ImportError:
    LIGHTGBM_INSTALLED = False

# --- PDF Export ---
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
    elif domain == "Music Streaming":
        regions = ['Global']
        channels = ['Premium', 'Ad-Supported', 'Family Plan']
        products = ['Subscription', 'Merch', 'Concert Tickets']
        data.update({
            'region': np.random.choice(regions, size=1500),
            'channel': np.random.choice(channels, size=1500, p=[0.6, 0.3, 0.1]),
            'product': np.random.choice(products, size=1500),
            'revenue': np.random.uniform(5, 50, size=1500),
            'customers': np.random.randint(1, 5, size=1500)
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

def safe_multiselect(label, options, default, key, query_params):
    """Creates a multiselect box with defaults clamped to available options."""
    # Get defaults from query params if available
    qp_defaults = query_params.get(key, [])
    
    # Ensure defaults from query params are valid options
    valid_qp_defaults = [opt for opt in qp_defaults if opt in options]
    
    # If there are valid query param defaults, use them. Otherwise use the function's default.
    final_defaults = valid_qp_defaults if valid_qp_defaults else default
    
    # Clamp the final defaults to ensure they are in the options list
    clamped_defaults = [d for d in final_defaults if d in options]
    
    return st.multiselect(label, options, default=clamped_defaults, key=key)

def update_query_params():
    """Updates URL query parameters based on current widget states."""
    params = {}
    for key, value in st.session_state.items():
        if key not in ['query_params', 'df', 'uploaded_file'] and value:
            params[key] = value
    st.query_params.from_dict(params)

def initialize_db(conn):
    """Initializes the SQLite database and table."""
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sales_data (
            date TEXT,
            region TEXT,
            channel TEXT,
            product TEXT,
            revenue REAL,
            customers INTEGER,
            upload_timestamp TEXT,
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
    
    # Use INSERT OR IGNORE to prevent adding duplicate rows
    cursor.executemany("""
        INSERT OR IGNORE INTO sales_data 
        (date, region, channel, product, revenue, customers, upload_timestamp) 
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, tuples)
    
    conn.commit()
    return cursor.rowcount

def create_download_link(val, filename):
    """Creates a download link for a file."""
    b64 = base64.b64encode(val)
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{filename}">Download {filename}</a>'

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
    
    # Map daily anomalies back to original data
    anomalous_dates = ts_data[ts_data['anomaly']]['date']
    df['anomaly'] = df['date'].isin(anomalous_dates)
    return df

@st.cache_data
def get_customer_segments(df):
    """Segments customers using K-Means clustering."""
    if 'revenue' not in df.columns or 'customers' not in df.columns or df.shape[0] < 3:
        return df.assign(segment='N/A')
        
    # Aggregate data to get customer-level features
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
    
    # Label segments
    centers = scaler.inverse_transform(kmeans.cluster_centers_)
    labels = {}
    for i, center in enumerate(centers):
        if center[0] > centers[:, 0].mean() and center[1] > centers[:, 1].mean():
            labels[i] = 'High Value'
        elif center[0] < centers[:, 0].mean() and center[1] < centers[:, 1].mean():
            labels[i] = 'Low Value'
        else:
            labels[i] = 'Mid Value'
            
    cust_data['segment'] = cust_data['segment_id'].map(labels)
    
    # Merge segments back to original dataframe
    df = df.merge(cust_data[['customers', 'segment']], on='customers', how='left')
    return df

@st.cache_resource
def get_forecast(df, periods=90):
    """Generates a revenue forecast using available models."""
    results = {}
    ts_df = df.groupby('date')['revenue'].sum().reset_index()
    ts_df.columns = ['ds', 'y']
    
    # Prophet
    if PROPHET_INSTALLED:
        try:
            m = Prophet()
            m.fit(ts_df)
            future = m.make_future_dataframe(periods=periods)
            forecast = m.predict(future)
            results['Prophet'] = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        except Exception as e:
            st.warning(f"Prophet forecast failed: {e}")
    
    # Fallback to a simple moving average if no other models are available
    if not results:
        last_date = ts_df['ds'].max()
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=periods)
        
        # Simple moving average forecast
        window = 30
        moving_avg = ts_df['y'].rolling(window=window).mean().iloc[-1]
        if pd.isna(moving_avg):
            moving_avg = ts_df['y'].mean()
            
        forecast_values = [moving_avg] * periods
        
        forecast_df = pd.DataFrame({
            'ds': future_dates,
            'yhat': forecast_values,
            'yhat_lower': [v * 0.8 for v in forecast_values],
            'yhat_upper': [v * 1.2 for v in forecast_values],
        })
        results['Simple Moving Average'] = forecast_df
        
    return results

#==============================================================================
# UI RENDERING
#==============================================================================

def main():
    # --- Load initial query params ---
    query_params = st.query_params.to_dict()
    for k, v in query_params.items():
        if isinstance(v, list) and len(v) == 1:
            query_params[k] = v[0]
            
    # --- Sidebar ---
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Theme Toggle
        theme = query_params.get("theme", "light")
        if st.toggle("Dark Theme", value=(theme == "dark"), key="theme_toggle"):
            st.session_state.theme = "dark"
        else:
            st.session_state.theme = "light"
        
        # This is a hack to set the theme as st.set_page_config must be at the top
        # The real solution requires a page reload. We'll manage via URL params.
        if st.session_state.theme != theme:
            st.query_params.theme = st.session_state.theme
            st.rerun()

        # --- File Uploader and Sample Data ---
        uploaded_file = st.file_uploader("Upload your sales CSV", type="csv", key="uploaded_file")
        
        st.markdown("**Or use a sample dataset:**")
        sample_domain = st.selectbox("Business Domain", ["Sports Apparel", "Video Games", "Music Streaming"])
        
        if st.button("Load Sample Data"):
            st.session_state.df = generate_sample_data(sample_domain)
            st.query_params.clear() # Clear params when loading new data
            st.rerun()

        if 'df' not in st.session_state and uploaded_file:
            st.session_state.df = load_data(uploaded_file)
        
        if 'df' in st.session_state and st.session_state.df is not None:
            df = st.session_state.df
            
            st.header("📊 Filters")
            
            # --- Date Range Filter with safe defaults ---
            min_date, max_date = df['date'].min().date(), df['date'].max().date()
            
            try:
                start_d_qp = pd.to_datetime(query_params.get("start_date", min_date)).date()
            except (ValueError, TypeError):
                start_d_qp = min_date
            
            try:
                end_d_qp = pd.to_datetime(query_params.get("end_date", max_date)).date()
            except (ValueError, TypeError):
                end_d_qp = max_date
            
            # Clamp dates to be within the data's range
            start_d_clamped = max(min_date, start_d_qp)
            end_d_clamped = min(max_date, end_d_qp)
            
            date_range = st.date_input(
                "Date Range",
                value=(start_d_clamped, end_d_clamped),
                min_value=min_date,
                max_value=max_date,
                key='date_range'
            )
            start_date, end_date = date_range if len(date_range) == 2 else (min_date, max_date)
            st.session_state.start_date = start_date.strftime('%Y-%m-%d')
            st.session_state.end_date = end_date.strftime('%Y-%m-%d')

            # --- Multiselect Filters with safe defaults ---
            regions = sorted(df['region'].unique())
            selected_regions = safe_multiselect('Region', regions, regions, 'regions', query_params)

            channels = sorted(df['channel'].unique())
            selected_channels = safe_multiselect('Channel', channels, channels, 'channels', query_params)
            
            # --- Apply Filters ---
            filtered_df = df[
                (df['date'].dt.date >= start_date) &
                (df['date'].dt.date <= end_date) &
                (df['region'].isin(selected_regions)) &
                (df['channel'].isin(selected_channels))
            ]
            
            # --- Run ML Models on filtered data ---
            filtered_df = get_anomalies(filtered_df)
            filtered_df = get_customer_segments(filtered_df)

            # --- Update URL on change ---
            # This must be outside the filter creation to capture their state
            st.button("Apply Filters & Share", on_click=update_query_params, use_container_width=True, type="primary")

            # --- Database Persistence ---
            st.header("💾 Data Persistence")
            use_db = st.toggle("Save data to local DB", key="use_db")
            if use_db:
                conn = sqlite3.connect("sales_data.db")
                initialize_db(conn)
                if st.button("Sync to Database"):
                    rows_added = insert_data(conn, df)
                    st.success(f"Synced to database. {rows_added} new records added.")
                conn.close()

            # --- Debugging & Environment ---
            with st.expander("🛠️ Environment Debugger"):
                if st.button("Clear All Caches"):
                    st.cache_data.clear()
                    st.cache_resource.clear()
                    st.success("Caches cleared. Rerun to see changes.")
                    st.rerun()

                st.write("**Optional Libraries Status:**")
                st.info(f"Prophet: {'✅ Installed' if PROPHET_INSTALLED else '❌ Not Installed'}")
                st.info(f"XGBoost: {'✅ Installed' if XGBOOST_INSTALLED else '❌ Not Installed'}")
                st.info(f"LightGBM: {'✅ Installed' if LIGHTGBM_INSTALLED else '❌ Not Installed'}")
                st.info(f"FPDF (for PDF): {'✅ Installed' if FPDF_INSTALLED else '❌ Not Installed'}")

    # --- Main Dashboard Area ---
    if 'df' not in st.session_state or st.session_state.df is None:
        st.info("👋 Welcome to the AI Revenue Recovery Dashboard! Please upload a sales CSV or load a sample dataset to begin.")
        st.markdown("Your CSV should contain columns like `date`, `region`, `channel`, `product`, `revenue`, and `customers`.")
        return

    st.title("💡 AI Revenue Recovery Dashboard")
    
    # --- KPIs ---
    total_revenue = filtered_df['revenue'].sum()
    anomalies = filtered_df[filtered_df['anomaly']]
    recoverable_revenue = anomalies[anomalies['revenue'] < filtered_df['revenue'].mean()]['revenue'].sum()
    upsell_potential = filtered_df[filtered_df['segment'] == 'Mid Value']['revenue'].sum() * 0.15 # Assume 15% upsell

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Revenue", f"${total_revenue:,.2f}")
    col2.metric("Recoverable Revenue (Anomalies)", f"${recoverable_revenue:,.2f}", help="Revenue lost from negative anomalies.")
    col3.metric("Upsell Potential", f"${upsell_potential:,.2f}", help="Estimated revenue from upselling 'Mid Value' customers.")

    # --- Charts ---
    st.markdown("---")
    
    # Revenue Trend with Anomalies
    st.subheader("Revenue Trend & Anomalies")
    trend_df = filtered_df.groupby('date').agg({'revenue': 'sum', 'anomaly': 'max'}).reset_index()
    fig_trend = px.line(trend_df, x='date', y='revenue', title='Daily Revenue')
    
    anomaly_points = trend_df[trend_df['anomaly']]
    if not anomaly_points.empty:
        fig_trend.add_trace(go.Scatter(
            x=anomaly_points['date'], y=anomaly_points['revenue'],
            mode='markers', name='Anomaly',
            marker=dict(color='red', size=10, symbol='circle')
        ))
    st.plotly_chart(fig_trend, use_container_width=True)
    
    # Revenue Breakdown
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
    
    # Forecasting
    st.markdown("---")
    st.subheader("Revenue Forecast")
    if not filtered_df.empty:
        forecasts = get_forecast(filtered_df)
        
        model_choice = st.selectbox("Select Forecast Model", list(forecasts.keys()))
        
        forecast_df = forecasts[model_choice]
        
        fig_forecast = go.Figure()
        fig_forecast.add_trace(go.Scatter(x=filtered_df['date'], y=filtered_df['revenue'], mode='lines', name='Actual Revenue'))
        fig_forecast.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat'], mode='lines', name='Forecast', line=dict(color='orange')))
        fig_forecast.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat_upper'], fill='tonexty', mode='lines', line_color='rgba(255,165,0,0.2)', name='Upper Bound'))
        fig_forecast.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat_lower'], fill='tonexty', mode='lines', line_color='rgba(255,165,0,0.2)', name='Lower Bound'))
        
        st.plotly_chart(fig_forecast, use_container_width=True)
    
    # --- PDF Export & Waitlist ---
    st.markdown("---")
    c1, c2 = st.columns([1, 3])
    with c1:
        st.subheader("Actions")
        if FPDF_INSTALLED:
            if st.button("Export KPIs to PDF"):
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=16)
                pdf.cell(200, 10, txt="AI Revenue Recovery KPIs", ln=True, align='C')
                pdf.set_font("Arial", size=12)
                pdf.cell(200, 10, txt=f"Date: {datetime.now().strftime('%Y-%m-%d')}", ln=True)
                pdf.ln(10)
                pdf.cell(200, 10, txt=f"Total Revenue: ${total_revenue:,.2f}", ln=True)
                pdf.cell(200, 10, txt=f"Recoverable Revenue (Anomalies): ${recoverable_revenue:,.2f}", ln=True)
                pdf.cell(200, 10, txt=f"Upsell Potential: ${upsell_potential:,.2f}", ln=True)
                
                pdf_output = pdf.output(dest='S').encode('latin-1')
                st.download_button(
                    label="Download PDF",
                    data=pdf_output,
                    file_name=f"kpi_report_{datetime.now().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf"
                )
        else:
            st.warning("PDF export requires `fpdf2`. Please install it.")
    
    with c2:
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
    # --- Apply Theme based on URL param ---
    # This must be done before any other st call for the theme to be set on first load
    # It will cause a quick flicker on theme change as the page reloads.
    page_theme = st.query_params.get("theme", "light")
    if page_theme == "dark":
        st.config.set_option('theme.backgroundColor', '#0E1117')
        st.config.set_option('theme.base', 'dark')
    else:
        st.config.set_option('theme.backgroundColor', '#FFFFFF')
        st.config.set_option('theme.base', 'light')

    main()
