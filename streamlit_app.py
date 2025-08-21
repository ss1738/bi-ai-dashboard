# AI BI Dashboard - Complete Production App
# Save as: streamlit_app.py
#
# Requirements.txt:
# streamlit>=1.28.0
# pandas>=2.0.0
# numpy>=1.24.0
# altair>=5.0.0
# plotly>=5.17.0
# scikit-learn>=1.3.0

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import base64
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

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
    "light_color": "#F9FAFB"
}

# Page Configuration
st.set_page_config(
    page_title="AI Revenue Recovery Platform",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ═══════════════════════════════════════════════════════════════════════════════
# 🎯 CSS STYLING - PRODUCTION READY
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
    
    /* Hide Streamlit branding */
    #MainMenu {{visibility: hidden;}}
    .stDeployButton {{display:none;}}
    footer {{visibility: hidden;}}
    .stApp > header {{visibility: hidden;}}
    
    /* Main app styling */
    .stApp {{
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }}
    
    .main .block-container {{
        padding: 2rem 1rem;
        max-width: 1200px;
    }}
    
    /* Hero Section */
    .hero {{
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
        padding: 4rem 2rem;
        border-radius: 20px;
        text-align: center;
        color: white;
        margin: 2rem 0;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        animation: fadeInUp 0.8s ease-out;
    }}
    
    .hero h1 {{
        font-size: 3.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        background: linear-gradient(45deg, #ffffff, #f0f9ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}
    
    .hero p {{
        font-size: 1.3rem;
        opacity: 0.95;
        margin-bottom: 2rem;
    }}
    
    /* Navigation */
    .nav-container {{
        background: white;
        padding: 1rem 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        margin: 2rem 0;
        display: flex;
        justify-content: center;
        flex-wrap: wrap;
        gap: 1rem;
    }}
    
    .nav-btn {{
        background: linear-gradient(135deg, var(--primary), var(--secondary));
        color: white;
        padding: 12px 24px;
        border: none;
        border-radius: 10px;
        cursor: pointer;
        font-weight: 600;
        transition: all 0.3s ease;
        text-decoration: none;
        display: inline-block;
    }}
    
    .nav-btn:hover {{
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.3);
    }}
    
    .nav-btn.active {{
        background: var(--dark);
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(31, 41, 55, 0.3);
    }}
    
    /* Cards */
    .metric-card {{
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 8px 30px rgba(0,0,0,0.08);
        border: 1px solid rgba(255,255,255,0.2);
        transition: all 0.3s ease;
        height: 100%;
    }}
    
    .metric-card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(0,0,0,0.12);
    }}
    
    .metric-value {{
        font-size: 2.5rem;
        font-weight: 700;
        color: var(--primary);
        margin: 0.5rem 0;
    }}
    
    .metric-label {{
        color: #6B7280;
        font-weight: 500;
        margin-bottom: 0.5rem;
    }}
    
    .metric-delta {{
        font-weight: 600;
        font-size: 0.9rem;
    }}
    
    .delta-positive {{ color: var(--success); }}
    .delta-negative {{ color: #EF4444; }}
    
    /* Success Messages */
    .success-box {{
        background: linear-gradient(135deg, var(--success), #059669);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        font-weight: 600;
    }}
    
    /* Feature Cards */
    .feature-card {{
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        text-align: center;
        transition: all 0.3s ease;
        border: 2px solid transparent;
    }}
    
    .feature-card:hover {{
        border-color: var(--primary);
        transform: translateY(-3px);
        box-shadow: 0 8px 30px rgba(99, 102, 241, 0.15);
    }}
    
    .feature-icon {{
        font-size: 3rem;
        margin-bottom: 1rem;
    }}
    
    /* Form Styling */
    .waitlist-form {{
        background: white;
        padding: 3rem 2rem;
        border-radius: 20px;
        box-shadow: 0 20px 60px rgba(0,0,0,0.1);
        max-width: 600px;
        margin: 2rem auto;
    }}
    
    .stTextInput > div > div > input {{
        border-radius: 10px;
        border: 2px solid #E5E7EB;
        padding: 12px 16px;
        font-size: 16px;
        transition: all 0.3s ease;
    }}
    
    .stTextInput > div > div > input:focus {{
        border-color: var(--primary);
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
    }}
    
    .stSelectbox > div > div > div {{
        border-radius: 10px;
        border: 2px solid #E5E7EB;
    }}
    
    /* Animations */
    @keyframes fadeInUp {{
        from {{
            opacity: 0;
            transform: translateY(30px);
        }}
        to {{
            opacity: 1;
            transform: translateY(0);
        }}
    }}
    
    @keyframes pulse {{
        0%, 100% {{ transform: scale(1); }}
        50% {{ transform: scale(1.05); }}
    }}
    
    .animate-pulse {{
        animation: pulse 2s infinite;
    }}
    
    /* Responsive Design */
    @media (max-width: 768px) {{
        .hero h1 {{ font-size: 2.5rem; }}
        .hero p {{ font-size: 1.1rem; }}
        .nav-container {{ flex-direction: column; }}
        .metric-card {{ margin-bottom: 1rem; }}
    }}
    
    /* Custom Plotly styling */
    .js-plotly-plot {{
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    }}
    </style>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 📊 DATA GENERATION - REALISTIC BUSINESS DATA
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def generate_business_data():
    """Generate realistic business data with revenue recovery opportunities"""
    np.random.seed(42)
    
    # Date range - last 2 years
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)
    dates = pd.date_range(start_date, end_date, freq='D')
    
    # Business segments
    segments = ['Enterprise', 'Mid-Market', 'SMB', 'Startup']
    channels = ['Direct Sales', 'Partner', 'Online', 'Retail']
    regions = ['North America', 'Europe', 'Asia Pacific', 'Latin America']
    products = ['Platform Pro', 'Analytics Suite', 'AI Insights', 'Basic Plan']
    
    data = []
    
    for date in dates:
        # Seasonal patterns
        month = date.month
        seasonal_multiplier = 1.2 if month in [11, 12] else (0.8 if month in [7, 8] else 1.0)
        
        # Weekly patterns
        weekday = date.weekday()
        weekly_multiplier = 0.7 if weekday >= 5 else 1.0
        
        for segment in segments:
            for channel in channels:
                for region in regions:
                    # Base revenue with business logic
                    base_revenue = {
                        'Enterprise': 50000,
                        'Mid-Market': 15000, 
                        'SMB': 5000,
                        'Startup': 1500
                    }[segment]
                    
                    # Channel multipliers
                    channel_mult = {
                        'Direct Sales': 1.3,
                        'Partner': 1.1,
                        'Online': 0.9,
                        'Retail': 0.8
                    }[channel]
                    
                    # Region multipliers
                    region_mult = {
                        'North America': 1.2,
                        'Europe': 1.0,
                        'Asia Pacific': 0.9,
                        'Latin America': 0.7
                    }[region]
                    
                    # Calculate final revenue with noise
                    revenue = (base_revenue * seasonal_multiplier * weekly_multiplier * 
                             channel_mult * region_mult * np.random.normal(1, 0.15))
                    revenue = max(0, revenue)
                    
                    # Calculate related metrics
                    customers = max(1, int(np.random.poisson(revenue / (base_revenue / 10))))
                    avg_deal_size = revenue / customers if customers > 0 else 0
                    churn_rate = np.random.uniform(0.02, 0.08)  # 2-8% monthly
                    
                    # Add some anomalies (revenue recovery opportunities)
                    if np.random.random() < 0.05:  # 5% chance of anomaly
                        revenue *= 0.3  # Significant revenue drop
                    
                    data.append({
                        'date': date,
                        'segment': segment,
                        'channel': channel,
                        'region': region,
                        'product': np.random.choice(products),
                        'revenue': revenue,
                        'customers': customers,
                        'avg_deal_size': avg_deal_size,
                        'churn_rate': churn_rate,
                        'cost_of_acquisition': revenue * np.random.uniform(0.15, 0.35),
                        'lifetime_value': revenue * np.random.uniform(2, 6)
                    })
    
    return pd.DataFrame(data)

# ═══════════════════════════════════════════════════════════════════════════════
# 🧠 AI/ML FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def perform_segmentation(df):
    """Perform customer segmentation analysis"""
    # Aggregate data by segment
    segment_data = df.groupby('segment').agg({
        'revenue': 'sum',
        'customers': 'sum',
        'avg_deal_size': 'mean',
        'churn_rate': 'mean',
        'lifetime_value': 'mean'
    }).reset_index()
    
    # Features for clustering
    features = ['revenue', 'customers', 'avg_deal_size', 'lifetime_value']
    X = segment_data[features]
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    segment_data['cluster'] = kmeans.fit_predict(X_scaled)
    
    # Add cluster labels
    cluster_labels = {0: 'High Value', 1: 'Growing', 2: 'Opportunity'}
    segment_data['cluster_name'] = segment_data['cluster'].map(cluster_labels)
    
    return segment_data

def detect_anomalies(df):
    """Detect revenue anomalies using Isolation Forest"""
    # Daily revenue aggregation
    daily_revenue = df.groupby('date')['revenue'].sum().reset_index()
    
    # Features for anomaly detection
    daily_revenue['day_of_week'] = daily_revenue['date'].dt.dayofweek
    daily_revenue['month'] = daily_revenue['date'].dt.month
    daily_revenue['revenue_lag1'] = daily_revenue['revenue'].shift(1)
    daily_revenue['revenue_lag7'] = daily_revenue['revenue'].shift(7)
    
    # Remove NaN values
    daily_revenue = daily_revenue.dropna()
    
    # Features for model
    features = ['revenue', 'day_of_week', 'month', 'revenue_lag1', 'revenue_lag7']
    X = daily_revenue[features]
    
    # Isolation Forest
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    daily_revenue['anomaly'] = iso_forest.fit_predict(X)
    daily_revenue['anomaly_score'] = iso_forest.score_samples(X)
    
    # Filter anomalies
    anomalies = daily_revenue[daily_revenue['anomaly'] == -1].copy()
    anomalies['potential_loss'] = anomalies['revenue'] * 0.3  # Estimated recovery potential
    
    return anomalies.sort_values('date', ascending=False)

def generate_forecast(df, days=30):
    """Generate revenue forecast using linear regression"""
    # Daily revenue aggregation
    daily_revenue = df.groupby('date')['revenue'].sum().reset_index()
    daily_revenue['day_num'] = (daily_revenue['date'] - daily_revenue['date'].min()).dt.days
    
    # Train model
    X = daily_revenue[['day_num']].values
    y = daily_revenue['revenue'].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Generate forecast
    last_day = daily_revenue['day_num'].max()
    future_days = np.arange(last_day + 1, last_day + days + 1).reshape(-1, 1)
    forecast = model.predict(future_days)
    
    # Create forecast dataframe
    future_dates = pd.date_range(daily_revenue['date'].max() + timedelta(days=1), periods=days)
    forecast_df = pd.DataFrame({
        'date': future_dates,
        'forecast': forecast,
        'type': 'forecast'
    })
    
    # Historical data
    historical_df = daily_revenue[['date', 'revenue']].copy()
    historical_df['forecast'] = historical_df['revenue']
    historical_df['type'] = 'historical'
    
    return pd.concat([historical_df, forecast_df])

# ═══════════════════════════════════════════════════════════════════════════════
# 🎯 NAVIGATION SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

def render_navigation():
    """Render professional navigation"""
    
    # Initialize session state
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'Home'
    
    pages = {
        'Home': '🏠',
        'Dashboard': '📊', 
        'Segmentation': '🎯',
        'Anomaly Detection': '🚨',
        'Forecasting': '📈',
        'Early Access': '🚀'
    }
    
    # Navigation HTML
    nav_html = '<div class="nav-container">'
    for page, icon in pages.items():
        active_class = 'active' if st.session_state.current_page == page else ''
        nav_html += f'''
        <button class="nav-btn {active_class}" onclick="setPage('{page}')">
            {icon} {page}
        </button>
        '''
    nav_html += '</div>'
    
    # JavaScript for navigation
    nav_html += '''
    <script>
    function setPage(page) {
        window.parent.postMessage({type: 'streamlit:setComponentValue', value: page}, '*');
    }
    </script>
    '''
    
    st.markdown(nav_html, unsafe_allow_html=True)
    
    # Page selection
    cols = st.columns(len(pages))
    for i, (page, icon) in enumerate(pages.items()):
        with cols[i]:
            if st.button(f"{icon} {page}", key=f"nav_{page}", use_container_width=True):
                st.session_state.current_page = page
                st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# 📱 PAGE COMPONENTS
# ═══════════════════════════════════════════════════════════════════════════════

def render_hero():
    """Render hero section"""
    st.markdown(f"""
    <div class="hero">
        <h1>💰 {BRAND['name']}</h1>
        <p>{BRAND['tagline']}</p>
        <div style="display: flex; justify-content: center; gap: 2rem; margin-top: 2rem;">
            <div style="text-align: center;">
                <div style="font-size: 2rem; font-weight: bold;">$500K+</div>
                <div style="opacity: 0.9;">Average Recovery</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 2rem; font-weight: bold;">87%</div>
                <div style="opacity: 0.9;">Success Rate</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 2rem; font-weight: bold;">30 Days</div>
                <div style="opacity: 0.9;">Average ROI Time</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_metrics_cards(df):
    """Render KPI metrics cards"""
    # Calculate metrics
    total_revenue = df['revenue'].sum()
    total_customers = df['customers'].sum()
    avg_deal_size = df['avg_deal_size'].mean()
    avg_churn = df['churn_rate'].mean()
    
    # Previous period comparison (mock data for demo)
    prev_revenue = total_revenue * 0.92  # 8% growth
    prev_customers = total_customers * 0.95  # 5% growth
    prev_deal_size = avg_deal_size * 0.98  # 2% growth
    prev_churn = avg_churn * 1.1  # 10% improvement (lower is better)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        delta = ((total_revenue - prev_revenue) / prev_revenue) * 100
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Revenue</div>
            <div class="metric-value">${total_revenue:,.0f}</div>
            <div class="metric-delta delta-positive">+{delta:.1f}% vs last period</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        delta = ((total_customers - prev_customers) / prev_customers) * 100
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Customers</div>
            <div class="metric-value">{total_customers:,.0f}</div>
            <div class="metric-delta delta-positive">+{delta:.1f}% vs last period</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        delta = ((avg_deal_size - prev_deal_size) / prev_deal_size) * 100
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Avg Deal Size</div>
            <div class="metric-value">${avg_deal_size:,.0f}</div>
            <div class="metric-delta delta-positive">+{delta:.1f}% vs last period</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        delta = ((prev_churn - avg_churn) / prev_churn) * 100  # Inverted for churn
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Churn Rate</div>
            <div class="metric-value">{avg_churn:.1%}</div>
            <div class="metric-delta delta-positive">-{delta:.1f}% vs last period</div>
        </div>
        """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 🏠 PAGE: HOME
# ═══════════════════════════════════════════════════════════════════════════════

def page_home():
    render_hero()
    
    st.markdown("## 🎯 Revenue Recovery Platform Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🤖</div>
            <h3>AI-Powered Analytics</h3>
            <p>Advanced machine learning algorithms identify revenue leakage patterns and optimization opportunities in real-time.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">⚡</div>
            <h3>Instant Insights</h3>
            <p>Get actionable insights in minutes, not months. Our platform processes millions of data points to surface critical revenue opportunities.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">💎</div>
            <h3>Proven ROI</h3>
            <p>Average clients recover $500K+ in lost revenue within 30 days. Join 1000+ companies maximizing their revenue potential.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Success Stories
    st.markdown("## 🏆 Success Stories")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="success-box">
            <h4>🚀 TechCorp Inc.</h4>
            <p>"Recovered $1.2M in lost revenue within 45 days using AI insights. The platform identified customer segments we were completely missing."</p>
            <p><strong>— Sarah Chen, Chief Revenue Officer</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="success-box">
            <h4>🎯 Global Dynamics</h4>
            <p>"Anomaly detection caught a $800K revenue leak in our enterprise segment. Without this platform, we would have lost millions."</p>
            <p><strong>— Michael Rodriguez, VP Sales</strong></p>
        </div>
        """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 📊 PAGE: DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

def page_dashboard(df):
    st.title("📊 Revenue Analytics Dashboard")
    st.markdown("### Real-time insights into your revenue performance")
    
    # KPI Cards
    render_metrics_cards(df)
    
    st.markdown("---")
    
    # Revenue Trend
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📈 Revenue Trend Analysis")
        
        # Daily revenue trend
        daily_revenue = df.groupby('date')['revenue'].sum().reset_index()
        
        fig = px.line(daily_revenue, x='date', y='revenue',
                     title="Daily Revenue Trend",
                     labels={'revenue': 'Revenue ($)', 'date': 'Date'})
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#374151'),
            title_font_size=16,
            height=400
        )
        fig.update_traces(line_color=BRAND['primary_color'], line_width=3)
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Revenue by Segment")
        
        segment_revenue = df.groupby('segment')['revenue'].sum().reset_index()
        segment_revenue = segment_revenue.sort_values('revenue', ascending=False)
        
        fig = px.pie(segment_revenue, values='revenue', names='segment',
                    title="Revenue Distribution",
                    color_discrete_sequence=px.colors.qualitative.Set3)
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#374151'),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Channel and Region Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Channel Performance")
        
        channel_data = df.groupby('channel').agg({
            'revenue': 'sum',
            'customers': 'sum'
        }).reset_index()
        
        fig = px.bar(channel_data, x='channel', y='revenue',
                    title="Revenue by Channel",
                    color='revenue',
                    color_continuous_scale='Viridis')
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#374151'),
            showlegend=False,
            height=350
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🌍 Regional Analysis")
        
        region_data = df.groupby('region').agg({
            'revenue': 'sum',
            'customers': 'sum'
        }).reset_index()
        
        fig = px.bar(region_data, x='region', y='revenue',
                    title="Revenue by Region",
                    color='revenue',
                    color_continuous_scale='Plasma')
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#374151'),
            showlegend=False,
            height=350
        )
        
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 🎯 PAGE: SEGMENTATION
# ═══════════════════════════════════════════════════════════════════════════════

def page_segmentation(df):
    st.title("🎯 Customer Segmentation Analysis")
    st.markdown("### AI-powered segmentation reveals hidden revenue opportunities")
    
    # Perform segmentation
    segment_data = perform_segmentation(df)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("💎 Segment Performance")
        
        for _, row in segment_data.iterrows():
            cluster_color = {
                'High Value': BRAND['success_color'],
                'Growing': BRAND['warning_color'], 
                'Opportunity': BRAND['secondary_color']
            }[row['cluster_name']]
            
            st.markdown(f"""
            <div class="metric-card" style="border-left: 4px solid {cluster_color};">
                <h4 style="color: {cluster_color};">{row['cluster_name']} - {row['segment']}</h4>
                <div class="metric-value" style="font-size: 1.5rem;">${row['revenue']:,.0f}</div>
                <div class="metric-label">Revenue</div>
                <p><strong>Customers:</strong> {row['customers']:,.0f}</p>
                <p><strong>Avg Deal Size:</strong> ${row['avg_deal_size']:,.0f}</p>
                <p><strong>LTV:</strong> ${row['lifetime_value']:,.0f}</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("📊 Segmentation Visualization")
        
        # Bubble chart
        fig = px.scatter(segment_data, 
                        x='avg_deal_size', 
                        y='lifetime_value',
                        size='revenue',
                        color='cluster_name',
                        hover_name='segment',
                        title="Segment Analysis: Deal Size vs Lifetime Value",
                        labels={
                            'avg_deal_size': 'Average Deal Size ($)',
                            'lifetime_value': 'Customer Lifetime Value ($)'
                        })
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#374151'),
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Revenue Opportunity Analysis
    st.markdown("---")
    st.subheader("💰 Revenue Recovery Opportunities")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        opportunity_segments = segment_data[segment_data['cluster_name'] == 'Opportunity']
        total_opportunity = opportunity_segments['revenue'].sum() * 0.3  # 30% uplift potential
        
        st.markdown(f"""
        <div class="metric-card" style="border: 2px solid {BRAND['warning_color']};">
            <div class="metric-label">Opportunity Segments</div>
            <div class="metric-value" style="color: {BRAND['warning_color']};">${total_opportunity:,.0f}</div>
            <div class="metric-delta">Potential Revenue Recovery</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        high_value_segments = segment_data[segment_data['cluster_name'] == 'High Value']
        retention_value = high_value_segments['revenue'].sum() * 0.05  # 5% retention improvement
        
        st.markdown(f"""
        <div class="metric-card" style="border: 2px solid {BRAND['success_color']};">
            <div class="metric-label">High Value Retention</div>
            <div class="metric-value" style="color: {BRAND['success_color']};">${retention_value:,.0f}</div>
            <div class="metric-delta">Additional Revenue Potential</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        growing_segments = segment_data[segment_data['cluster_name'] == 'Growing']
        growth_potential = growing_segments['revenue'].sum() * 0.4  # 40% growth potential
        
        st.markdown(f"""
        <div class="metric-card" style="border: 2px solid {BRAND['primary_color']};">
            <div class="metric-label">Growing Segments</div>
            <div class="metric-value" style="color: {BRAND['primary_color']};">${growth_potential:,.0f}</div>
            <div class="metric-delta">Acceleration Opportunity</div>
        </div>
        """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 🚨 PAGE: ANOMALY DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def page_anomaly_detection(df):
    st.title("🚨 AI Anomaly Detection")
    st.markdown("### Detect revenue leaks before they become costly problems")
    
    # Detect anomalies
    anomalies = detect_anomalies(df)
    
    if not anomalies.empty:
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        total_anomalies = len(anomalies)
        total_loss = anomalies['potential_loss'].sum()
        avg_loss_per_day = anomalies['potential_loss'].mean()
        recent_anomalies = len(anomalies[anomalies['date'] >= (datetime.now() - timedelta(days=7))])
        
        with col1:
            st.markdown(f"""
            <div class="metric-card" style="border: 2px solid #EF4444;">
                <div class="metric-label">Total Anomalies Detected</div>
                <div class="metric-value" style="color: #EF4444;">{total_anomalies}</div>
                <div class="metric-delta">Last 2 Years</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card" style="border: 2px solid #F59E0B;">
                <div class="metric-label">Potential Revenue Loss</div>
                <div class="metric-value" style="color: #F59E0B;">${total_loss:,.0f}</div>
                <div class="metric-delta">Recoverable Amount</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card" style="border: 2px solid #8B5CF6;">
                <div class="metric-label">Avg Daily Impact</div>
                <div class="metric-value" style="color: #8B5CF6;">${avg_loss_per_day:,.0f}</div>
                <div class="metric-delta">Per Anomaly</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card" style="border: 2px solid #10B981;">
                <div class="metric-label">Recent Anomalies</div>
                <div class="metric-value" style="color: #10B981;">{recent_anomalies}</div>
                <div class="metric-delta">Last 7 Days</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Anomaly Timeline
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📈 Anomaly Timeline")
            
            # Daily revenue with anomalies highlighted
            daily_revenue = df.groupby('date')['revenue'].sum().reset_index()
            
            fig = go.Figure()
            
            # Normal revenue
            fig.add_trace(go.Scatter(
                x=daily_revenue['date'],
                y=daily_revenue['revenue'],
                mode='lines',
                name='Daily Revenue',
                line=dict(color=BRAND['primary_color'], width=2)
            ))
            
            # Anomalies
            fig.add_trace(go.Scatter(
                x=anomalies['date'],
                y=anomalies['revenue'],
                mode='markers',
                name='Anomalies',
                marker=dict(color='red', size=8, symbol='x')
            ))
            
            fig.update_layout(
                title="Revenue Timeline with Anomaly Detection",
                xaxis_title="Date",
                yaxis_title="Revenue ($)",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#374151'),
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🔍 Recent Anomalies")
            
            recent_anomalies_df = anomalies.head(10)
            
            for _, anomaly in recent_anomalies_df.iterrows():
                severity = "HIGH" if anomaly['potential_loss'] > 10000 else "MEDIUM" if anomaly['potential_loss'] > 5000 else "LOW"
                color = "#EF4444" if severity == "HIGH" else "#F59E0B" if severity == "MEDIUM" else "#10B981"
                
                st.markdown(f"""
                <div style="background: white; padding: 1rem; border-radius: 8px; margin: 0.5rem 0; border-left: 4px solid {color};">
                    <div style="font-weight: bold; color: {color};">{severity} PRIORITY</div>
                    <div style="font-size: 0.9rem; color: #666;">Date: {anomaly['date'].strftime('%Y-%m-%d')}</div>
                    <div>Revenue: ${anomaly['revenue']:,.0f}</div>
                    <div>Potential Loss: ${anomaly['potential_loss']:,.0f}</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Detailed Anomaly Analysis
        st.markdown("---")
        st.subheader("📋 Detailed Anomaly Analysis")
        
        # Display anomalies table
        display_anomalies = anomalies[['date', 'revenue', 'potential_loss', 'anomaly_score']].copy()
        display_anomalies['date'] = display_anomalies['date'].dt.strftime('%Y-%m-%d')
        display_anomalies['revenue'] = display_anomalies['revenue'].apply(lambda x: f"${x:,.0f}")
        display_anomalies['potential_loss'] = display_anomalies['potential_loss'].apply(lambda x: f"${x:,.0f}")
        display_anomalies['anomaly_score'] = display_anomalies['anomaly_score'].round(3)
        
        st.dataframe(
            display_anomalies.rename(columns={
                'date': 'Date',
                'revenue': 'Revenue',
                'potential_loss': 'Potential Loss',
                'anomaly_score': 'Anomaly Score'
            }),
            use_container_width=True,
            height=300
        )
    
    else:
        st.info("No significant anomalies detected in the current dataset.")

# ═══════════════════════════════════════════════════════════════════════════════
# 📈 PAGE: FORECASTING
# ═══════════════════════════════════════════════════════════════════════════════

def page_forecasting(df):
    st.title("📈 AI Revenue Forecasting")
    st.markdown("### Predict future revenue trends with machine learning")
    
    # Generate forecast
    forecast_df = generate_forecast(df, days=90)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🔮 90-Day Revenue Forecast")
        
        # Create forecast visualization
        fig = go.Figure()
        
        # Historical data
        historical = forecast_df[forecast_df['type'] == 'historical']
        fig.add_trace(go.Scatter(
            x=historical['date'],
            y=historical['forecast'],
            mode='lines',
            name='Historical Revenue',
            line=dict(color=BRAND['primary_color'], width=3)
        ))
        
        # Forecast data
        forecast = forecast_df[forecast_df['type'] == 'forecast']
        fig.add_trace(go.Scatter(
            x=forecast['date'],
            y=forecast['forecast'],
            mode='lines',
            name='Forecast',
            line=dict(color=BRAND['secondary_color'], width=3, dash='dash')
        ))
        
        # Add confidence bands (simulated)
        upper_bound = forecast['forecast'] * 1.1
        lower_bound = forecast['forecast'] * 0.9
        
        fig.add_trace(go.Scatter(
            x=forecast['date'],
            y=upper_bound,
            fill=None,
            mode='lines',
            line_color='rgba(0,0,0,0)',
            showlegend=False,
            name='Upper Bound'
        ))
        
        fig.add_trace(go.Scatter(
            x=forecast['date'],
            y=lower_bound,
            fill='tonexty',
            mode='lines',
            line_color='rgba(0,0,0,0)',
            name='Confidence Interval',
            fillcolor='rgba(236, 72, 153, 0.2)'
        ))
        
        fig.update_layout(
            title="Revenue Forecast with Confidence Intervals",
            xaxis_title="Date",
            yaxis_title="Revenue ($)",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#374151'),
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Forecast Insights")
        
        # Calculate forecast metrics
        historical_avg = historical['forecast'].tail(30).mean()
        forecast_avg = forecast['forecast'].mean()
        growth_rate = ((forecast_avg - historical_avg) / historical_avg) * 100
        
        total_forecast_revenue = forecast['forecast'].sum()
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">90-Day Revenue Forecast</div>
            <div class="metric-value">${total_forecast_revenue:,.0f}</div>
            <div class="metric-delta delta-positive">Next 3 Months</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Predicted Growth Rate</div>
            <div class="metric-value">{growth_rate:+.1f}%</div>
            <div class="metric-delta">vs Historical Average</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Key insights
        st.markdown("### 🎯 Key Insights")
        
        if growth_rate > 5:
            st.markdown("""
            <div class="success-box">
                <strong>🚀 Strong Growth Predicted</strong><br>
                The model predicts strong revenue growth. Consider scaling operations to meet demand.
            </div>
            """, unsafe_allow_html=True)
        elif growth_rate < -5:
            st.markdown("""
            <div style="background: #FEE2E2; color: #991B1B; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
                <strong>⚠️ Declining Trend Detected</strong><br>
                Revenue decline predicted. Immediate action required to identify and address root causes.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background: #FEF3C7; color: #92400E; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
                <strong>📊 Stable Growth</strong><br>
                Revenue growth is stable. Look for opportunities to accelerate growth through optimization.
            </div>
            """, unsafe_allow_html=True)
    
    # Scenario Analysis
    st.markdown("---")
    st.subheader("🎲 Scenario Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 🐻 Conservative Scenario")
        conservative = total_forecast_revenue * 0.85
        st.metric("Revenue Forecast", f"${conservative:,.0f}", "-15% adjustment")
        
    with col2:
        st.markdown("#### 📈 Base Case")
        st.metric("Revenue Forecast", f"${total_forecast_revenue:,.0f}", "Current model prediction")
        
    with col3:
        st.markdown("#### 🚀 Optimistic Scenario")
        optimistic = total_forecast_revenue * 1.25
        st.metric("Revenue Forecast", f"${optimistic:,.0f}", "+25% adjustment")

# ═══════════════════════════════════════════════════════════════════════════════
# 🚀 PAGE: EARLY ACCESS WAITLIST
# ═══════════════════════════════════════════════════════════════════════════════

def page_early_access():
    st.title("🚀 Join the Early Access Program")
    st.markdown("### Be among the first to experience the future of revenue recovery")
    
    # Value proposition
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div class="waitlist-form">
            <h3 style="color: #6366F1; text-align: center; margin-bottom: 2rem;">🎯 Exclusive Benefits</h3>
            
            <div style="margin: 1.5rem 0;">
                <h4>💰 Free Revenue Assessment</h4>
                <p>Get a complimentary $25K analysis of your revenue optimization opportunities</p>
            </div>
            
            <div style="margin: 1.5rem 0;">
                <h4>⚡ Priority Access</h4>
                <p>Skip the waitlist and get immediate access to our platform when it launches</p>
            </div>
            
            <div style="margin: 1.5rem 0;">
                <h4>🎁 50% Launch Discount</h4>
                <p>Save thousands on your first year subscription - exclusive to early access members</p>
            </div>
            
            <div style="margin: 1.5rem 0;">
                <h4>🤝 Direct Founder Access</h4>
                <p>Monthly office hours with our founders to shape the product roadmap</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="waitlist-form">', unsafe_allow_html=True)
        st.markdown("#### 📝 Secure Your Spot")
        
        # Waitlist Form
        with st.form("waitlist_form", clear_on_submit=True):
            col1, col2 = st.columns(2)
            
            with col1:
                first_name = st.text_input("First Name *", placeholder="John")
            with col2:
                last_name = st.text_input("Last Name *", placeholder="Smith")
            
            email = st.text_input("Business Email *", placeholder="john.smith@company.com")
            company = st.text_input("Company Name *", placeholder="Acme Corporation")
            
            col1, col2 = st.columns(2)
            with col1:
                role = st.selectbox("Your Role *", [
                    "Select your role...",
                    "CEO/Founder", 
                    "Chief Revenue Officer",
                    "VP Sales",
                    "VP Marketing", 
                    "Head of Analytics",
                    "Director of Operations",
                    "Other"
                ])
            
            with col2:
                company_size = st.selectbox("Company Size *", [
                    "Select size...",
                    "Startup (1-10 employees)",
                    "Small (11-50 employees)", 
                    "Medium (51-200 employees)",
                    "Large (201-1000 employees)",
                    "Enterprise (1000+ employees)"
                ])
            
            annual_revenue = st.selectbox("Annual Revenue *", [
                "Select revenue range...",
                "Under $1M",
                "$1M - $10M",
                "$10M - $50M", 
                "$50M - $100M",
                "$100M+"
            ])
            
            pain_points = st.multiselect("Current Revenue Challenges (Select all that apply)", [
                "Revenue forecasting accuracy",
                "Customer churn/retention", 
                "Lead conversion optimization",
                "Pricing strategy",
                "Sales process efficiency",
                "Customer segmentation",
                "Revenue leakage detection",
                "Cross-sell/upsell opportunities"
            ])
            
            timeline = st.radio("When do you need a solution?", [
                "Immediately (within 30 days)",
                "Soon (1-3 months)",
                "Planning ahead (3-6 months)",
                "Exploring options (6+ months)"
            ])
            
            # Interest level
            st.markdown("**How interested are you in our solution?**")
            interest = st.slider("", 1, 10, 7, help="1 = Slightly interested, 10 = Extremely interested")
            
            # Additional comments
            comments = st.text_area("Additional Comments (Optional)", 
                                   placeholder="Tell us about your specific revenue challenges or questions...")
            
            # Consent checkboxes
            consent_updates = st.checkbox("I agree to receive product updates and early access notifications")
            consent_privacy = st.checkbox("I agree to the Privacy Policy and Terms of Service *")
            
            submitted = st.form_submit_button("🚀 Join Early Access Program", 
                                            type="primary", use_container_width=True)
            
            if submitted:
                # Validation
                errors = []
                if not first_name: errors.append("First name is required")
                if not last_name: errors.append("Last name is required")
                if not email or "@" not in email: errors.append("Valid email is required")
                if not company: errors.append("Company name is required")
                if role == "Select your role...": errors.append("Please select your role")
                if company_size == "Select size...": errors.append("Please select company size")
                if annual_revenue == "Select revenue range...": errors.append("Please select revenue range")
                if not consent_privacy: errors.append("You must agree to the Privacy Policy")
                
                if errors:
                    for error in errors:
                        st.error(error)
                else:
                    # Success message
                    st.markdown("""
                    <div class="success-box">
                        <h4>🎉 Welcome to Early Access!</h4>
                        <p>Thank you for joining our exclusive early access program. We'll be in touch within 24 hours with your free revenue assessment details.</p>
                        <p><strong>What's Next:</strong></p>
                        <ul>
                            <li>Check your email for confirmation</li>
                            <li>Our team will contact you within 24 hours</li>
                            <li>Schedule your free revenue assessment call</li>
                            <li>Get priority access when we launch</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Store submission (in real app, this would go to database/CRM)
                    if 'waitlist_submissions' not in st.session_state:
                        st.session_state.waitlist_submissions = []
                    
                    submission = {
                        'timestamp': datetime.now().isoformat(),
                        'name': f"{first_name} {last_name}",
                        'email': email,
                        'company': company,
                        'role': role,
                        'company_size': company_size,
                        'annual_revenue': annual_revenue,
                        'pain_points': pain_points,
                        'timeline': timeline,
                        'interest_level': interest,
                        'comments': comments,
                        'consent_updates': consent_updates
                    }
                    
                    st.session_state.waitlist_submissions.append(submission)
                    
                    # Show balloons for celebration
                    st.balloons()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Social proof
    st.markdown("---")
    st.markdown("### 🏆 Join 2,500+ Revenue Leaders Already Waiting")
    
    col1, col2, col3, col4 = st.columns(4)
    
    companies = [
        ("TechCorp", "500+ employees"),
        ("Global Dynamics", "1000+ employees"), 
        ("Revenue Solutions Inc", "200+ employees"),
        ("Growth Partners", "150+ employees")
    ]
    
    for i, (company, size) in enumerate(companies):
        with [col1, col2, col3, col4][i]:
            st.markdown(f"""
            <div style="text-align: center; padding: 1rem; background: white; border-radius: 10px; margin: 0.5rem 0;">
                <div style="font-weight: bold; color: #6366F1;">{company}</div>
                <div style="font-size: 0.9rem; color: #666;">{size}</div>
            </div>
            """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 🚀 MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # Load CSS
    load_css()
    
    # Generate data
    df = generate_business_data()
    
    # Navigation
    render_navigation()
    
    # Route to pages
    if st.session_state.current_page == 'Home':
        page_home()
    elif st.session_state.current_page == 'Dashboard':
        page_dashboard(df)
    elif st.session_state.current_page == 'Segmentation':
        page_segmentation(df)
    elif st.session_state.current_page == 'Anomaly Detection':
        page_anomaly_detection(df)
    elif st.session_state.current_page == 'Forecasting':
        page_forecasting(df)
    elif st.session_state.current_page == 'Early Access':
        page_early_access()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 2rem; color: #666; background: white; border-radius: 15px; margin-top: 2rem;">
        <p><strong>AI Revenue Recovery Platform</strong> • Transforming revenue optimization with artificial intelligence</p>
        <p style="font-size: 0.9rem;">© 2025 Revenue Recovery Inc. All rights reserved. • <a href="#" style="color: #6366F1;">Privacy Policy</a> • <a href="#" style="color: #6366F1;">Terms of Service</a></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
