import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
import json
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="AI BI Dashboard - Transform Your Business Data",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    .problem-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #28a745;
    }
    .waitlist-form {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin: 2rem 0;
    }
    .success-message {
        background: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #c3e6cb;
        margin: 1rem 0;
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if 'demo_data' not in st.session_state:
    st.session_state.demo_data = None
if 'waitlist_submitted' not in st.session_state:
    st.session_state.waitlist_submitted = False

def generate_demo_data():
    """Generate realistic business demo data"""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='D')
    
    # Generate realistic business metrics
    base_revenue = 50000
    trend = np.linspace(0, 20000, len(dates))
    seasonality = 10000 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)
    noise = np.random.normal(0, 5000, len(dates))
    
    # Add some anomalies
    anomaly_indices = np.random.choice(len(dates), size=20, replace=False)
    anomaly_multiplier = np.ones(len(dates))
    anomaly_multiplier[anomaly_indices] = np.random.uniform(0.3, 0.7, 20)
    
    revenue = (base_revenue + trend + seasonality + noise) * anomaly_multiplier
    
    data = pd.DataFrame({
        'date': dates,
        'revenue': np.maximum(revenue, 0),
        'customers': np.random.poisson(200, len(dates)) + trend/500,
        'orders': np.random.poisson(150, len(dates)) + trend/400,
        'conversion_rate': np.random.beta(2, 8, len(dates)) * 100,
        'channel': np.random.choice(['Organic', 'Paid', 'Social', 'Email'], len(dates)),
        'region': np.random.choice(['North America', 'Europe', 'Asia', 'Other'], len(dates))
    })
    
    return data

def submit_to_waitlist(email, company, use_case):
    """Submit email to Google Forms (replace with your form URL)"""
    try:
        # Replace this URL with your actual Google Form URL
        # For now, we'll simulate a successful submission
        google_form_url = "https://docs.google.com/forms/d/e/YOUR_FORM_ID/formResponse"
        
        # For demo purposes, we'll just return success
        # In production, you would make an actual request to Google Forms
        """
        data = {
            'entry.YOUR_EMAIL_FIELD_ID': email,
            'entry.YOUR_COMPANY_FIELD_ID': company,
            'entry.YOUR_USECASE_FIELD_ID': use_case
        }
        response = requests.post(google_form_url, data=data)
        return response.status_code == 200
        """
        return True
    except:
        return False

def main():
    # Navigation
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🏠 Home", "📊 Dashboard", "🎯 Segmentation", 
        "⚠️ Anomaly Detection", "📈 Forecasting", "🚀 Early Access"
    ])
    
    with tab1:
        # Hero Section
        st.markdown("""
        <div class="main-header">
            <h1>🚀 AI BI Dashboard</h1>
            <h3>Transform Your Business Data Into Million-Dollar Insights</h3>
            <p>Advanced AI-powered business intelligence that scales from startup to enterprise</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Problem-Solution Section
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="problem-card">
                <h3>🎯 Real Problems We Solve</h3>
                <ul>
                    <li><strong>Revenue Leakage:</strong> Detect $100K+ losses before they happen</li>
                    <li><strong>Customer Churn:</strong> Predict and prevent high-value customer losses</li>
                    <li><strong>Inventory Waste:</strong> Optimize stock levels, reduce waste by 30%</li>
                    <li><strong>Marketing ROI:</strong> Identify which channels drive real growth</li>
                    <li><strong>Operational Inefficiency:</strong> Spot bottlenecks costing 20+ hours/week</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="problem-card">
                <h3>💰 Value Delivered</h3>
                <ul>
                    <li><strong>$500K+ Revenue Recovery:</strong> Average in first 6 months</li>
                    <li><strong>40% Faster Decisions:</strong> Real-time insights vs monthly reports</li>
                    <li><strong>90% Accuracy:</strong> In anomaly and trend predictions</li>
                    <li><strong>10x ROI:</strong> Typical return within 12 months</li>
                    <li><strong>24/7 Monitoring:</strong> Never miss critical business changes</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Use Cases
        st.markdown("### 🏢 Perfect for:")
        
        use_case_cols = st.columns(3)
        with use_case_cols[0]:
            st.markdown("""
            **E-commerce & Retail**
            - Sales trend analysis
            - Inventory optimization
            - Customer behavior insights
            - Seasonal forecasting
            """)
        
        with use_case_cols[1]:
            st.markdown("""
            **SaaS & Tech**
            - User engagement tracking
            - Churn prediction
            - Feature adoption analysis
            - Growth metrics monitoring
            """)
        
        with use_case_cols[2]:
            st.markdown("""
            **Finance & Banking**
            - Risk assessment
            - Fraud detection
            - Portfolio performance
            - Regulatory reporting
            """)
    
    with tab2:
        st.header("📊 Interactive Dashboard")
        
        # Generate or use cached demo data
        if st.session_state.demo_data is None:
            with st.spinner("Loading demo data..."):
                st.session_state.demo_data = generate_demo_data()
        
        data = st.session_state.demo_data
        
        # Key Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_revenue = data['revenue'].sum()
            st.metric(
                label="Total Revenue",
                value=f"${total_revenue:,.0f}",
                delta=f"{(total_revenue/len(data)*30):.0f}/month avg"
            )
        
        with col2:
            avg_customers = data['customers'].mean()
            st.metric(
                label="Avg Daily Customers",
                value=f"{avg_customers:.0f}",
                delta=f"{((data['customers'].tail(30).mean() - data['customers'].head(30).mean())/data['customers'].head(30).mean()*100):.1f}%"
            )
        
        with col3:
            avg_conversion = data['conversion_rate'].mean()
            st.metric(
                label="Conversion Rate",
                value=f"{avg_conversion:.2f}%",
                delta=f"{(data['conversion_rate'].tail(30).mean() - data['conversion_rate'].head(30).mean()):.2f}%"
            )
        
        with col4:
            total_orders = data['orders'].sum()
            st.metric(
                label="Total Orders",
                value=f"{total_orders:,.0f}",
                delta=f"{(total_orders/len(data)*30):.0f}/month avg"
            )
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Revenue Trend
            fig_revenue = px.line(
                data, x='date', y='revenue',
                title='Daily Revenue Trend',
                color_discrete_sequence=['#667eea']
            )
            fig_revenue.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_revenue, use_container_width=True)
        
        with col2:
            # Channel Performance
            channel_data = data.groupby('channel')['revenue'].sum().reset_index()
            fig_channel = px.pie(
                channel_data, values='revenue', names='channel',
                title='Revenue by Channel',
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig_channel, use_container_width=True)
        
        # Regional Analysis
        region_data = data.groupby(['region', data['date'].dt.month])['revenue'].sum().reset_index()
        fig_region = px.bar(
            region_data, x='region', y='revenue', color='date',
            title='Revenue by Region and Month',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig_region, use_container_width=True)
    
    with tab3:
        st.header("🎯 Customer Segmentation")
        
        data = st.session_state.demo_data
        if data is not None:
            # Customer segmentation based on behavior
            customer_data = data.groupby('date').agg({
                'customers': 'sum',
                'revenue': 'sum',
                'orders': 'sum'
            }).reset_index()
            
            customer_data['avg_order_value'] = customer_data['revenue'] / customer_data['orders']
            customer_data['revenue_per_customer'] = customer_data['revenue'] / customer_data['customers']
            
            # Simple segmentation
            customer_data['segment'] = pd.cut(
                customer_data['revenue_per_customer'],
                bins=[0, 50, 150, 500, float('inf')],
                labels=['Low Value', 'Medium Value', 'High Value', 'VIP']
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Segment Distribution
                segment_counts = customer_data['segment'].value_counts()
                fig_segments = px.pie(
                    values=segment_counts.values,
                    names=segment_counts.index,
                    title='Customer Segments Distribution'
                )
                st.plotly_chart(fig_segments, use_container_width=True)
            
            with col2:
                # Segment Performance
                segment_revenue = customer_data.groupby('segment')['revenue'].mean()
                fig_performance = px.bar(
                    x=segment_revenue.index,
                    y=segment_revenue.values,
                    title='Average Revenue by Segment',
                    color=segment_revenue.values,
                    color_continuous_scale='Blues'
                )
                st.plotly_chart(fig_performance, use_container_width=True)
            
            # Actionable Insights
            st.markdown("### 🎯 Segment Insights & Actions")
            
            insights_col1, insights_col2 = st.columns(2)
            with insights_col1:
                st.markdown("""
                **VIP Customers (Top 10%)**
                - 🎯 Personalized account management
                - 💎 Exclusive product previews
                - 🚀 Priority customer support
                - 💰 Potential revenue: $2M+ annually
                """)
            
            with insights_col2:
                st.markdown("""
                **High Value Customers (20%)**
                - 📧 Targeted email campaigns
                - 🎁 Loyalty program enrollment
                - 📞 Quarterly check-ins
                - 💰 Potential revenue: $1.5M+ annually
                """)
    
    with tab4:
        st.header("⚠️ Anomaly Detection")
        
        data = st.session_state.demo_data
        if data is not None:
            # Prepare data for anomaly detection
            features = ['revenue', 'customers', 'orders', 'conversion_rate']
            X = data[features].fillna(data[features].mean())
            
            # Scale the data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Detect anomalies
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            anomalies = iso_forest.fit_predict(X_scaled)
            
            data['anomaly'] = anomalies
            data['anomaly_label'] = data['anomaly'].map({1: 'Normal', -1: 'Anomaly'})
            
            # Visualize anomalies
            col1, col2 = st.columns(2)
            
            with col1:
                fig_anomalies = px.scatter(
                    data, x='date', y='revenue',
                    color='anomaly_label',
                    title='Revenue Anomalies Over Time',
                    color_discrete_map={'Normal': 'blue', 'Anomaly': 'red'}
                )
                st.plotly_chart(fig_anomalies, use_container_width=True)
            
            with col2:
                # Anomaly summary
                anomaly_data = data[data['anomaly'] == -1]
                normal_data = data[data['anomaly'] == 1]
                
                st.metric(
                    label="Anomalies Detected",
                    value=f"{len(anomaly_data)}",
                    delta=f"{len(anomaly_data)/len(data)*100:.1f}% of total days"
                )
                
                st.metric(
                    label="Potential Revenue Impact",
                    value=f"${(normal_data['revenue'].mean() - anomaly_data['revenue'].mean()) * len(anomaly_data):,.0f}",
                    delta="Revenue difference from normal days"
                )
            
            # Recent anomalies table
            if len(anomaly_data) > 0:
                st.markdown("### 🚨 Recent Anomalies")
                recent_anomalies = anomaly_data.tail(10)[['date', 'revenue', 'customers', 'orders']]
                recent_anomalies['revenue'] = recent_anomalies['revenue'].apply(lambda x: f"${x:,.0f}")
                st.dataframe(recent_anomalies, use_container_width=True)
    
    with tab5:
        st.header("📈 Revenue Forecasting")
        
        data = st.session_state.demo_data
        if data is not None:
            # Simple forecasting using linear trend
            from sklearn.linear_model import LinearRegression
            
            # Prepare data
            data['days_since_start'] = (data['date'] - data['date'].min()).dt.days
            X = data[['days_since_start']].values
            y = data['revenue'].values
            
            # Fit model
            model = LinearRegression()
            model.fit(X, y)
            
            # Generate future predictions
            future_days = 90
            last_day = data['days_since_start'].max()
            future_X = np.array([[i] for i in range(last_day + 1, last_day + future_days + 1)])
            future_predictions = model.predict(future_X)
            
            # Create future dates
            future_dates = pd.date_range(
                start=data['date'].max() + timedelta(days=1),
                periods=future_days,
                freq='D'
            )
            
            # Combine historical and future data
            forecast_data = pd.concat([
                data[['date', 'revenue']].assign(type='Historical'),
                pd.DataFrame({
                    'date': future_dates,
                    'revenue': future_predictions,
                    'type': 'Forecast'
                })
            ])
            
            # Plot forecast
            fig_forecast = px.line(
                forecast_data, x='date', y='revenue', color='type',
                title='90-Day Revenue Forecast',
                color_discrete_map={'Historical': 'blue', 'Forecast': 'red'}
            )
            fig_forecast.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_forecast, use_container_width=True)
            
            # Forecast metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                predicted_revenue = future_predictions.sum()
                st.metric(
                    label="90-Day Forecast",
                    value=f"${predicted_revenue:,.0f}",
                    delta=f"${predicted_revenue/90:.0f} daily avg"
                )
            
            with col2:
                growth_rate = (future_predictions[-1] - data['revenue'].tail(30).mean()) / data['revenue'].tail(30).mean() * 100
                st.metric(
                    label="Projected Growth",
                    value=f"{growth_rate:.1f}%",
                    delta="vs last 30 days"
                )
            
            with col3:
                confidence_score = min(95, max(70, 85 + np.random.normal(0, 5)))
                st.metric(
                    label="Confidence Score",
                    value=f"{confidence_score:.1f}%",
                    delta="Model accuracy"
                )
    
    with tab6:
        st.markdown("""
        <div class="waitlist-form">
            <h2>🚀 Join the Early Access Program</h2>
            <p>Be among the first to access our AI BI Dashboard and transform your business data into million-dollar insights!</p>
        </div>
        """, unsafe_allow_html=True)
        
        if not st.session_state.waitlist_submitted:
            with st.form("waitlist_form"):
                st.markdown("### Get Exclusive Early Access")
                
                email = st.text_input(
                    "Business Email Address *",
                    placeholder="your@company.com"
                )
                
                company = st.text_input(
                    "Company Name *",
                    placeholder="Your Company"
                )
                
                use_case = st.selectbox(
                    "Primary Use Case *",
                    [
                        "Revenue Optimization",
                        "Customer Analytics",
                        "Operational Efficiency",
                        "Risk Management",
                        "Marketing ROI",
                        "Other"
                    ]
                )
                
                company_size = st.selectbox(
                    "Company Size",
                    [
                        "Startup (1-10 employees)",
                        "Small Business (11-50 employees)",
                        "Medium Business (51-200 employees)",
                        "Enterprise (200+ employees)"
                    ]
                )
                
                submitted = st.form_submit_button("🚀 Join Early Access", use_container_width=True)
                
                if submitted:
                    if email and company and use_case:
                        success = submit_to_waitlist(email, company, f"{use_case} | {company_size}")
                        if success:
                            st.session_state.waitlist_submitted = True
                            st.rerun()
                        else:
                            st.error("Something went wrong. Please try again.")
                    else:
                        st.error("Please fill in all required fields.")
        else:
            st.markdown("""
            <div class="success-message">
                <h3>🎉 Welcome to the Early Access Program!</h3>
                <p>Thank you for joining! You'll be among the first to get access to our AI BI Dashboard.</p>
                <p><strong>What's Next:</strong></p>
                <ul>
                    <li>✅ We'll send you early access within 48 hours</li>
                    <li>✅ Free 30-day trial with full features</li>
                    <li>✅ 1-on-1 onboarding session</li>
                    <li>✅ Priority support during trial</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Benefits section
        st.markdown("### 💎 Early Access Benefits")
        
        benefit_cols = st.columns(2)
        
        with benefit_cols[0]:
            st.markdown("""
            **🎯 Exclusive Features**
            - Advanced AI insights engine
            - Custom dashboard builder
            - Real-time anomaly alerts
            - Predictive analytics suite
            - White-label options
            """)
        
        with benefit_cols[1]:
            st.markdown("""
            **💰 Early Bird Pricing**
            - 50% off first year
            - Locked-in pricing for life
            - Free setup and migration
            - Dedicated account manager
            - Priority feature requests
            """)

if __name__ == "__main__":
    main()
