# Create a fixed Streamlit app and requirements.txt for the user

app_code = r'''# AI BI Dashboard - Complete Production App (Altair-only, Fixed)
# Save as: streamlit_app.py
#
# requirements.txt (create alongside this file):
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
import base64
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

    .stApp {{
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }}

    .main .block-container {{
        padding: 2rem 1rem;
        max-width: 1200px;
    }}

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
        font-size: 3.2rem;
        font-weight: 700;
        margin-bottom: 1rem;
        background: linear-gradient(45deg, #ffffff, #f0f9ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}
    .hero p {{font-size: 1.2rem; opacity: 0.95; margin-bottom: 2rem;}}

    .nav-container {{
        background: white;
        padding: 1rem 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        margin: 2rem 0;
        display: flex;
        justify-content: center;
        flex-wrap: wrap;
        gap: 0.75rem;
    }}
    .nav-btn {{
        background: linear-gradient(135deg, var(--primary), var(--secondary));
        color: white; padding: 10px 18px; border: none; border-radius: 10px;
        cursor: pointer; font-weight: 600; transition: all 0.2s ease;
        text-decoration: none; display: inline-block;
    }}
    .nav-btn:hover {{transform: translateY(-2px); box-shadow: 0 8px 25px rgba(99, 102, 241, 0.3);}}
    .nav-btn.active {{background: var(--dark);}}

    .metric-card {{
        background: white; padding: 1.5rem; border-radius: 15px;
        box-shadow: 0 8px 30px rgba(0,0,0,0.08);
        border: 1px solid rgba(255,255,255,0.2);
        transition: all 0.3s ease; height: 100%; margin-bottom: 1rem;
    }}
    .metric-card:hover {{transform: translateY(-5px); box-shadow: 0 15px 40px rgba(0,0,0,0.12);}}
    .metric-value {{font-size: 2rem; font-weight: 700; color: var(--primary); margin: 0.5rem 0;}}
    .metric-label {{color: #6B7280; font-weight: 500; margin-bottom: 0.5rem;}}
    .metric-delta {{font-weight: 600; font-size: 0.9rem;}}
    .delta-positive {{ color: var(--success); }}
    .delta-negative {{ color: #EF4444; }}

    .success-box {{
        background: linear-gradient(135deg, var(--success), #059669);
        color: white; padding: 1.25rem; border-radius: 12px; margin: 1rem 0; font-weight: 600;
    }}

    .feature-card {{
        background: white; padding: 1.5rem; border-radius: 15px; text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05); border: 2px solid transparent; height: 100%;
    }}
    .feature-card:hover {{border-color: var(--primary); transform: translateY(-3px);
        box-shadow: 0 8px 30px rgba(99, 102, 241, 0.15);}}
    .feature-icon {{font-size: 2.2rem; margin-bottom: 0.75rem;}}

    .stAltairChart {{
        background: white; border-radius: 15px; padding: 0.75rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08); margin: 0.75rem 0;
    }}

    @keyframes fadeInUp {{
        from {{opacity: 0; transform: translateY(30px);}}
        to {{opacity: 1; transform: translateY(0);}}
    }}
    </style>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 📊 DATA GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def generate_business_data(seed: int = 42) -> pd.DataFrame:
    """Generate realistic business data with revenue recovery opportunities (2 years)."""
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
        month = date.month
        seasonal_mult = 1.2 if month in (11, 12) else (0.8 if month in (7, 8) else 1.0)
        weekday = date.weekday()
        weekly_mult = 0.7 if weekday >= 5 else 1.0

        for segment in segments:
            base_revenue = {"Enterprise": 50000, "Mid-Market": 15000, "SMB": 5000, "Startup": 1500}[segment]
            for channel in channels:
                channel_mult = {"Direct Sales": 1.3, "Partner": 1.1, "Online": 0.9, "Retail": 0.8}[channel]
                for region in regions:
                    region_mult = {"North America": 1.2, "Europe": 1.0, "Asia Pacific": 0.9, "Latin America": 0.7}[region]

                    revenue = (
                        base_revenue
                        * seasonal_mult
                        * weekly_mult
                        * channel_mult
                        * region_mult
                        * np.random.normal(1, 0.15)
                    )
                    revenue = max(0, revenue)

                    customers = max(1, int(np.random.poisson(revenue / (base_revenue / 10))))
                    avg_deal_size = revenue / customers if customers else 0.0
                    churn_rate = np.random.uniform(0.02, 0.08)

                    # Inject anomalies (recovery opportunities)
                    if np.random.random() < 0.05:
                        revenue *= 0.3

                    rows.append(
                        {
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
                        }
                    )
    return pd.DataFrame(rows)

# ═══════════════════════════════════════════════════════════════════════════════
# 🧠 AI/ML FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def perform_segmentation(df: pd.DataFrame) -> pd.DataFrame:
    """Cluster segments by revenue, customers, avg deal size, LTV."""
    segment_data = (
        df.groupby("segment")
        .agg(
            revenue=("revenue", "sum"),
            customers=("customers", "sum"),
            avg_deal_size=("avg_deal_size", "mean"),
            churn_rate=("churn_rate", "mean"),
            lifetime_value=("lifetime_value", "mean"),
        )
        .reset_index()
    )
    features = ["revenue", "customers", "avg_deal_size", "lifetime_value"]
    X = segment_data[features].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
    segment_data["cluster"] = kmeans.fit_predict(X_scaled)

    cluster_labels = {0: "High Value", 1: "Growing", 2: "Opportunity"}
    # If labels map wrongly due to randomness, keep numeric too
    segment_data["cluster_name"] = segment_data["cluster"].map(cluster_labels).fillna(segment_data["cluster"].astype(str))
    return segment_data

def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Detect daily revenue anomalies with Isolation Forest."""
    daily = df.groupby("date", as_index=False)["revenue"].sum()
    daily["day_of_week"] = daily["date"].dt.dayofweek
    daily["month"] = daily["date"].dt.month
    daily["revenue_lag1"] = daily["revenue"].shift(1)
    daily["revenue_lag7"] = daily["revenue"].shift(7)
    daily = daily.dropna()

    feats = ["revenue", "day_of_week", "month", "revenue_lag1", "revenue_lag7"]
    X = daily[feats].values

    iso = IsolationForest(contamination=0.1, random_state=42)
    daily["anomaly"] = iso.fit_predict(X)
    daily["anomaly_score"] = iso.score_samples(X)

    anomalies = daily[daily["anomaly"] == -1].copy()
    anomalies["potential_loss"] = anomalies["revenue"] * 0.3
    return anomalies.sort_values("date", ascending=False)

def generate_forecast(df: pd.DataFrame, days: int = 30) -> pd.DataFrame:
    """Simple linear forecast for daily revenue."""
    daily = df.groupby("date", as_index=False)["revenue"].sum()
    daily["day_num"] = (daily["date"] - daily["date"].min()).dt.days

    X = daily[["day_num"]].values
    y = daily["revenue"].values

    model = LinearRegression()
    model.fit(X, y)

    last_day = daily["day_num"].max()
    future_days = np.arange(last_day + 1, last_day + days + 1).reshape(-1, 1)
    forecast = model.predict(future_days)

    future_dates = pd.date_range(daily["date"].max() + timedelta(days=1), periods=days)
    forecast_df = pd.DataFrame({"date": future_dates, "value": forecast, "type": "Forecast"})
    hist_df = daily[["date", "revenue"]].rename(columns={"revenue": "value"})
    hist_df["type"] = "Historical"

    return pd.concat([hist_df, forecast_df], ignore_index=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 🎯 NAVIGATION
# ═══════════════════════════════════════════════════════════════════════════════

def render_navigation():
    if "current_page" not in st.session_state:
        st.session_state.current_page = "Home"

    pages = {"Home": "🏠", "Dashboard": "📊", "Segmentation": "🎯", "Anomaly Detection": "🚨", "Forecasting": "📈", "Early Access": "🚀"}

    st.markdown('<div class="nav-container">', unsafe_allow_html=True)
    cols = st.columns(len(pages))
    for i, (page, icon) in enumerate(pages.items()):
        active = " active" if st.session_state.current_page == page else ""
        with cols[i]:
            if st.button(f"{icon} {page}", key=f"nav_{page}", use_container_width=True):
                st.session_state.current_page = page
    st.markdown("</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 📱 COMPONENTS
# ═══════════════════════════════════════════════════════════════════════════════

def render_hero():
    st.markdown(
        f"""
        <div class="hero">
            <h1>💰 {BRAND['name']}</h1>
            <p>{BRAND['tagline']}</p>
            <div style="display:flex;justify-content:center;gap:2rem;margin-top:2rem;">
                <div style="text-align:center;">
                    <div style="font-size:2rem;font-weight:bold;">$500K+</div>
                    <div style="opacity:0.9;">Average Recovery</div>
                </div>
                <div style="text-align:center;">
                    <div style="font-size:2rem;font-weight:bold;">87%</div>
                    <div style="opacity:0.9;">Success Rate</div>
                </div>
                <div style="text-align:center;">
                    <div style="font-size:2rem;font-weight:bold;">30 Days</div>
                    <div style="opacity:0.9;">Average ROI Time</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_metrics_cards(df: pd.DataFrame):
    total_revenue = df["revenue"].sum()
    total_customers = df["customers"].sum()
    avg_deal_size = df["avg_deal_size"].mean()
    avg_churn = df["churn_rate"].mean()

    prev_revenue = total_revenue * 0.92
    prev_customers = total_customers * 0.95
    prev_deal_size = avg_deal_size * 0.98
    prev_churn = avg_churn * 1.1

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        delta = ((total_revenue - prev_revenue) / prev_revenue) * 100
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Total Revenue</div>
                <div class="metric-value">${total_revenue:,.0f}</div>
                <div class="metric-delta {'delta-positive' if delta>=0 else 'delta-negative'}">{delta:+.1f}% vs last period</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        delta = ((total_customers - prev_customers) / prev_customers) * 100
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Total Customers</div>
                <div class="metric-value">{total_customers:,.0f}</div>
                <div class="metric-delta {'delta-positive' if delta>=0 else 'delta-negative'}">{delta:+.1f}% vs last period</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        delta = ((avg_deal_size - prev_deal_size) / prev_deal_size) * 100
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Avg Deal Size</div>
                <div class="metric-value">${avg_deal_size:,.0f}</div>
                <div class="metric-delta {'delta-positive' if delta>=0 else 'delta-negative'}">{delta:+.1f}% vs last period</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        delta = ((prev_churn - avg_churn) / prev_churn) * 100
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Churn Rate</div>
                <div class="metric-value">{avg_churn:.1%}</div>
                <div class="metric-delta {'delta-positive' if delta>=0 else 'delta-negative'}">{-delta:+.1f}% vs last period</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ═══════════════════════════════════════════════════════════════════════════════
# 🏠 PAGE: HOME
# ═══════════════════════════════════════════════════════════════════════════════

def page_home():
    render_hero()
    st.markdown("## 🎯 Revenue Recovery Platform Features")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">🤖</div>
                <h3>AI-Powered Analytics</h3>
                <p>ML algorithms surface revenue leakage and optimization opportunities in real time.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">⚡</div>
                <h3>Instant Insights</h3>
                <p>Actionable insights in minutes, not months—no heavy setup required.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">💎</div>
                <h3>Proven ROI</h3>
                <p>Clients recover $500K+ within 30 days on average.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("## 🏆 Success Stories")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """
            <div class="success-box">
                <h4>🚀 TechCorp Inc.</h4>
                <p>"Recovered $1.2M in 45 days—found missing enterprise segments."</p>
                <p><strong>— Sarah Chen, CRO</strong></p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="success-box">
                <h4>🎯 Global Dynamics</h4>
                <p>"Anomaly detection caught an $800K revenue leak. Massive save."</p>
                <p><strong>— Michael Rodriguez, VP Sales</strong></p>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ═══════════════════════════════════════════════════════════════════════════════
# 📊 PAGE: DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

def page_dashboard(df: pd.DataFrame):
    st.title("📊 Revenue Analytics Dashboard")
    st.markdown("### Real-time insights into your revenue performance")

    render_metrics_cards(df)
    st.markdown("---")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("📈 Revenue Trend (Daily)")
        daily = df.groupby("date", as_index=False)["revenue"].sum()
        line = (
            alt.Chart(daily)
            .mark_line(point=True, strokeWidth=2)
            .encode(
                x=alt.X("date:T", title="Date"),
                y=alt.Y("revenue:Q", title="Revenue ($)", axis=alt.Axis(format="~s")),
                tooltip=[alt.Tooltip("date:T", format="%Y-%m-%d"), alt.Tooltip("revenue:Q", format=",.0f")],
            )
            .properties(height=360)
            .interactive()
        )
        st.altair_chart(line, use_container_width=True)

    with col2:
        st.subheader("🎯 Revenue by Segment")
        seg = df.groupby("segment", as_index=False)["revenue"].sum().sort_values("revenue", ascending=False)
        pie = (
            alt.Chart(seg)
            .mark_arc()
            .encode(
                theta=alt.Theta("revenue:Q"),
                color=alt.Color("segment:N"),
                tooltip=["segment:N", alt.Tooltip("revenue:Q", format=",.0f")],
            )
            .properties(height=360)
        )
        st.altair_chart(pie, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 Channel Performance")
        channel = df.groupby("channel", as_index=False).agg(revenue=("revenue", "sum"), customers=("customers", "sum"))
        bars = (
            alt.Chart(channel)
            .mark_bar()
            .encode(
                x=alt.X("revenue:Q", title="Revenue ($)", axis=alt.Axis(format="~s")),
                y=alt.Y("channel:N", sort="-x", title="Channel"),
                tooltip=["channel:N", alt.Tooltip("revenue:Q", format=",.0f")],
            )
            .properties(height=320)
        )
        st.altair_chart(bars, use_container_width=True)

    with col2:
        st.subheader("🌍 Regional Analysis")
        region = df.groupby("region", as_index=False).agg(revenue=("revenue", "sum"), customers=("customers", "sum"))
        bars = (
            alt.Chart(region)
            .mark_bar()
            .encode(
                x=alt.X("revenue:Q", title="Revenue ($)", axis=alt.Axis(format="~s")),
                y=alt.Y("region:N", sort="-x", title="Region"),
                tooltip=["region:N", alt.Tooltip("revenue:Q", format=",.0f")],
            )
            .properties(height=320)
        )
        st.altair_chart(bars, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 🎯 PAGE: SEGMENTATION
# ═══════════════════════════════════════════════════════════════════════════════

def page_segmentation(df: pd.DataFrame):
    st.title("🎯 Customer Segmentation Analysis")
    st.markdown("### AI-powered segmentation reveals hidden revenue opportunities")

    segdata = perform_segmentation(df)

    # Cluster scatter plot (Avg deal vs LTV, size by revenue)
    scatter = (
        alt.Chart(segdata)
        .mark_circle(size=300)
        .encode(
            x=alt.X("avg_deal_size:Q", title="Average Deal Size ($)", axis=alt.Axis(format="~s")),
            y=alt.Y("lifetime_value:Q", title="Lifetime Value ($)", axis=alt.Axis(format="~s")),
            size=alt.Size("revenue:Q", title="Total Revenue"),
            color=alt.Color("cluster_name:N", title="Cluster"),
            tooltip=[
                "segment:N",
                alt.Tooltip("revenue:Q", format=",.0f"),
                alt.Tooltip("customers:Q", format=",d"),
                alt.Tooltip("avg_deal_size:Q", format=",.0f"),
                alt.Tooltip("lifetime_value:Q", format=",.0f"),
                "cluster_name:N",
            ],
        )
        .properties(height=420, title="Segment Clusters")
        .interactive()
    )
    st.altair_chart(scatter, use_container_width=True)

    st.markdown("#### Segment Summary")
    st.dataframe(segdata, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 🚨 PAGE: ANOMALY DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def page_anomalies(df: pd.DataFrame):
    st.title("🚨 Revenue Anomaly Detection")
    st.markdown("### Isolation Forest flags potential revenue losses")

    anomalies = detect_anomalies(df)
    if anomalies.empty:
        st.success("No significant anomalies detected. 🎉")
        return

    st.markdown("#### Anomalies (most recent first)")
    st.dataframe(anomalies[["date", "revenue", "anomaly_score", "potential_loss"]], use_container_width=True)

    base = alt.Chart(df.groupby("date", as_index=False)["revenue"].sum()).encode(x=alt.X("date:T", title="Date"))

    line = base.mark_line().encode(y=alt.Y("revenue:Q", title="Revenue ($)", axis=alt.Axis(format="~s")), tooltip=["date:T", alt.Tooltip("revenue:Q", format=",.0f")])
    points = alt.Chart(anomalies).mark_point(filled=True, size=80).encode(
        x="date:T",
        y="revenue:Q",
        tooltip=["date:T", alt.Tooltip("revenue:Q", format=",.0f"), alt.Tooltip("potential_loss:Q", format=",.0f")],
        color=alt.value("#EF4444"),
    )
    st.altair_chart((line + points).properties(height=380, title="Revenue with Anomalies"), use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 📈 PAGE: FORECASTING
# ═══════════════════════════════════════════════════════════════════════════════

def page_forecasting(df: pd.DataFrame):
    st.title("📈 Revenue Forecast")
    horizon = st.slider("Forecast horizon (days)", 7, 120, 30, 1)
    fc = generate_forecast(df, days=horizon)

    chart = (
        alt.Chart(fc)
        .mark_line(point=True)
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("value:Q", title="Revenue ($)", axis=alt.Axis(format="~s")),
            color="type:N",
            tooltip=["type:N", "date:T", alt.Tooltip("value:Q", format=",.0f")],
        )
        .properties(height=420, title="Historical vs Forecasted Revenue")
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 🚀 PAGE: EARLY ACCESS
# ═══════════════════════════════════════════════════════════════════════════════

def page_early_access():
    st.title("🚀 Get Early Access")
    st.markdown("Join the waitlist to pilot the platform.")

    with st.form("waitlist"):
        name = st.text_input("Full Name")
        email = st.text_input("Work Email")
        role = st.selectbox("Role", ["Founder/CEO", "CRO/Revenue Ops", "Sales Leader", "Data/BI", "Other"])
        use_case = st.text_input("Primary Use Case (e.g., reduce churn, upsell, pricing)")
        submit = st.form_submit_button("Request Access")

    if submit:
        if not name or not email:
            st.error("Please fill in name and email.")
        else:
            st.success("Thanks! We'll reach out with next steps. ✅")

# ═══════════════════════════════════════════════════════════════════════════════
# 🧭 APP ENTRY
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    load_css()
    render_navigation()

    # Data
    data = generate_business_data()

    page = st.session_state.get("current_page", "Home")
    if page == "Home":
        page_home()
    elif page == "Dashboard":
        page_dashboard(data)
    elif page == "Segmentation":
        page_segmentation(data)
    elif page == "Anomaly Detection":
        page_anomalies(data)
    elif page == "Forecasting":
        page_forecasting(data)
    elif page == "Early Access":
        page_early_access()
    else:
        page_home()

if __name__ == "__main__":
    main()
'''

reqs = """streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
altair>=5.0.0
scikit-learn>=1.3.0
"""

# Write files
with open('/mnt/data/streamlit_app.py', 'w', encoding='utf-8') as f:
    f.write(app_code)

with open('/mnt/data/requirements.txt', 'w', encoding='utf-8') as f:
    f.write(reqs)

print("Files created:")
print(" - /mnt/data/streamlit_app.py")
print(" - /mnt/data/requirements.txt")
