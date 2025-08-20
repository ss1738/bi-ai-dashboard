import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from io import BytesIO
from dataclasses import dataclass
from datetime import datetime

# ---- XLSX export helper ----
def df_to_xlsx_bytes(sheets: dict) -> bytes:
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="xlsxwriter") as writer:
        for name, d in (sheets or {}).items():
            if d is None:
                continue
            try:
                (d if isinstance(d, pd.DataFrame) else pd.DataFrame(d)).to_excel(
                    writer, index=False, sheet_name=(name or "sheet")[:31]
                )
            except Exception:
                try:
                    pd.DataFrame(d).to_excel(
                        writer, index=False, sheet_name=(name or "sheet")[:31]
                    )
                except Exception:
                    pass
    return bio.getvalue()

# ---- Demo dataset ----
def load_demo():
    dates = pd.date_range("2024-01-01", periods=365)
    df = pd.DataFrame({
        "date": dates,
        "region": np.random.choice(["AMER","APAC","EMEA"], len(dates)),
        "channel": np.random.choice(["Online","Retail","Wholesale"], len(dates)),
        "product": np.random.choice(["A","B","C"], len(dates)),
        "revenue": np.random.randint(100,1000,len(dates)),
        "units": np.random.randint(1,100,len(dates))
    })
    return df

# ---- Sidebar Navigation ----
pages = ["📊 Dashboard","🧩 Segmentation","🚨 Anomalies","📈 Forecast","🧠 AI Insights","🕒 Early Access Waitlist"]
page = st.sidebar.radio("Go to", pages)

# ---- File uploader ----
st.sidebar.subheader("Data source")
upload = st.sidebar.file_uploader("Upload CSV/Parquet/Excel")
if upload:
    try:
        if upload.name.endswith(".csv"):
            df = pd.read_csv(upload)
        elif upload.name.endswith(".parquet"):
            df = pd.read_parquet(upload)
        else:
            df = pd.read_excel(upload)
    except Exception:
        df = load_demo()
else:
    df = load_demo()

# ---- Permalink Filter Helper ----
def encode_filters(region, channel):
    return f"?region={region}&channel={channel}"

def parse_filters():
    query = st.query_params
    r = query.get("region")
    c = query.get("channel")
    return r,c

# Apply query param filters if exist
q_region,q_channel = parse_filters()
if q_region:
    df = df[df["region"]==q_region]
if q_channel:
    df = df[df["channel"]==q_channel]

# ---- Pages ----
if page == "📊 Dashboard":
    st.title("AI BI Dashboard")
    st.subheader("Overview")
    st.metric("Total Revenue", f"£{df['revenue'].sum():,}")
    st.metric("Units Sold", f"{df['units'].sum():,}")

    st.subheader("Download Data")
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv_bytes, file_name="filtered.csv")
    st.download_button("Download JSON", df.to_json(orient="records").encode("utf-8"), file_name="filtered.json")
    xlsx_bytes = df_to_xlsx_bytes({"data_filtered":df})
    st.download_button("Download XLSX", xlsx_bytes, file_name="export.xlsx")

    st.button("🔗 Copy link to this section", key="copy_dl_btn")
    st.components.v1.html("""
    <script>
    (function(){
      const findBtn = () => Array.from(window.parent.document.querySelectorAll('button')).find(b => /Copy link/.test(b.textContent));
      function bind(){
        const btn = findBtn();
        if(!btn || btn.__bound) return;
        btn.__bound = true;
        btn.addEventListener('click', () => {
          try{
            const url = window.location.origin + window.location.pathname + '#download-data';
            navigator.clipboard.writeText(url);
          }catch(e){}
        });
      }
      setTimeout(bind, 300);
      setTimeout(bind, 1000);
    })();
    </script>
    """, height=0)

elif page == "🧩 Segmentation":
    st.title("Segmentation")
    st.write("Cluster users/products using KMeans…")

elif page == "🚨 Anomalies":
    st.title("Anomalies")
    st.write("Detect anomalies via Z-score / Isolation Forest…")
    st.session_state["__last_anomalies_df"] = df.head()

elif page == "📈 Forecast":
    st.title("Forecast")
    st.write("Holt-Winters / ARIMA forecast preview…")
    fdf = pd.DataFrame({"date":pd.date_range("2025-01-01", periods=10),"forecast":np.random.rand(10)})
    st.line_chart(fdf.set_index("date"))
    st.session_state["__last_forecast_df"] = fdf

elif page == "🧠 AI Insights":
    st.title("AI Insights")
    st.write("Rule-based insights like YoY growth, movers, correlations…")

elif page == "🕒 Early Access Waitlist":
    st.title("Early Access Waitlist")
    st.markdown("Fill the form to join the waitlist:")
    st.components.v1.iframe("https://docs.google.com/forms/d/e/1FAIpQLSf-demo-form/viewform?embedded=true", height=600)

# ---- E-commerce Preset Toggle ----
with st.sidebar.expander("Presets"):
    if st.button("E-commerce Preset"):
        # Force filters: region=APAC, channel=Online, metric=revenue
        url = st.experimental_get_query_params()
        st.query_params["region"] = "APAC"
        st.query_params["channel"] = "Online"
        st.experimental_rerun()

# ---- Landing Page (static) ----
if page == "Landing":
    st.title("🚀 AI BI Dashboard")
    st.subheader("Upload your CSV → get insights in minutes")
    st.write("✔️ Anomalies  ✔️ Forecast  ✔️ Scenario Simulations")
    st.write("Demo below:")
    st.video("https://www.youtube.com/embed/dQw4w9WgXcQ")
    st.write("Join our waitlist:")
    st.components.v1.iframe("https://docs.google.com/forms/d/e/1FAIpQLSf-demo-form/viewform?embedded=true", height=600)
