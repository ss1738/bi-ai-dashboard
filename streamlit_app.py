import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.arima.model import ARIMA

# ---------- Page config ----------
st.set_page_config(page_title="📊 AI BI Dashboard", layout="wide")
st.markdown("<a id='top'></a>", unsafe_allow_html=True)
st.title("📊 Interactive BI Dashboard + 🤖 AI Insights")

# ---------- Helpers ----------
def retail_demo_df():
    return pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=30, freq="D"),
        "category": (["Electronics","Fashion","Groceries"] * 10)[:30],
        "sales": [1200, 900, 600] * 10,
        "profit": [200, 150, 80] * 10,
    })

def saas_mrr_df():
    vals = [5000,5200,5400,5600,5900,6200,6400,6600,7000,7400,7800,8200,8600,9000,9400,9800,10200,10600]
    return pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=len(vals), freq="M"),
        "category": ["MRR"] * len(vals),
        "sales": vals,
        "profit": [v * 0.7 for v in vals],
    })

def csv_template_bytes():
    tmp = pd.DataFrame({"date": [], "category": [], "sales": [], "profit": []})
    return tmp.to_csv(index=False).encode("utf-8")

# ---------- Sidebar: value capture, samples, filters ----------
st.sidebar.header("📬 Stay in the loop")
email = st.sidebar.text_input("Your email")
if st.sidebar.button("Notify me"):
    if email.strip():
        st.sidebar.success("Thanks! We’ll notify you about updates.")
        # Tip: replace the link below with your Google Form / Tally / webhook
        st.sidebar.markdown("[Join the waitlist](https://forms.gle/)" )
    else:
        st.sidebar.warning("Please enter a valid email.")

st.sidebar.markdown("---")
st.sidebar.header("📁 Load data")

sample_choice = st.sidebar.selectbox(
    "Load a sample dataset (optional)",
    ["—", "Retail (demo)", "SaaS MRR"]
)

uploaded_file = st.sidebar.file_uploader("Or upload CSV", type="csv", key="u_csv")

# Decide source of df
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded successfully.")
    except Exception as e:
        st.error(f"Could not read the CSV: {e}")
        df = pd.DataFrame()
elif sample_choice == "Retail (demo)":
    df = retail_demo_df()
    st.info("📂 Using Retail demo data.")
elif sample_choice == "SaaS MRR":
    df = saas_mrr_df()
    st.info("📂 Using SaaS MRR sample data.")
else:
    st.info("📂 No file uploaded. Using default demo data…")
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=20, freq="D"),
        "category": ["Electronics","Fashion","Groceries","Electronics","Fashion",
                     "Groceries","Electronics","Fashion","Groceries","Electronics"]*2,
        "sales": [1200, 900, 600, 1500, 1100, 800, 1700, 950, 720, 1400,
                  1300, 920, 640, 1600, 1120, 850, 1750, 980, 750, 1450],
        "profit": [200, 150, 80, 300, 220, 120, 330, 180, 100, 260,
                   210, 160, 90, 320, 230, 130, 350, 190, 110, 270],
    })

# Normalize common types
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
for col in df.columns:
    if df[col].dtype == "object" and col not in ("category",):
        # best-effort numeric conversion
        df[col] = pd.to_numeric(df[col], errors="ignore")

# Sidebar filters
st.sidebar.markdown("---")
st.sidebar.header("🔎 Filters")
cat_col = "category" if "category" in df.columns else None
time_col = "date" if "date" in df.columns else None

df_flt = df.copy()
if cat_col:
    categories = sorted(df_flt[cat_col].dropna().astype(str).unique())
    chosen = st.sidebar.multiselect("Categories", categories, default=categories)
    df_flt = df_flt[df_flt[cat_col].astype(str).isin(chosen)]

if time_col and df_flt[time_col].notna().any():
    dmin, dmax = pd.to_datetime(df_flt[time_col].min()), pd.to_datetime(df_flt[time_col].max())
    dr = st.sidebar.date_input("Date range", [dmin.date(), dmax.date()])
    if isinstance(dr, (list, tuple)) and len(dr) == 2:
        df_flt = df_flt[(df_flt[time_col] >= pd.to_datetime(dr[0])) &
                        (df_flt[time_col] <= pd.to_datetime(dr[1]))]

st.sidebar.markdown("---")
st.sidebar.markdown("[⬇️ Go to Download](#download-data)")
st.sidebar.markdown("🐞 Problems? [Email me](mailto:you@example.com)")

# ---------- Tabs ----------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📊 Dashboard", "📌 Segmentation", "⚠️ Anomalies", "🔮 Forecast", "🤖 AI Insights"]
)

# === Dashboard ===
with tab1:
    if df_flt.empty:
        st.error("No data after filters.")
    else:
        st.subheader("📈 Key Metrics")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", len(df_flt))
        c2.metric("Columns", df_flt.shape[1])
        c3.metric("Total Sales", f"{df_flt['sales'].sum():,.0f}" if "sales" in df_flt else "—")
        c4.metric("Avg Sales", f"{df_flt['sales'].mean():,.2f}" if "sales" in df_flt else "—")

        st.subheader("📊 Charts")
        if time_col and "sales" in df_flt and df_flt[time_col].notna().any():
            fig_ts = px.line(
                df_flt.sort_values(time_col),
                x=time_col, y="sales",
                color=cat_col if cat_col else None,
                markers=True, title="Sales Over Time"
            )
            st.plotly_chart(fig_ts, use_container_width=True)

        if cat_col and "sales" in df_flt:
            gp = df_flt.groupby(cat_col, as_index=False)["sales"].sum().sort_values("sales", ascending=False)
            st.plotly_chart(px.bar(gp, x=cat_col, y="sales", title="Sales by Category"),
                            use_container_width=True)

        if cat_col and "profit" in df_flt:
            gp2 = df_flt.groupby(cat_col, as_index=False)["profit"].sum()
            st.plotly_chart(px.pie(gp2, names=cat_col, values="profit", title="Profit Share by Category"),
                            use_container_width=True)

        st.markdown("### 📄 Template")
        st.download_button(
            "Download CSV template",
            csv_template_bytes(),
            file_name="template.csv",
            mime="text/csv",
        )

# === Segmentation ===
with tab2:
    st.subheader("📌 Customer Segmentation (KMeans)")
    if all(c in df_flt.columns for c in ["sales", "profit"]) and len(df_flt) >= 3:
        try:
            X = df_flt[["sales", "profit"]].dropna()
            if len(X) >= 3:
                kmeans = KMeans(n_clusters=3, random_state=42).fit(X)
                tmp = df_flt.loc[X.index].copy()
                tmp["segment"] = kmeans.labels_.astype(str)
                st.plotly_chart(
                    px.scatter(tmp, x="sales", y="profit", color="segment", title="Customer Segments"),
                    use_container_width=True
                )
                st.dataframe(tmp[[ "sales","profit","segment" ]].head(20), use_container_width=True)
            else:
                st.info("Not enough rows after dropping NA to cluster.")
        except Exception as e:
            st.error(f"Segmentation failed: {e}")
    else:
        st.info("Need 'sales' and 'profit' columns with at least 3 rows.")

# === Anomalies ===
with tab3:
    st.subheader("⚠️ Anomaly Detection (IsolationForest)")
    if "sales" in df_flt and len(df_flt) > 10:
        try:
            X2 = df_flt[["sales"]].dropna()
            if len(X2) > 10:
                model = IsolationForest(contamination=0.1, random_state=42).fit(X2)
                df_loc = df_flt.loc[X2.index].copy()
                df_loc["anomaly"] = model.predict(X2)  # -1 = anomaly
                st.plotly_chart(
                    px.scatter(
                        df_loc,
                        x=(time_col if time_col else df_loc.index),
                        y="sales",
                        color="anomaly",
                        title="Anomalies in Sales (−1 = anomaly)"
                    ),
                    use_container_width=True
                )
                st.dataframe(df_loc[df_loc["anomaly"] == -1], use_container_width=True)
            else:
                st.info("Need > 10 usable rows for anomaly detection.")
        except Exception as e:
            st.error(f"Anomaly detection failed: {e}")
    else:
        st.info("Need 'sales' column and > 10 rows.")

# === Forecast ===
with tab4:
    st.subheader("🔮 Sales Forecast (ARIMA)")
    if time_col and "sales" in df_flt and df_flt[time_col].notna().any():
        try:
            ts = df_flt.set_index(time_col)["sales"].dropna()
            if len(ts) >= 8:
                ts = ts.resample("D").sum().asfreq("D").fillna(method="ffill")
                model = ARIMA(ts, order=(1, 1, 1))
                fit = model.fit()
                fc = fit.forecast(steps=7)
                hist = pd.DataFrame({"date": ts.index, "value": ts.values})
                fc_df = pd.DataFrame({"date": fc.index, "forecast": fc.values})
                fig = px.line(hist, x="date", y="value", title="7-Day Sales Forecast")
                fig.add_scatter(x=fc_df["date"], y=fc_df["forecast"], mode="lines+markers", name="Forecast")
                st.plotly_chart(fig, use_container_width=True)
                st.download_button("Download forecast (CSV)",
                                   fc_df.to_csv(index=False).encode(),
                                   "forecast.csv", "text/csv")
            else:
                st.info("Need at least 8 periods to forecast.")
        except Exception as e:
            st.error(f"Forecasting failed: {e}")
    else:
        st.info("Need a valid date column and 'sales' to forecast.")

# === AI Insights (simple rule-based) ===
with tab5:
    st.subheader("🤖 AI Insights (MVP)")
    bullets = []
    if "sales" in df_flt and len(df_flt):
        bullets.append(f"Total sales: **{df_flt['sales'].sum():,.0f}**; average per row: **{df_flt['sales'].mean():,.2f}**.")
    if cat_col and "sales" in df_flt and len(df_flt):
        gp = df_flt.groupby(cat_col, as_index=False)["sales"].sum().sort_values("sales", ascending=False)
        if len(gp):
            top = gp.iloc[0]
            share = 100 * top["sales"] / max(df_flt["sales"].sum(), 1)
            bullets.append(f"Top {cat_col}: **{top[cat_col]}** ({top['sales']:,.0f}, {share:.1f}% of total).")
    if time_col and "sales" in df_flt and df_flt[time_col].notna().any():
        by_week = df_flt.set_index(time_col)["sales"].resample("W").sum()
        if len(by_week) >= 2:
            delta = (by_week.iloc[-1] - by_week.iloc[-2]) / max(by_week.iloc[-2], 1)
            bullets.append(f"Last week vs prior: **{delta*100:+.1f}%**.")
    if "profit" in df_flt and "sales" in df_flt and df_flt["sales"].sum() > 0:
        margin = 100 * df_flt["profit"].sum() / df_flt["sales"].sum()
        bullets.append(f"Overall margin: **{margin:.1f}%**.")
    if not bullets:
        bullets.append("Provide at least a numeric value column (e.g., sales) for insights.")
    for b in bullets:
        st.markdown(f"- {b}")

# ---------- Download section (anchor) ----------
st.markdown("<a id='download-data'></a>", unsafe_allow_html=True)
st.subheader("⬇️ Download Data")
if len(df_flt):
    csv = df_flt.to_csv(index=False).encode("utf-8")
    st.download_button("Download filtered data (CSV)", csv, "filtered_data.csv", "text/csv")
else:
    st.warning("No data to download.")

st.markdown("[🔝 Back to Top](#top)")
