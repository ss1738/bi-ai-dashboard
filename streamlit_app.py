# AI BI Dashboard – Streamlit (Clean Rewrite)
# -------------------------------------------------
# Save as: streamlit_app.py
#
# Optional: .streamlit/secrets.toml for Google Form integration
# [WAITLIST]
# google_form_iframe_url = "https://docs.google.com/forms/d/e/<FORM_ID>/viewform?embedded=true"
# use_direct_post = true
# form_id = "<FORM_ID>"
# email_entry_id = "entry.123456"
# name_entry_id  = "entry.654321"
# extra_entry_id = "entry.135792"
# -------------------------------------------------

import io
import os
import json
from dataclasses import dataclass
from io import BytesIO
from typing import Optional, List

import numpy as np
import pandas as pd

import streamlit as st
import altair as alt

# ---- XLSX export helper ----
def df_to_xlsx_bytes(sheets: dict) -> bytes:
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="xlsxwriter") as writer:
        for name, d in (sheets or {}).items():
            if d is None:
                continue
            try:
                # allow empty frames but write headers
                (d if isinstance(d, pd.DataFrame) else pd.DataFrame(d)).to_excel(writer, index=False, sheet_name=(name or "sheet")[:31])
            except Exception:
                try:
                    pd.DataFrame(d).to_excel(writer, index=False, sheet_name=(name or "sheet")[:31])
                except Exception:
                    pass
    return bio.getvalue()

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.holtwinters import ExponentialSmoothing

try:
    import requests
except Exception:
    requests = None

st.set_page_config(
    page_title="AI BI Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------
# Light CSS polish
# ----------------------------
st.markdown(
    """
    <style>
      .metric-card {background: #fafafa; border: 1px solid #eee; padding: 14px 16px; border-radius: 14px;}
      .small-muted {color: #666; font-size: 12px;}
      .stTabs [data-baseweb="tab-list"] {gap: 6px;}
      .stTabs [data-baseweb="tab"] {background: #f7f7f9; padding: 10px 12px; border-radius: 10px; border: 1px solid #eee;}
      .stTabs [aria-selected="true"] {background: white; border-color: #ddd;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# WAITLIST config via secrets
# ----------------------------
@dataclass
class WaitlistConfig:
    google_form_iframe_url: Optional[str] = None
    use_direct_post: bool = False
    form_id: Optional[str] = None
    email_entry_id: Optional[str] = None
    name_entry_id: Optional[str] = None
    extra_entry_id: Optional[str] = None


def get_waitlist_config() -> WaitlistConfig:
    cfg = WaitlistConfig()
    try:
        w = st.secrets.get("WAITLIST", {})
        cfg.google_form_iframe_url = w.get("google_form_iframe_url")
        cfg.use_direct_post = bool(w.get("use_direct_post", False))
        cfg.form_id = w.get("form_id")
        cfg.email_entry_id = w.get("email_entry_id")
        cfg.name_entry_id = w.get("name_entry_id")
        cfg.extra_entry_id = w.get("extra_entry_id")
    except Exception:
        pass
    return cfg

WAITLIST_CFG = get_waitlist_config()

# ----------------------------
# Helpers
# ----------------------------

def _find_datetime_column(df: pd.DataFrame) -> Optional[str]:
    for col in df.columns:
        if np.issubdtype(df[col].dtype, np.datetime64):
            return col
    for guess in ["date", "timestamp", "time", "datetime"]:
        if guess in df.columns:
            try:
                df[guess] = pd.to_datetime(df[guess])
                return guess
            except Exception:
                continue
    return None


@st.cache_data(show_spinner=False)
def load_data(file) -> pd.DataFrame:
    if file is None:
        return pd.DataFrame()
    name = getattr(file, "name", "uploaded")
    try:
        if name.lower().endswith(".csv"):
            df = pd.read_csv(file)
        elif name.lower().endswith(".parquet"):
            df = pd.read_parquet(file)
        elif name.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(file)
        else:
            df = pd.read_csv(file)
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        return pd.DataFrame()
    # parse likely datetime columns
    for col in df.columns:
        if df[col].dtype == object and any(k in col.lower() for k in ["date", "time"]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass
    return df


@st.cache_data(show_spinner=False)
def make_demo_data(n_days: int = 365, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = pd.Timestamp.today().normalize() - pd.Timedelta(days=n_days)
    dates = pd.date_range(start, periods=n_days, freq="D")
    categories = ["Online", "Retail", "Wholesale"]
    regions = ["EMEA", "APAC", "AMER"]

    data: List[dict] = []
    base = 1000
    season = np.sin(np.linspace(0, 6 * np.pi, n_days)) * 150
    trend = np.linspace(0, 250, n_days)
    for i, d in enumerate(dates):
        for c in categories:
            for r in regions:
                noise = rng.normal(0, 80)
                amount = max(0, base + season[i] + trend[i] + noise + rng.normal(0, 30))
                units = max(1, int(10 + season[i]/20 + rng.normal(0, 3)))
                price = amount / units
                data.append({
                    "date": d,
                    "channel": c,
                    "region": r,
                    "revenue": round(amount, 2),
                    "units": units,
                    "price": round(price, 2),
                })
    return pd.DataFrame(data)


def kpi_card(label: str, value: str, helptext: str = "") -> None:
    with st.container(border=True):
        st.markdown(
            f"""
            <div class='metric-card'>
              <h4>{label}</h4>
              <h2>{value}</h2>
              <p class='small-muted'>{helptext}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.title("AI BI Dashboard")
with st.sidebar:
    st.caption("Upload data or use demo. We auto-detect date & fields.")
    data_src = st.radio("Data source", ["Use demo data", "Upload CSV/Parquet/Excel"], index=0, key="src")

    if data_src == "Upload CSV/Parquet/Excel":
        upl = st.file_uploader("Upload a file", type=["csv", "parquet", "xlsx", "xls"], key="upl")
        df = load_data(upl)
        if df.empty:
            st.info("No data yet – using demo until a valid file is uploaded.")
            df = make_demo_data()
    else:
        df = make_demo_data()

    date_col = _find_datetime_column(df)
    if not date_col:
        try:
            df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
            date_col = df.columns[0]
        except Exception:
            date_col = None

    if date_col:
        min_d, max_d = pd.to_datetime(df[date_col]).min(), pd.to_datetime(df[date_col]).max()
        date_range = st.date_input(
            "Date range",
            value=(min_d.date(), max_d.date()),
            min_value=min_d.date(),
            max_value=max_d.date(),
            key="dr",
        )
    else:
        date_range = None

    cat_cols = [c for c in df.columns if df[c].dtype == object]
    cat_filters = {}
    for i, c in enumerate(cat_cols[:3]):
        vals = sorted(df[c].dropna().unique().tolist())[:50]
        sel = st.multiselect(f"Filter {c}", vals, default=vals, key=f"cat_{i}")
        cat_filters[c] = sel

    st.divider()
    st.subheader("Quick actions")
    try:
        csv_sidebar = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download filtered data (CSV)", csv_sidebar, file_name="data_filtered.csv", key="dl_sidebar")
        # JSON + XLSX mirrors
        try:
            st.download_button("Download filtered (JSON)", df.to_json(orient="records").encode("utf-8"), file_name="data_filtered.json", key="dl_json_sidebar")
        except Exception:
            pass
        try:
            anoms = st.session_state.get("__last_anomalies_df")
            fcast = st.session_state.get("__last_forecast_df")
            xlsx_bytes = df_to_xlsx_bytes({
                "data_filtered": df,
                "anomalies": anoms if isinstance(anoms, pd.DataFrame) else pd.DataFrame(),
                "forecast": fcast if isinstance(fcast, pd.DataFrame) else pd.DataFrame(),
            })
            st.download_button("Export (XLSX)", xlsx_bytes, file_name="export.xlsx", key="dl_xlsx_sidebar")
        except Exception:
            st.caption("(Install XlsxWriter for XLSX export)")
    except Exception:
        pass

# Apply filters
if date_col and date_range is not None and len(date_range) == 2:
    start_d, end_d = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    df = df[(pd.to_datetime(df[date_col]) >= start_d) & (pd.to_datetime(df[date_col]) <= end_d)]

for c, allowed in (cat_filters or {}).items():
    if allowed:
        df = df[df[c].isin(allowed)]

num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

# ----------------------------
# Tabs
# ----------------------------
TABS = st.tabs([
    "📊 Dashboard",
    "🧩 Segmentation",
    "🚨 Anomalies",
    "📈 Forecast",
    "🧠 AI Insights",
    "🕒 Early Access Waitlist",
])

# ----------------------------
# TAB 1: Dashboard
# ----------------------------
with TABS[0]:
    st.markdown("### Overview")

    # ===== Smart Chart Carousel =====
    st.markdown("#### 🔄 Smart Chart Carousel")
    rev_col_guess = next((c for c in ["revenue","sales","amount","value"] if c in df.columns), (num_cols[0] if num_cols else None))
    dims = [c for c in df.columns if df[c].dtype == object]
    colcc1, colcc2, colcc3, colcc4 = st.columns([2,2,1,1])
    metric_sel = colcc1.selectbox("Metric", [rev_col_guess] + [c for c in num_cols if c != rev_col_guess] if rev_col_guess else num_cols, index=0 if rev_col_guess else 0, key="cc_metric")
    dim_sel = colcc2.selectbox("Breakdown (optional)", ["(none)"] + dims[:3], index=0, key="cc_dim")
    if "carousel_idx" not in st.session_state:
        st.session_state.carousel_idx = 0
    if colcc3.button("◀", key="cc_prev"):
        st.session_state.carousel_idx = (st.session_state.carousel_idx - 1) % 4
    if colcc4.button("▶", key="cc_next"):
        st.session_state.carousel_idx = (st.session_state.carousel_idx + 1) % 4

    chart_obj = None
    if not df.empty and metric_sel in df.columns:
        if date_col:
            base = df.assign(_d=pd.to_datetime(df[date_col]).dt.to_period("D").dt.start_time)
            agg = base.groupby(["_d"] + ([dim_sel] if dim_sel != "(none)" else []))[metric_sel].sum().reset_index().rename(columns={"_d":"date"})
        else:
            agg = df.copy()

        idx = st.session_state.carousel_idx
        if idx == 0:  # line
            chart_obj = alt.Chart(agg).mark_line().encode(
                x=alt.X("date:T" if "date" in agg.columns else agg.columns[0]+":Q", title="Date" if "date" in agg.columns else agg.columns[0]),
                y=alt.Y(f"{metric_sel}:Q"),
                color=alt.Color(f"{dim_sel}:N") if dim_sel != "(none)" else alt.value("steelblue"),
                tooltip=list(agg.columns),
            ).properties(height=320)
        elif idx == 1:  # bar
            chart_obj = alt.Chart(agg).mark_bar().encode(
                x=alt.X(f"{dim_sel}:N", title=dim_sel) if dim_sel != "(none)" else alt.X("date:T", title="Date"),
                y=alt.Y(f"{metric_sel}:Q"),
                color=alt.Color(f"{dim_sel}:N") if dim_sel != "(none)" else alt.value("steelblue"),
                tooltip=list(agg.columns),
            ).properties(height=320)
        elif idx == 2:  # area
            chart_obj = alt.Chart(agg).mark_area(opacity=0.4).encode(
                x=alt.X("date:T" if "date" in agg.columns else agg.columns[0]+":Q"),
                y=alt.Y(f"{metric_sel}:Q"),
                color=alt.Color(f"{dim_sel}:N") if dim_sel != "(none)" else alt.value("steelblue"),
                tooltip=list(agg.columns),
            ).properties(height=320)
        else:  # heatmap
            if dim_sel == "(none)" or "date" not in agg.columns:
                st.info("Heatmap needs a date and a breakdown. Choose a categorical breakdown.")
            else:
                chart_obj = alt.Chart(agg).mark_rect().encode(
                    x=alt.X("date:T"), y=alt.Y(f"{dim_sel}:N"), color=alt.Color(f"{metric_sel}:Q"), tooltip=list(agg.columns)
                ).properties(height=320)

    if chart_obj is not None:
        st.altair_chart(chart_obj, use_container_width=True)
        with st.popover("Export chart"):
            st.caption("If PNG unavailable on your deploy, you'll get Vega-Lite JSON instead.")
            png_bytes = None
            try:
                from altair_saver import save as alt_save  # optional
                bio = io.BytesIO()
                alt_save(chart_obj, bio, format="png", method="vl-convert")
                png_bytes = bio.getvalue()
            except Exception:
                png_bytes = None
            if png_bytes:
                st.download_button("Download PNG", png_bytes, file_name="chart.png")
            st.download_button("Download Vega-Lite JSON", chart_obj.to_json().encode("utf-8"), file_name="chart.vl.json")

    # KPIs (after carousel to keep top of page compact)
    if date_col is None:
        st.warning("No datetime column detected. Some charts may be limited.")
    else:
        by_date = (
            df.assign(_date=pd.to_datetime(df[date_col]).dt.date)
              .groupby("_date")
              .agg({c: "sum" for c in num_cols})
              .reset_index()
              .rename(columns={"_date": "date"})
        )
        rev_col = next((c for c in ["revenue", "sales", "amount", "value"] if c in df.columns), None)
        units_col = "units" if "units" in df.columns else None
        if rev_col is None and num_cols:
            rev_col = num_cols[0]
        c1, c2, c3, c4 = st.columns(4)
        if rev_col in by_date.columns:
            total_rev = float(by_date[rev_col].sum())
            last = by_date.iloc[-1][rev_col]
            prev = by_date.iloc[-2][rev_col] if len(by_date) > 1 else last
            mom = ((last - prev) / prev * 100) if prev else np.nan
            with c1: kpi_card("Total Revenue", f"£{total_rev:,.0f}", f"Last vs prior: {mom:+.1f}%" if not np.isnan(mom) else "")
        else:
            with c1: kpi_card("Total (first numeric)", f"{by_date[num_cols[0]].sum():,.0f}")
        if units_col and units_col in by_date.columns:
            with c2: kpi_card("Units Sold", f"{int(by_date[units_col].sum()):,}")
        else:
            with c2:
                any_second = num_cols[1] if len(num_cols) > 1 else None
                kpi_card("Secondary Total", f"{by_date[any_second].sum():,.0f}" if any_second else f"Rows: {len(df):,}")
        if rev_col in by_date.columns and units_col and units_col in by_date.columns:
            avg_price = (by_date[rev_col].sum() / max(1, by_date[units_col].sum()))
            with c3: kpi_card("Avg Price", f"£{avg_price:,.2f}")
        else:
            with c3: kpi_card("Columns", f"{len(df.columns):,}")
        with c4: kpi_card("Active Days", f"{by_date['date'].nunique():,}")

    # Peek at data
    with st.expander("Peek at data"):
        st.dataframe(df.head(500), use_container_width=True)

    # Download section with anchor
    st.markdown("<div id='download-data'></div>", unsafe_allow_html=True)
    st.subheader("Download Data")
    st.caption("Exports the CURRENTLY FILTERED dataset shown above.")
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download filtered data (CSV)", csv_bytes, file_name="data_filtered.csv", key="dl_main")

    # JSON export (filtered dataset)
    try:
        st.download_button(
            "Download filtered data (JSON)",
            df.to_json(orient="records").encode("utf-8"),
            file_name="data_filtered.json",
            key="dl_json_main",
        )
    except Exception:
        pass

    # Multi-sheet XLSX: filtered + anomalies + forecast
    anoms = st.session_state.get("__last_anomalies_df")
    fcast = st.session_state.get("__last_forecast_df")
    try:
        xlsx_bytes = df_to_xlsx_bytes({
            "data_filtered": df,
            "anomalies": anoms if isinstance(anoms, pd.DataFrame) else pd.DataFrame(),
            "forecast": fcast if isinstance(fcast, pd.DataFrame) else pd.DataFrame(),
        })
        st.download_button("Download full export (XLSX)", xlsx_bytes, file_name="export.xlsx", key="dl_xlsx_main")
    except Exception as _e:
        st.info("Add 'XlsxWriter' to requirements.txt for XLSX export if this button doesn't work.")

    # Copy deep-link to this section
    st.button("🔗 Copy link to this section", key="copy_dl_btn")
    st.components.v1.html(
        """
        <script>
        (function(){
          const findBtn = () => Array.from(window.parent.document.querySelectorAll('button')).find(b => /Copy link to this section/.test(b.textContent));
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
        """,
        height=0,
    )

    st.components.v1.html(
        """
        <script>
        setTimeout(() => {
          try { if (window.location.hash === "#download-data") {
            const el = document.getElementById("download-data");
            if (el) el.scrollIntoView({behavior: "smooth", block: "start"}); }
          } catch (e) {}
        }, 400);
        </script>
        """,
        height=0,
    )

    # Scenario Simulator (save & compare)
    with st.expander("🧪 Scenario Simulator (what-if) — save & compare", expanded=False):
        colA, colB, colC = st.columns(3)
        price_uplift = colA.slider("Price change %", -30, 30, 0, step=1)
        units_uplift = colB.slider("Units change %", -50, 50, 0, step=1)
        mktg_boost   = colC.slider("Marketing lift % (revenue)", 0, 50, 0, step=1)

        base_rev_col = next((c for c in ["revenue","sales","amount","value"] if c in df.columns), None)
        base_units_col = "units" if "units" in df.columns else None
        if base_rev_col:
            sim = df.copy()
            pr = 1 + price_uplift/100
            un = 1 + units_uplift/100
            mk = 1 + mktg_boost/100
            if base_units_col:
                approx_price = (sim[base_rev_col] / sim[base_units_col]).replace([np.inf,-np.inf], np.nan).fillna(1.0)
                sim_rev = approx_price * pr * sim[base_units_col] * un * mk
            else:
                sim_rev = sim[base_rev_col] * pr * mk
            sim_total = sim_rev.sum()
            base_total = df[base_rev_col].sum()
            uplift_pct = ((sim_total - base_total) / base_total * 100) if base_total else 0
            st.metric("Simulated Total Revenue", f"£{sim_total:,.0f}", f"{uplift_pct:+.1f}% vs base")

            if "scenarios" not in st.session_state:
                st.session_state.scenarios = []
            c1, c2 = st.columns([1,2])
            name = c1.text_input("Scenario name", value=f"Scenario {len(st.session_state.scenarios)+1}")
            if c1.button("Save scenario", key="save_scn"):
                st.session_state.scenarios.append({
                    "name": name,
                    "price%": price_uplift,
                    "units%": units_uplift,
                    "mktg%": mktg_boost,
                    "total": float(sim_total),
                })
                st.success(f"Saved {name}")
            if st.session_state.scenarios:
                st.markdown("**Saved scenarios**")
                st.dataframe(pd.DataFrame(st.session_state.scenarios))
                s1, s2 = st.columns(2)
                a = s1.selectbox("Compare A", [s["name"] for s in st.session_state.scenarios], index=0, key="cmpA")
                b = s2.selectbox("Compare B", [s["name"] for s in st.session_state.scenarios], index=min(1, len(st.session_state.scenarios)-1), key="cmpB")
                if a and b and a != b:
                    A = next(s for s in st.session_state.scenarios if s["name"]==a)
                    B = next(s for s in st.session_state.scenarios if s["name"]==b)
                    diff = B["total"] - A["total"]
                    st.info(f"**{b} vs {a}**: Δ £{diff:,.0f} ({(diff/max(1,A['total']))*100:+.1f}%)")

    # Region x Time heatmap
    st.markdown("---")
    st.markdown("### 🌍 Region × Time heatmap (demo)")
    if date_col and "region" in df.columns:
        gran_geo = st.selectbox("Granularity", ["M","W","D"], index=0, key="geo_gran")
        rev_col_geo = next((c for c in ["revenue","sales","amount","value"] if c in df.columns), None) or (num_cols[0] if num_cols else None)
        geo = (df.assign(_p=pd.to_datetime(df[date_col]).dt.to_period(gran_geo).dt.start_time)
                 .groupby(["region","_p"])[rev_col_geo].sum().reset_index().rename(columns={"_p":"date","region":"Region", f"{rev_col_geo}":"Value"}))
        heat = alt.Chart(geo).mark_rect().encode(
            x=alt.X("date:T", title="Period"), y=alt.Y("Region:N"), color=alt.Color("Value:Q"),
            tooltip=["Region","date:T", alt.Tooltip("Value:Q", format=",.0f")]
        ).properties(height=220)
        st.altair_chart(heat, use_container_width=True)
    else:
        st.info("Need a datetime column and a 'region' column for the heatmap.")

# ----------------------------
# TAB 2: Segmentation
# ----------------------------
with TABS[1]:
    st.markdown("### Customer/Product Segmentation")
    if not num_cols:
        st.warning("No numeric columns available for clustering.")
    else:
        with st.form("seg_form"):
            feats = st.multiselect("Features for clustering", num_cols, default=num_cols[: min(4, len(num_cols))])
            k = st.slider("Number of clusters (k)", 2, 10, 4)
            submitted = st.form_submit_button("Run KMeans")
        if submitted and feats:
            try:
                X = df[feats].dropna()
                scaler = StandardScaler()
                Xs = scaler.fit_transform(X)
                km = KMeans(n_clusters=k, n_init=20, random_state=42)
                labels = km.fit_predict(Xs)
                prof = pd.DataFrame(X, index=X.index)
                prof["cluster"] = labels
                profile = prof.groupby("cluster").agg("mean").round(2)
                st.success("Segmentation complete.")
                st.dataframe(profile, use_container_width=True)
                pca = PCA(n_components=2, random_state=42)
                pts = pca.fit_transform(Xs)
                plot_df = pd.DataFrame({"pc1": pts[:, 0], "pc2": pts[:, 1], "cluster": labels.astype(str)})
                sc = alt.Chart(plot_df).mark_circle(size=60, opacity=0.6).encode(
                    x="pc1:Q", y="pc2:Q", color="cluster:N",
                    tooltip=["cluster", alt.Tooltip("pc1:Q", format=".2f"), alt.Tooltip("pc2:Q", format=".2f")],
                ).properties(height=420)
                st.altair_chart(sc, use_container_width=True)
                out = df.copy()
                out.loc[X.index, "cluster"] = labels
                st.download_button("Download segmented data (CSV)", out.to_csv(index=False).encode("utf-8"), file_name="segments.csv", key="dl_segments")
            except Exception as e:
                st.error(f"Segmentation failed: {e}")

# ----------------------------
# TAB 3: Anomalies
# ----------------------------
with TABS[2]:
    st.markdown("### Anomaly Detection")
    st.caption("Tip: Use single-metric or multivariate scoring. Z-score highlights >2.5σ deviations.")
    if date_col is None:
        st.warning("Needs a datetime column to aggregate over time.")
    else:
        target_col = st.selectbox("Target metric", options=[c for c in ["revenue", "sales", "amount", "value"] if c in df.columns] or num_cols, key="anom_target")
        gran = st.selectbox("Granularity", ["D", "W", "M"], index=0, help="Aggregate by Day/Week/Month", key="anom_gran")
        method = st.selectbox("Method", ["IsolationForest", "Z-Score"], index=0, key="anom_method")

        agg = (
            df.assign(_date=pd.to_datetime(df[date_col]).dt.to_period(gran).dt.start_time)
              .groupby("_date")[target_col].sum()
              .reset_index().rename(columns={"_date": "date", target_col: "y"})
        )

        if len(agg) < 10:
            st.warning("Not enough data points after aggregation.")
        else:
            try:
                if method == "IsolationForest":
                    X = agg[["y"]].values
                    iso = IsolationForest(contamination=0.05, random_state=42)
                    preds = iso.fit_predict(X)
                    scores = iso.decision_function(X)
                    agg["anomaly"] = (preds == -1)
                    agg["score"] = scores
                else:
                    m, s = agg["y"].mean(), agg["y"].std()
                    z = (agg["y"] - m) / (s if s else 1)
                    agg["anomaly"] = z.abs() > 2.5
                    agg["score"] = -z.abs()
                st.success("Anomaly scoring complete.")
                base = alt.Chart(agg).encode(x="date:T")
                line = base.mark_line().encode(y="y:Q")
                pts = base.mark_circle(size=80, opacity=0.9).encode(
                    y="y:Q",
                    color=alt.condition("datum.anomaly", alt.value("crimson"), alt.value("steelblue")),
                    tooltip=["date:T", alt.Tooltip("y:Q", format=",.0f"), "anomaly:N", alt.Tooltip("score:Q", format=".3f")],
                )
                st.altair_chart(line + pts, use_container_width=True)
                st.dataframe(agg[agg["anomaly"]], use_container_width=True)
                st.download_button("Download anomalies (CSV)", agg[agg["anomaly"]].to_csv(index=False).encode("utf-8"), file_name="anomalies.csv", key="dl_anoms")
                # keep latest for XLSX export
                st.session_state["__last_anomalies_df"] = agg.copy()
            except Exception as e:
                st.error(f"Anomaly detection failed: {e}")

        # Multivariate
        st.markdown("---")
        st.markdown("#### Multivariate anomalies (IsolationForest across multiple KPIs)")
        with st.form("mv_anom"):
            feats = st.multiselect("Features for anomaly scoring", num_cols, default=[c for c in num_cols if c != "price"][:3])
            submit_mv = st.form_submit_button("Score multivariate anomalies")
        if submit_mv and feats:
            try:
                agg_mv = (df.assign(_d=pd.to_datetime(df[date_col]).dt.to_period(gran).dt.start_time)
                            .groupby("_d")[feats].sum().reset_index().rename(columns={"_d":"date"}))
                X = agg_mv[feats].fillna(method="ffill").fillna(method="bfill").values
                iso = IsolationForest(contamination=0.05, random_state=42)
                pred = iso.fit_predict(X)
                agg_mv["anomaly"] = (pred == -1)
                st.success("Multivariate anomaly scoring complete.")
                st.dataframe(agg_mv[agg_mv["anomaly"]], use_container_width=True)
                for f in feats[:2]:
                    st.altair_chart(alt.Chart(agg_mv).mark_line().encode(x="date:T", y=f"{f}:Q"), use_container_width=True)
            except Exception as e:
                st.error(f"Multivariate anomaly failed: {e}")

# ----------------------------
# TAB 4: Forecast
# ----------------------------
with TABS[3]:
    st.markdown("### Forecasting (Exponential Smoothing)")
    if date_col is None:
        st.warning("Needs a datetime column.")
    else:
        target_col = st.selectbox("Target metric", options=[c for c in ["revenue", "sales", "amount", "value"] if c in df.columns] or num_cols, key="fc_target")
        gran = st.selectbox("Granularity", ["D", "W", "M"], index=0, key="fc_gran")
        horizon = st.slider("Forecast horizon (periods)", 7 if gran == "D" else 8, 60, 30, key="fc_h")

        ts = (
            df.assign(_date=pd.to_datetime(df[date_col]).dt.to_period(gran).dt.start_time)
              .groupby("_date")[target_col].sum()
        )
        # infer frequency for statsmodels
        freq_series = pd.Series(pd.to_datetime(df[date_col]).dt.to_period(gran).dt.start_time).sort_values()
        try:
            ts = ts.asfreq(pd.infer_freq(freq_series) or gran)
        except Exception:
            ts = ts.asfreq(gran)

        if ts.isna().any():
            ts = ts.fillna(method="ffill").fillna(method="bfill")

        if len(ts) < 12:
            st.warning("Need at least 12 periods for a sensible fit.")
        else:
            try:
                seasonal = {"D": 7, "W": 52, "M": 12}[gran]
                model = ExponentialSmoothing(ts, trend="add", seasonal="add", seasonal_periods=seasonal)
                fit = model.fit(optimized=True, use_brute=True)
                fc = fit.forecast(horizon)
                resid = fit.resid
                sigma = resid.std() if hasattr(resid, "std") else float(np.std(resid))
                ci_hi = fc + 1.96 * sigma
                ci_lo = fc - 1.96 * sigma
                fdf = pd.DataFrame({"date": fc.index, "forecast": fc.values, "lo": ci_lo.values, "hi": ci_hi.values})
                hist = ts.reset_index(); hist.columns = ["date", "y"]
                line_hist = alt.Chart(hist).mark_line().encode(x="date:T", y="y:Q")
                line_fc = alt.Chart(fdf).mark_line(strokeDash=[4,3]).encode(x="date:T", y="forecast:Q", tooltip=["date:T", alt.Tooltip("forecast:Q", format=",.0f")])
                band = alt.Chart(fdf).mark_area(opacity=0.2).encode(x="date:T", y="lo:Q", y2="hi:Q")
                st.altair_chart(line_hist + band + line_fc, use_container_width=True)
                st.download_button("Download forecast (CSV)", fdf.to_csv(index=False).encode("utf-8"), file_name="forecast.csv", key="dl_fc")
                # keep latest for XLSX export
                st.session_state["__last_forecast_df"] = fdf.copy()
                st.success("Forecast ready.")
            except Exception as e:
                st.error(f"Forecasting failed: {e}")

# ----------------------------
# TAB 5: AI Insights (heuristic)
# ----------------------------
with TABS[4]:
    st.markdown("### AI-ish Insights (No API Key)")
    st.caption("Heuristic insights + quick answers. No API needed.")

    if df.empty:
        st.warning("No data loaded.")
    else:
        rev_col = next((c for c in ["revenue", "sales", "amount", "value"] if c in df.columns), None)
        if rev_col is None and num_cols:
            rev_col = num_cols[0]

        # Insight Cards
        cards = []
        try:
            for c in [col for col in df.columns if df[col].dtype == object][:2]:
                top = df.groupby(c)[rev_col].sum().sort_values(ascending=False)
                if len(top) >= 2:
                    a, b = top.index[0], top.index[1]
                    uplift_pct = (top.iloc[0] - top.iloc[1]) / (top.iloc[1] if top.iloc[1] else 1) * 100
                    cards.append(("🚀 Growth Driver", f"Focus on **{a}** in **{c}** – leads by {uplift_pct:,.1f}% vs {b}."))
        except Exception:
            pass
        if date_col and rev_col in df.columns:
            tmp = df.assign(_d=pd.to_datetime(df[date_col]).dt.to_period("M").dt.start_time)
            m = tmp.groupby("_d")[rev_col].sum()
            if len(m) >= 3:
                gr1 = (m.iloc[-1] - m.iloc[-2]) / (m.iloc[-2] if m.iloc[-2] else 1) * 100
                gr2 = (m.iloc[-2] - m.iloc[-3]) / (m.iloc[-3] if m.iloc[-3] else 1) * 100
                dir_txt = "accelerating" if gr1 > gr2 else "slowing"
                emo = "🚀" if gr1>0 else "⚠️"
                cards.append((f"{emo} Momentum", f"Growth is **{dir_txt}**: last month {gr1:,.1f}% vs prior {gr2:,.1f}%."))
        if not cards:
            cards.append(("ℹ️ Note", "Define a revenue/units column for sharper insights."))

        cols_cards = st.columns(min(3, len(cards))) if cards else []
        for i, (title, text) in enumerate(cards):
            with cols_cards[i % len(cols_cards)]:
                with st.container(border=True):
                    st.markdown(f"""
                    **{title}**  
                    {text}
                    """)

        # Top Movers with sparklines
        st.markdown("---")
        st.markdown("### 📈 Top Movers (period-over-period)")
        dim_cols = [c for c in df.columns if df[c].dtype == object]
        if date_col and rev_col and dim_cols:
            gran = st.selectbox("Compare by", ["W","M"], index=0, key="movers_gran")
            dim  = st.selectbox("Dimension", dim_cols[:3] or dim_cols, index=0, key="movers_dim")
            g = (df.assign(_p=pd.to_datetime(df[date_col]).dt.to_period(gran).dt.start_time)
                   .groupby([dim,"_p"])[rev_col].sum().reset_index().sort_values("_p"))
            g["prev"] = g.groupby(dim)[rev_col].shift(1)
            g["pct_change"] = (g[rev_col] - g["prev"]) / g["prev"] * 100
            last_period = g["_p"].max()
            movers = g[g["_p"]==last_period].dropna(subset=["pct_change"]).sort_values("pct_change", ascending=False)
            col1, col2 = st.columns(2)
            col1.markdown("**Top Gainers**")
            col1.dataframe(movers.head(10)[[dim, rev_col, "pct_change"]].rename(columns={rev_col:"value"}), use_container_width=True)
            col2.markdown("**Top Decliners**")
            col2.dataframe(movers.tail(10)[[dim, rev_col, "pct_change"]].rename(columns={rev_col:"value"}), use_container_width=True)
            top6 = pd.concat([movers.head(3)[dim], movers.tail(3)[dim]]).unique().tolist()
            if top6:
                small = g[g[dim].isin(top6)]
                spark = alt.Chart(small).mark_line().encode(
                    x=alt.X("_p:T", title=""), y=alt.Y(f"{rev_col}:Q", title=""), facet=alt.Facet(f"{dim}:N", columns=3),
                    tooltip=[dim, "_p:T", alt.Tooltip(f"{rev_col}:Q", format=",.0f")]
                ).properties(height=120)
                st.altair_chart(spark, use_container_width=True)
        else:
            st.info("Need a date column, a numeric metric, and a categorical column.")

        # Quick Q&A
        st.divider()
        st.markdown("**Ask a quick question** (e.g., *Which region had highest revenue?*)")
        q = st.text_input("Question", key="qa_q")
        if q:
            try:
                answer = ""
                ql = q.lower()
                if any(k in ql for k in ["highest","top","max"]) and rev_col:
                    target_cat = None
                    for c in [col for col in df.columns if df[col].dtype == object]:
                        if c.lower() in ql:
                            target_cat = c
                            break
                    if target_cat is None and [col for col in df.columns if df[col].dtype == object]:
                        target_cat = [col for col in df.columns if df[col].dtype == object][0]
                    top = df.groupby(target_cat)[rev_col].sum().sort_values(ascending=False)
                    answer = f"**{top.index[0]}** leads in {target_cat} with £{top.iloc[0]:,.0f}."
                elif any(k in ql for k in ["trend","growth"]) and date_col and rev_col:
                    m = df.assign(_d=pd.to_datetime(df[date_col]).dt.to_period("M").dt.start_time).groupby("_d")[rev_col].sum()
                    answer = f"Last 3 months: {', '.join([f'£{v:,.0f}' for v in m.tail(3)])}."
                else:
                    answer = "Try *highest/lowest by [category]* or *trend/growth* questions."
                st.info(answer)
            except Exception as e:
                st.error(f"Could not answer: {e}")

# ----------------------------
# TAB 6: Early Access Waitlist
# ----------------------------
with TABS[5]:
    st.markdown("### Early Access Waitlist")
    st.caption("Choose one: **iframe embed** or **Direct POST** to Google Forms, plus a simple local CSV capture.")

    # A) IFRAME EMBED – simplest
    with st.expander("Option A – Embed your Google Form (iframe)", expanded=bool(WAITLIST_CFG.google_form_iframe_url)):
        form_url = st.text_input("Google Form embed URL", value=WAITLIST_CFG.google_form_iframe_url or "", placeholder="https://docs.google.com/forms/d/e/<FORM_ID>/viewform?embedded=true", key="g_iframe_url")
        if form_url:
            st.components.v1.iframe(src=form_url, height=700, scrolling=True)
            st.success("Embedded form displayed above.")
        else:
            st.info("Paste the Google Form 'embedded' URL to show it here.")

    # B) DIRECT POST – capture email/name then submit to Google Forms
    with st.expander("Option B – Direct submit to Google Form (email capture)", expanded=WAITLIST_CFG.use_direct_post):
        if requests is None:
            st.error("The 'requests' library is required for direct submit. Add 'requests' to requirements.txt.")
        else:
            colA, colB = st.columns(2)
            with colA:
                form_id = st.text_input("Google Form ID", value=WAITLIST_CFG.form_id or "", key="g_form_id")
                email_entry = st.text_input("Email entry id (entry.xxxxx)", value=WAITLIST_CFG.email_entry_id or "", key="g_email_entry")
                name_entry = st.text_input("Name entry id (optional)", value=WAITLIST_CFG.name_entry_id or "", key="g_name_entry")
                extra_entry = st.text_input("Additional free-text entry id (optional)", value=WAITLIST_CFG.extra_entry_id or "", key="g_extra_entry")
            with colB:
                st.markdown("**What users will fill**")
                name_val = st.text_input("Your name (optional)", key="wl_name")
                email_val = st.text_input("Your email", placeholder="you@example.com", key="wl_email")
                extra_val = st.text_area("Anything else? (optional)", key="wl_extra")
                consent = st.checkbox("I agree to be contacted about Early Access.", key="consent_direct")
                submit = st.button("Join Waitlist ✅", key="wl_join")

            if submit:
                if not (form_id and email_entry and email_val and consent):
                    st.error("Missing required fields: form id, email entry id, email, and consent.")
                else:
                    try:
                        url = f"https://docs.google.com/forms/d/e/{form_id}/formResponse"
                        payload = {email_entry: email_val}
                        if name_entry and name_val:
                            payload[name_entry] = name_val
                        if extra_entry and extra_val:
                            payload[extra_entry] = extra_val
                        payload.update({"fvv": 1, "partialResponse": [], "pageHistory": 0, "fbzx": "-1234567890"})
                        resp = requests.post(url, data=payload, timeout=10)
                        if resp.status_code == 200:
                            st.success("You're on the waitlist! Check your inbox for a confirmation shortly.")
                            st.toast("Waitlist joined! 🎉")
                        else:
                            st.warning(f"Google Form returned status {resp.status_code}. We'll also capture locally as backup.")
                            row = {"ts": pd.Timestamp.utcnow().isoformat(), "name": name_val, "email": email_val, "extra": extra_val}
                            key_local = "_local_waitlist"
                            wl = st.session_state.get(key_local, [])
                            wl.append(row)
                            st.session_state[key_local] = wl
                            st.success("Saved to local session backup.")
                    except Exception as e:
                        st.error(f"Submit failed: {e}")

    st.divider()
    st.markdown("#### Simple native (no Google) capture – CSV download backup")
    with st.form("native_capture"):
        n_name = st.text_input("Name (optional)", key="n_name")
        n_email = st.text_input("Email", key="n_email")
        n_notes = st.text_area("Notes (optional)", key="n_notes")
        n_ok = st.checkbox("I agree to be contacted about Early Access.", key="consent_native")
        n_submit = st.form_submit_button("Add to local list")
    if n_submit:
        if not (n_email and n_ok):
            st.error("Email and consent required.")
        else:
            row = {"ts": pd.Timestamp.utcnow().isoformat(), "name": n_name, "email": n_email, "notes": n_notes}
            key_local = "_native_waitlist"
            bag = st.session_state.get(key_local, [])
            bag.append(row)
            st.session_state[key_local] = bag
            st.success("Added.")

    bag = st.session_state.get("_native_waitlist", [])
    if bag:
        nd = pd.DataFrame(bag)
        st.dataframe(nd.tail(200), use_container_width=True)
        st.download_button("Download local waitlist CSV", nd.to_csv(index=False).encode("utf-8"), file_name="waitlist_local.csv", key="dl_local")

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.caption("AI BI Dashboard • MVP → GTM • Polished UI, error-handled, and Early Access ready.")
