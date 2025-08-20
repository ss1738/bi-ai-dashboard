# AI BI Dashboard — Streamlit (Report + Saved Views)
# Save as: streamlit_app.py
# Tip: add to requirements.txt (free):
# pandas
# numpy
# altair
# scikit-learn
# statsmodels
# XlsxWriter  # for XLSX export

import base64
from io import BytesIO
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# ----------------------------
# Small helpers
# ----------------------------

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


def make_demo_data(n_days: int = 365, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = pd.Timestamp.today().normalize() - pd.Timedelta(days=n_days)
    dates = pd.date_range(start, periods=n_days, freq="D")
    categories = ["Online", "Retail", "Wholesale"]
    regions = ["EMEA", "APAC", "AMER"]
    products = ["A","B","C","D"]
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
                    "product": rng.choice(products),
                    "revenue": round(amount, 2),
                    "units": units,
                    "price": round(price, 2),
                })
    return pd.DataFrame(data)


# ----------------------------
# Query params / Saved views
# ----------------------------

def parse_query_list(val: Optional[str]) -> Optional[List[str]]:
    if val is None:
        return None
    if isinstance(val, list):  # streamlit may give list
        val = val[0]
    val = str(val).strip()
    if not val:
        return None
    return [v for v in val.split(",") if v]


def get_query_filters():
    qp = st.query_params
    r = parse_query_list(qp.get("region"))
    c = parse_query_list(qp.get("channel"))
    return r, c


def set_query_filters(regions: Optional[List[str]], channels: Optional[List[str]]):
    # Build new dict while keeping other params
    qp = dict(st.query_params)
    if regions:
        qp["region"] = ",".join(regions)
    else:
        qp.pop("region", None)
    if channels:
        qp["channel"] = ",".join(channels)
    else:
        qp.pop("channel", None)
    st.query_params.clear()
    for k, v in qp.items():
        st.query_params[k] = v


def make_share_link(regions: Optional[List[str]], channels: Optional[List[str]], anchor: Optional[str] = None) -> str:
    base = st.get_option("browser.serverAddress") or ""
    # Try to reconstruct full URL (works on Streamlit Cloud behind proxy)
    # Fallback to relative path
    try:
        # In hosted envs, we cannot always know full origin; use location via JS in UI for copy.
        path = "?"
        parts = []
        if regions:
            parts.append("region=" + ",".join(regions))
        if channels:
            parts.append("channel=" + ",".join(channels))
        path += "&".join(parts) if parts else ""
        if anchor:
            path += f"#{anchor}"
        return path
    except Exception:
        return ""


# ----------------------------
# App shell
# ----------------------------
st.set_page_config(page_title="AI BI Dashboard", page_icon="📊", layout="wide")

st.sidebar.title("AI BI Dashboard")

# Data source
src = st.sidebar.radio("Data source", ["Use demo data", "Upload CSV/Parquet/Excel"], index=0)
if src == "Upload CSV/Parquet/Excel":
    upl = st.sidebar.file_uploader("Upload a file", type=["csv","parquet","xlsx","xls"])
    if upl is not None:
        try:
            if upl.name.lower().endswith(".csv"):
                df = pd.read_csv(upl)
            elif upl.name.lower().endswith(".parquet"):
                df = pd.read_parquet(upl)
            else:
                df = pd.read_excel(upl)
        except Exception as e:
            st.sidebar.error(f"Failed to read file: {e}")
            df = make_demo_data()
    else:
        df = make_demo_data()
else:
    df = make_demo_data()

# Basic date parsing
if "date" in df.columns and not np.issubdtype(df["date"].dtype, np.datetime64):
    try:
        df["date"] = pd.to_datetime(df["date"])
    except Exception:
        pass

# Filter controls
regions_all = sorted(df["region"].dropna().unique().tolist()) if "region" in df.columns else []
channels_all = sorted(df["channel"].dropna().unique().tolist()) if "channel" in df.columns else []
q_regions, q_channels = get_query_filters()

with st.sidebar.expander("Filters", expanded=True):
    sel_regions = st.multiselect("Region", regions_all, default=(q_regions or regions_all))
    sel_channels = st.multiselect("Channel", channels_all, default=(q_channels or channels_all))
    if st.button("Apply filters"):
        set_query_filters(sel_regions, sel_channels)
        st.rerun()

# Saved views management
if "saved_views" not in st.session_state:
    st.session_state.saved_views = {}
with st.sidebar.expander("Saved Views", expanded=False):
    name_new = st.text_input("Name this view")
    col_sv1, col_sv2 = st.columns([1,1])
    if col_sv1.button("Save current"):
        st.session_state.saved_views[name_new or f"View {len(st.session_state.saved_views)+1}"] = {
            "regions": sel_regions,
            "channels": sel_channels,
        }
        st.success("Saved.")
    if col_sv2.button("Clear all"):
        st.session_state.saved_views = {}
    if st.session_state.saved_views:
        st.markdown("**Your views**")
        for nm, params in st.session_state.saved_views.items():
            c1, c2, c3 = st.columns([1,1,2])
            if c1.button(f"Apply: {nm}", key=f"apply_{nm}"):
                set_query_filters(params.get("regions"), params.get("channels"))
                st.rerun()
            share = make_share_link(params.get("regions"), params.get("channels"))
            c2.write("[Share link](" + (share or "?") + ")")
            if c3.button(f"Delete {nm}", key=f"del_{nm}"):
                st.session_state.saved_views.pop(nm, None)
                st.rerun()

# Apply filters to dataframe
if sel_regions:
    df = df[df["region"].isin(sel_regions)]
if sel_channels:
    df = df[df["channel"].isin(sel_channels)]

# Navigation
pages = ["📊 Dashboard","🧩 Segmentation","🚨 Anomalies","📈 Forecast","🧠 AI Insights","🕒 Early Access Waitlist"]
page = st.sidebar.radio("Go to", pages, index=0)

# ----------------------------
# Dashboard
# ----------------------------
if page == "📊 Dashboard":
    st.title("📊 Dashboard")

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    total_rev = float(df.get("revenue", pd.Series(dtype=float)).sum()) if "revenue" in df.columns else float(df.select_dtypes(include=np.number).sum().sum())
    total_units = int(df.get("units", pd.Series(dtype=float)).sum()) if "units" in df.columns else len(df)
    avg_price = (df["revenue"].sum() / max(1, df["units"].sum())) if set(["revenue","units"]).issubset(df.columns) else np.nan
    c1.metric("Total Revenue", f"£{total_rev:,.0f}")
    c2.metric("Units Sold", f"{total_units:,}")
    c3.metric("Avg Price", f"£{avg_price:,.2f}" if not np.isnan(avg_price) else "–")
    c4.metric("Rows", f"{len(df):,}")

    # Simple trend
    if "date" in df.columns and "revenue" in df.columns:
        daily = df.groupby(pd.to_datetime(df["date"]).dt.to_period("D").dt.start_time)["revenue"].sum().reset_index()
        st.altair_chart(
            alt.Chart(daily).mark_line().encode(x="date:T", y=alt.Y("revenue:Q", title="Revenue"), tooltip=["date:T", alt.Tooltip("revenue:Q", format=",.0f")]).properties(height=320),
            use_container_width=True,
        )

    # Download center
    st.markdown("<div id='download-data'></div>", unsafe_allow_html=True)
    st.subheader("Download Data")
    st.caption("Exports the CURRENTLY FILTERED dataset shown above.")
    st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"), file_name="data_filtered.csv")
    st.download_button("Download JSON", df.to_json(orient="records").encode("utf-8"), file_name="data_filtered.json")
    xlsx_bytes = df_to_xlsx_bytes({
        "data_filtered": df,
        "anomalies": st.session_state.get("__last_anomalies_df", pd.DataFrame()),
        "forecast": st.session_state.get("__last_forecast_df", pd.DataFrame()),
    })
    st.download_button("Download XLSX", xlsx_bytes, file_name="export.xlsx")

    # Copy deep-link to this section
    st.button("🔗 Copy link to this section", key="copy_dl_btn")
    st.components.v1.html(
        """
        <script>
        (function(){
          function bind(){
            const btn = Array.from(window.parent.document.querySelectorAll('button')).find(b => /Copy link to this section/.test(b.textContent));
            if(!btn || btn.__bound){ setTimeout(bind, 600); return; }
            btn.__bound = true;
            btn.addEventListener('click', () => {
              try{
                const url = window.location.origin + window.location.pathname + window.location.search + '#download-data';
                navigator.clipboard.writeText(url);
              }catch(e){}
            });
          }
          setTimeout(bind, 400);
        })();
        </script>
        """,
        height=0,
    )

    # ----------------------------
    # 📄 Report Generator
    # ----------------------------
    st.markdown("---")
    with st.expander("📄 Generate Report (Markdown / HTML)", expanded=False):
        include_chart = st.checkbox("Include trend chart (best-effort)", value=True)
        include_movers = st.checkbox("Include Top Movers (by region)", value=True)
        note = st.text_area("Add a note (optional)", placeholder="Context, decisions, risks…")

        # Build content
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        active_regions = ", ".join(sel_regions) if sel_regions else "All"
        active_channels = ", ".join(sel_channels) if sel_channels else "All"

        # Movers table
        movers_tbl = None
        if include_movers and "date" in df.columns and "revenue" in df.columns and "region" in df.columns:
            g = (df.assign(_p=pd.to_datetime(df["date"]).dt.to_period("M").dt.start_time)
                   .groupby(["region","_p"])  ["revenue"].sum().reset_index().sort_values("_p"))
            g["prev"] = g.groupby("region")["revenue"].shift(1)
            g["pct_change"] = (g["revenue"] - g["prev"]) / g["prev"] * 100
            lastp = g["_p"].max()
            movers_tbl = g[g["_p"]==lastp].dropna(subset=["pct_change"]).sort_values("pct_change", ascending=False)
            movers_tbl = movers_tbl.rename(columns={"_p":"period","revenue":"value"})[["region","value","pct_change"]]

        # Markdown report
        md_lines = [
            f"# AI BI Report", 
            f"Generated: {now}",
            "",
            f"**Filters** — Region: {active_regions}; Channel: {active_channels}",
            "",
            f"**KPIs**:",
            f"- Total Revenue: £{total_rev:,.0f}",
            f"- Units Sold: {total_units:,}",
            f"- Avg Price: {'£'+format(avg_price,',.2f') if not np.isnan(avg_price) else '–'}",
        ]
        if note:
            md_lines += ["", "**Notes**:", note]
        if include_movers and movers_tbl is not None and not movers_tbl.empty:
            md_lines += ["", "**Top Movers (last month vs prior) — by Region**:"]
            # Use a simple table-like block
            head = movers_tbl.head(10).copy()
            head["value"] = head["value"].map(lambda x: f"£{x:,.0f}")
            head["pct_change"] = head["pct_change"].map(lambda x: f"{x:+.1f}%")
            md_lines += ["", head.to_csv(index=False)]
        md_text = "\n".join(md_lines)

        # HTML report
        html_parts = [
            "<html><head><meta charset='utf-8'><style>body{font-family:Inter,Arial,sans-serif;padding:20px;}table{border-collapse:collapse;width:100%;}td,th{border:1px solid #eee;padding:6px 8px;text-align:left;}h1{margin-top:0;} .muted{color:#666}</style></head><body>",
            f"<h1>AI BI Report</h1><div class='muted'>Generated: {now}</div>",
            f"<p><b>Filters</b> — Region: {active_regions}; Channel: {active_channels}</p>",
            f"<ul><li>Total Revenue: £{total_rev:,.0f}</li><li>Units Sold: {total_units:,}</li><li>Avg Price: {('£'+format(avg_price,',.2f')) if not np.isnan(avg_price) else '–'}</li></ul>",
        ]
        # Trend chart embed (PNG best-effort)
        if include_chart and "date" in df.columns and "revenue" in df.columns:
            try:
                daily = df.groupby(pd.to_datetime(df["date"]).dt.to_period("D").dt.start_time)["revenue"].sum().reset_index()
                chart = alt.Chart(daily).mark_line().encode(x="date:T", y="revenue:Q")
                # Try PNG via altair_saver
                from altair_saver import save as alt_save  # optional
                png_io = BytesIO()
                alt_save(chart, png_io, format="png", method="vl-convert")
                b64 = base64.b64encode(png_io.getvalue()).decode()
                html_parts.append(f"<h3>Trend</h3><img src='data:image/png;base64,{b64}' style='max-width:100%'>")
            except Exception:
                pass
        if include_movers and movers_tbl is not None and not movers_tbl.empty:
            html_parts.append("<h3>Top Movers (last month vs prior) — by Region</h3>")
            html_parts.append(movers_tbl.head(10).to_html(index=False))
        if note:
            html_parts.append("<h3>Notes</h3><p>" + st._escape_html(note) + "</p>")
        html_parts.append("</body></html>")
        html = "".join(html_parts).encode("utf-8")

        colr1, colr2 = st.columns(2)
        colr1.download_button("Download Markdown (.md)", md_text.encode("utf-8"), file_name="ai_bi_report.md")
        colr2.download_button("Download HTML (.html)", html, file_name="ai_bi_report.html")

# ----------------------------
# Other tabs (lightweight placeholders to preserve exports)
# ----------------------------
elif page == "🧩 Segmentation":
    st.title("🧩 Segmentation")
    st.info("Coming next: KMeans clustering with profiles. (Kept minimal to focus on reports & views.)")

elif page == "🚨 Anomalies":
    st.title("🚨 Anomalies")
    st.info("Stub anomalies saved so XLSX export includes a sheet.")
    st.session_state["__last_anomalies_df"] = df.head(100).copy()
    st.dataframe(st.session_state["__last_anomalies_df"], use_container_width=True)

elif page == "📈 Forecast":
    st.title("📈 Forecast")
    if "date" in df.columns and "revenue" in df.columns:
        daily = df.groupby(pd.to_datetime(df["date"]).dt.to_period("D").dt.start_time)["revenue"].sum().reset_index()
        # Fake forecast for demo purposes
        last = daily["date"].max()
        fut = pd.date_range(last + pd.Timedelta(days=1), periods=14, freq="D")
        fdf = pd.DataFrame({"date": fut, "forecast": np.linspace(daily["revenue"].tail(1).values[0], daily["revenue"].tail(1).values[0]*1.05, len(fut))})
        st.line_chart(fdf.set_index("date"))
        st.session_state["__last_forecast_df"] = fdf.copy()
    else:
        st.warning("Need date & revenue columns for forecast preview.")

elif page == "🧠 AI Insights":
    st.title("🧠 AI Insights")
    st.info("Heuristic cards & movers can plug in here — focus of this update is Reports & Saved Views.")

elif page == "🕒 Early Access Waitlist":
    st.title("🕒 Early Access Waitlist")
    st.markdown("Embed your Google Form or capture locally.")
    st.components.v1.iframe("https://docs.google.com/forms/d/e/1FAIpQLSf-demo-form/viewform?embedded=true", height=650)

# Footer
st.markdown("---")
st.caption("Report & Saved Views shipped • Permalinks via URL • XLSX/JSON/CSV exports • Deep link #download-data")
