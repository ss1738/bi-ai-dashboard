# AI BI Dashboard — Streamlit (Branding + Saved Views + Branded Reports with Embedded Logo)
# Save as: streamlit_app.py
#
# Optional (free) extras in requirements.txt for best experience:
# pandas
# numpy
# altair
# XlsxWriter
# altair_saver
# vl-convert-python

from io import BytesIO
from datetime import datetime
from typing import List, Optional
import base64
import os

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# ----------------------------
# Branding / Theme
# ----------------------------
BRAND = {
    "product_name": "AI BI Dashboard",
    "tagline": "Insights in minutes — not months.",
    "accent": "#5b8def",
    "footer": "AI BI Dashboard © 2025 • Generated automatically",
    # If you have your own logo, place it at assets/logo.png (PNG) and it will be embedded.
    "logo_path": "assets/logo.png",
}

st.set_page_config(page_title="AI BI Dashboard", page_icon="📊", layout="wide")

st.markdown(
    f"""
    <style>
      :root {{ --accent: {BRAND['accent']}; }}
      .metric-card {{ background:#fff; border:1px solid #eee; padding:14px 16px; border-radius:14px; }}
      .accent {{ color: var(--accent); }}
      .app-banner {{
        background: var(--accent);
        color: #fff;
        padding: 16px 18px;
        border-radius: 12px;
        margin: 12px 0 18px 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.06);
      }}
      .app-banner h1 {{ margin: 0 0 6px 0; font-size: 24px; }}
      .app-banner p  {{ margin: 0; opacity: 0.95; }}
      .app-banner a  {{ color: #fff; text-decoration: underline; }}
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------------------
# Utilities
# ----------------------------
def html_escape(s: str) -> str:
    try:
        return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    except Exception:
        return ""

def df_to_xlsx_bytes(sheets: dict) -> bytes:
    bio = BytesIO()
    try:
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
                        pd.DataFrame(d).to_excel(writer, index=False, sheet_name=(name or "sheet")[:31])
                    except Exception:
                        pass
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

# -------- Embedded logo handling --------
def _get_logo_base64(path: str) -> Optional[str]:
    try:
        if os.path.exists(path):
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode()
    except Exception:
        pass
    return None

def _get_placeholder_logo_b64(text: str = "AI BI") -> str:
    # Simple 240x80 SVG converted to base64 PNG-like data via data URL (keep SVG for tiny size)
    svg = f"""<svg xmlns='http://www.w3.org/2000/svg' width='240' height='80'>
      <rect width='240' height='80' rx='10' ry='10' fill='{BRAND['accent']}'/>
      <text x='50%' y='55%' text-anchor='middle' font-family='Arial, Helvetica, sans-serif' font-size='28' fill='white'>{html_escape(text)}</text>
    </svg>"""
    return base64.b64encode(svg.encode("utf-8")).decode()

LOGO_B64 = _get_logo_base64(BRAND["logo_path"])
if LOGO_B64:
    LOGO_HTML = f"<img src='data:image/png;base64,{LOGO_B64}' alt='Logo' style='height:48px;'>"
    MD_LOGO = f"![Logo](data:image/png;base64,{LOGO_B64})"
else:
    # Fallback to tiny inline SVG (base64)
    svg_b64 = _get_placeholder_logo_b64("AI BI")
    LOGO_HTML = f"<img src='data:image/svg+xml;base64,{svg_b64}' alt='Logo' style='height:48px;'>"
    MD_LOGO = f"![Logo](data:image/svg+xml;base64,{svg_b64})"

# ----------------------------
# In-app Branding Banner
# ----------------------------
if "show_banner" not in st.session_state:
    st.session_state.show_banner = True

if st.session_state.show_banner:
    col_b1, col_b2 = st.columns([8, 1])
    with col_b1:
        st.markdown(
            f"""
            <div class="app-banner">
              <h1>🚀 {html_escape(BRAND['product_name'])}</h1>
              <p>{html_escape(BRAND['tagline'])}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col_b2:
        if st.button("Hide", help="Hide banner", key="hide_banner_btn"):
            st.session_state.show_banner = False

# ----------------------------
# Query params / Saved views
# ----------------------------
def parse_query_list(val: Optional[str]) -> Optional[List[str]]:
    if val is None:
        return None
    if isinstance(val, list):
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
    qp = dict(st.query_params)
    if regions: qp["region"] = ",".join(regions)
    else: qp.pop("region", None)
    if channels: qp["channel"] = ",".join(channels)
    else: qp.pop("channel", None)
    st.query_params.clear()
    for k, v in qp.items():
        st.query_params[k] = v

def make_share_link(regions: Optional[List[str]], channels: Optional[List[str]], anchor: Optional[str] = None) -> str:
    path = "?"
    parts = []
    if regions: parts.append("region=" + ",".join(regions))
    if channels: parts.append("channel=" + ",".join(channels))
    path += "&".join(parts) if parts else ""
    if anchor:
        path += f"#{anchor}"
    return path

# ----------------------------
# Data source (Sidebar)
# ----------------------------
st.sidebar.title("AI BI Dashboard")

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

# Sidebar Filters
regions_all = sorted(df["region"].dropna().unique().tolist()) if "region" in df.columns else []
channels_all = sorted(df["channel"].dropna().unique().tolist()) if "channel" in df.columns else []
q_regions, q_channels = get_query_filters()

with st.sidebar.expander("Filters", expanded=True):
    sel_regions = st.multiselect("Region", regions_all, default=(q_regions or regions_all))
    sel_channels = st.multiselect("Channel", channels_all, default=(q_channels or channels_all))
    if st.button("Apply filters", type="primary"):
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
            if c3.button(f"Delete " + nm, key=f"del_{nm}"):
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
    total_rev = float(df.get("revenue", pd.Series(dtype=float)).sum()) if "revenue" in df.columns else 0.0
    total_units = int(df.get("units", pd.Series(dtype=float)).sum()) if "units" in df.columns else 0
    avg_price = (df["revenue"].sum() / max(1, df["units"].sum())) if {"revenue","units"}.issubset(df.columns) else float("nan")
    c1.metric("Total Revenue", f"£{total_rev:,.0f}")
    c2.metric("Units Sold", f"{total_units:,}")
    c3.metric("Avg Price", f"£{avg_price:,.2f}" if not np.isnan(avg_price) else "–")
    c4.metric("Rows", f"{len(df):,}")

    # Trend chart
    if {"date","revenue"}.issubset(df.columns):
        daily = df.groupby(pd.to_datetime(df["date"]).dt.to_period("D").dt.start_time)["revenue"].sum().reset_index()
        st.altair_chart(
            alt.Chart(daily).mark_line().encode(
                x="date:T", y=alt.Y("revenue:Q", title="Revenue"),
                tooltip=["date:T", alt.Tooltip("revenue:Q", format=",.0f")]
            ).properties(height=320),
            use_container_width=True,
        )

    # Download center (with deep-link anchor)
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

    # Copy deep-link button & smooth scroll for #download-data
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
          setTimeout(() => {
            try { if (window.location.hash === "#download-data") {
              const el = document.getElementById("download-data");
              if (el) el.scrollIntoView({behavior: "smooth", block: "start"}); }
            } catch (e) {}
          }, 500);
        })();
        </script>
        """,
        height=0,
    )

    # ----------------------------
    # 📄 Report Generator (Branded, with embedded logo)
    # ----------------------------
    st.markdown("---")
    with st.expander("📄 Generate Report (Markdown / HTML)", expanded=False):
        include_chart = st.checkbox("Include trend chart (best-effort)", value=True)
        include_movers = st.checkbox("Include Top Movers (by region)", value=True)
        note = st.text_area("Add a note (optional)", placeholder="Context, decisions, risks…")

        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        active_regions = ", ".join(sel_regions) if sel_regions else "All"
        active_channels = ", ".join(sel_channels) if sel_channels else "All"

        # Movers table
        movers_tbl = None
        if include_movers and {"date","revenue","region"}.issubset(df.columns):
            g = (df.assign(_p=pd.to_datetime(df["date"]).dt.to_period("M").dt.start_time)
                   .groupby(["region","_p"])["revenue"].sum().reset_index().sort_values("_p"))
            g["prev"] = g.groupby("region")["revenue"].shift(1)
            g["pct_change"] = (g["revenue"] - g["prev"]) / g["prev"] * 100
            lastp = g["_p"].max()
            movers_tbl = g[g["_p"] == lastp].dropna(subset=["pct_change"]).sort_values("pct_change", ascending=False)
            movers_tbl = movers_tbl.rename(columns={"_p":"period","revenue":"value"})[["region","value","pct_change"]].copy()

        # Optional chart image (best-effort, requires altair_saver + vl-convert-python)
        chart_b64 = None
        if include_chart and {"date","revenue"}.issubset(df.columns):
            try:
                daily = df.groupby(pd.to_datetime(df["date"]).dt.to_period("D").dt.start_time)["revenue"].sum().reset_index()
                chart = alt.Chart(daily).mark_line().encode(x="date:T", y="revenue:Q")
                from altair_saver import save as alt_save  # optional
                png_io = BytesIO()
                alt_save(chart, png_io, format="png", method="vl-convert")
                chart_b64 = base64.b64encode(png_io.getvalue()).decode()
            except Exception:
                chart_b64 = None  # still build report

        # Markdown (can embed base64 images in many viewers; not all will render)
        md_lines = [
            f"# {BRAND['product_name']} — Report",
            f"{MD_LOGO}",
            f"_Generated: {now}_",
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
            head = movers_tbl.head(10).copy()
            head["value"] = head["value"].map(lambda x: f"£{x:,.0f}")
            head["pct_change"] = head["pct_change"].map(lambda x: f"{x:+.1f}%")
            md_lines += ["", head.to_csv(index=False)]
        md_lines += ["", "---", BRAND["footer"]]
        md_bytes = ("\n".join(md_lines)).encode("utf-8")

        # HTML (fully branded with embedded logo)
        kpi_list_html = f"""
        <ul>
          <li><b>Total Revenue</b>: £{total_rev:,.0f}</li>
          <li><b>Units Sold</b>: {total_units:,}</li>
          <li><b>Avg Price</b>: {('£'+format(avg_price,',.2f')) if not np.isnan(avg_price) else '–'}</li>
        </ul>
        """

        movers_html = ""
        if include_movers and movers_tbl is not None and not movers_tbl.empty:
            movers_html = "<h3>Top Movers (last month vs prior) — by Region</h3>" + movers_tbl.head(12).to_html(index=False)

        chart_html = f"<img src='data:image/png;base64,{chart_b64}' style='max-width:100%;border:1px solid #eee;border-radius:8px'/>" if chart_b64 else ""

        html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{BRAND['product_name']} — Report</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    :root {{ --accent: {BRAND['accent']}; }}
    * {{ box-sizing: border-box; }}
    body {{ font-family: Inter, Segoe UI, Roboto, Arial, sans-serif; color:#222; margin:0; background:#fff; }}
    .wrap {{ max-width: 980px; margin: 0 auto; padding: 24px; }}
    .card {{ background:#fff; border:1px solid #eee; border-radius:14px; padding:18px; margin:12px 0; }}
    .muted {{ color:#666; }}
    .header {{ display:flex; align-items:center; gap:12px; }}
    .title {{ margin:0; font-size:28px; }}
    .tagline {{ margin:2px 0 0; color:#555; }}
    .badge {{ background: var(--accent); color:#fff; padding:2px 8px; border-radius:999px; font-size:12px; }}
    h2 {{ margin-top: 0.2rem; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #eee; padding: 8px; text-align: left; }}
    .footer {{ margin-top: 32px; padding: 16px; color:#666; font-size:13px; border-top:1px dashed #e5e5e5; }}
    @media print {{
      .card {{ break-inside: avoid; }}
      .footer {{ position: fixed; bottom: 0; left: 0; right: 0; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="header">
      {LOGO_HTML}
      <div>
        <h1 class="title">{BRAND['product_name']}</h1>
        <div class="tagline">{BRAND['tagline']}</div>
      </div>
      <div style="flex:1"></div>
      <span class="badge">Auto-generated</span>
    </div>

    <div class="card">
      <div class="muted">Generated</div>
      <div>{now}</div>
    </div>

    <div class="card">
      <div class="muted">Filters</div>
      <div>Region: {html_escape(active_regions)}; Channel: {html_escape(active_channels)}</div>
    </div>

    <div class="card">
      <h2>KPIs</h2>
      {kpi_list_html}
    </div>

    {("<div class='card'><h2>Trend</h2>"+chart_html+"</div>") if chart_html else ""}

    {(f"<div class='card'>{movers_html}</div>") if movers_html else ""}

    {("<div class='card'><h2>Notes</h2><div>"+html_escape(note)+"</div></div>") if note else ""}

    <div class="footer">
      {BRAND['footer']}
    </div>
  </div>
</body>
</html>"""

        html_bytes = html.encode("utf-8")

        colr1, colr2 = st.columns(2)
        colr1.download_button("Download Markdown (.md)", md_bytes, file_name="ai_bi_report.md")
        colr2.download_button("Download HTML (.html)", html_bytes, file_name="ai_bi_report.html")

# ----------------------------
# Other tabs (placeholders to preserve XLSX bundle)
# ----------------------------
elif page == "🧩 Segmentation":
    st.title("🧩 Segmentation")
    st.info("KMeans clustering + profiles can go here. (Kept minimal while we focus on reports & saved views.)")

elif page == "🚨 Anomalies":
    st.title("🚨 Anomalies")
    st.caption("Stub anomalies saved so XLSX export includes a sheet.")
    st.session_state["__last_anomalies_df"] = df.head(100).copy()
    st.dataframe(st.session_state["__last_anomalies_df"], use_container_width=True)

elif page == "📈 Forecast":
    st.title("📈 Forecast")
    if {"date","revenue"}.issubset(df.columns):
        daily = df.groupby(pd.to_datetime(df["date"]).dt.to_period("D").dt.start_time)["revenue"].sum().reset_index()
        last = daily["date"].max()
        fut = pd.date_range(last + pd.Timedelta(days=1), periods=14, freq="D")
        fdf = pd.DataFrame({"date": fut, "forecast": np.linspace(daily["revenue"].iloc[-1], daily["revenue"].iloc[-1]*1.05, len(fut))})
        st.line_chart(fdf.set_index("date"))
        st.session_state["__last_forecast_df"] = fdf.copy()
    else:
        st.warning("Need date & revenue columns for forecast preview.")

elif page == "🧠 AI Insights":
    st.title("🧠 AI Insights")
    st.info("Heuristic insight cards / top movers target this tab next.")

elif page == "🕒 Early Access Waitlist":
    st.title("🕒 Early Access Waitlist")
    st.markdown("Embed your Google Form or capture locally.")
    st.components.v1.iframe("https://docs.google.com/forms/d/e/1FAIpQLSf-demo-form/viewform?embedded=true", height=650)

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.caption("Branding, Saved Views, Deep links, CSV/JSON/XLSX exports, and Branded Reports with embedded logo.")
