# BI Dashboard with AI

A Streamlit dashboard for sales data. It does the usual BI and ML (KPIs, charts, segmentation,
anomaly detection, forecasting) and adds two LLM features: an insights summary, and a chat where
you can ask questions about the data in plain English.

The LLM answers are grounded in a factual summary of the data, so they work from the real
numbers instead of guessing.

## Features

ML and BI:
- KPIs, interactive Plotly charts, filters, light/dark theme
- Customer segmentation (KMeans)
- Revenue anomaly detection (IsolationForest) with drill-down
- Sales forecasting (Prophet, with a moving-average fallback)
- SQLite persistence and PDF export

LLM features:
- AI Insights: reads the current filtered data and writes a short set of recommendations.
- Ask Your Data: ask a question like "which region is underperforming, and by how much?" and get an answer from the data.
- Works with Groq (free) or OpenAI. It also runs without a key, showing the data summary, so it never breaks.

## Run locally
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```
In the sidebar, choose a provider and paste a key:
- Groq: free key at https://console.groq.com
- OpenAI: an OPENAI_API_KEY
- Or leave it on None to run without a key.

## Deploy a live demo (free)
Push to GitHub, open share.streamlit.io, point it at streamlit_app.py, and add GROQ_API_KEY in
the app secrets so visitors get AI answers.

## How the grounding works
The model is given a factual summary of the data (totals, breakdowns by region and channel,
anomalies, segments, trend) and told to use only those numbers. The ML does the calculations;
the LLM does the explanation.

## Stack
Python, Streamlit, pandas, scikit-learn, Plotly, Prophet, Groq/OpenAI

Built by Satyawan Singh.
