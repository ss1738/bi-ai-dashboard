# 📊 AI-Native BI Dashboard

An interactive **Business Intelligence dashboard** with **real GenAI** built on top of classical
ML — upload your sales data (or use a sample), explore it, and then let an LLM analyse it and
answer your questions in plain English, grounded in the actual numbers.

Built with Streamlit by an ML engineer — real models and real LLM integration, not a template.

## Features
**Classical ML & BI**
- KPI metrics, interactive Plotly charts, filters, dark/light theme
- Customer segmentation (**KMeans**)
- Revenue-anomaly detection (**IsolationForest**) with drill-down
- Sales forecasting (**Prophet**, with a moving-average fallback)
- SQLite persistence and PDF export

**GenAI (new)**
- **🤖 AI Insights** — an LLM reads the *current filtered data* and writes specific, actionable
  recommendations. Grounded in a factual data summary, so it uses only real numbers.
- **💬 Ask Your Data** — conversational analytics: ask questions in plain English
  ("which region is underperforming, and by how much?") and get answers grounded in your data.
- **Model-agnostic**: swap between **Groq** (free) and **OpenAI**. Works key-free too (shows the
  grounded data summary), so the app never breaks without an API key.

## 🚀 Run locally
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```
In the sidebar, pick an LLM provider and paste a key:
- **Groq** — free key at https://console.groq.com (fastest to try).
- **OpenAI** — an `OPENAI_API_KEY`.
- Or leave it on **None** to use the app with the grounded data summaries (no key needed).

## 🌐 Deploy a live demo (free)
Push to GitHub → [share.streamlit.io](https://share.streamlit.io) → point at `streamlit_app.py`
→ add `GROQ_API_KEY` in the app's Secrets so visitors get AI answers.

## Why it's built this way
The AI is **grounded**: the model is given a factual summary of the data and told to use only
those numbers, which is what prevents confident-but-wrong answers. Classical ML (clustering,
anomaly detection, forecasting) does the quantitative work; the LLM does the explanation and
the conversation. The LLM is swappable; the value is the pipeline around it.

## Stack
Python · Streamlit · pandas · scikit-learn · Plotly · Prophet · Groq / OpenAI

---
Built by **Satyawan Singh** — AI/ML engineer. Custom models & systems from scratch.
