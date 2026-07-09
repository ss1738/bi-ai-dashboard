"""
Real LLM-powered insights + conversational analytics for the dashboard.

Swappable provider (Groq / OpenAI). Everything is grounded in a compact, factual summary
of the *current filtered* data, the model is given the numbers and told to use only those,
so it can't invent figures. Works with no key too (returns the summary + a note).
"""
from __future__ import annotations
import pandas as pd


def summarize(df: pd.DataFrame) -> str:
    """A compact, LLM-friendly factual summary of the data (no raw row dump)."""
    if df is None or len(df) == 0:
        return "No data after the current filters."
    lines = [f"Rows: {len(df):,}. Columns: {list(df.columns)}."]
    if "date" in df:
        lines.append(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}.")
    if "revenue" in df:
        lines.append(f"Total revenue: {df['revenue'].sum():,.0f}. "
                     f"Mean/row: {df['revenue'].mean():,.1f}. Median: {df['revenue'].median():,.1f}.")
    for col in ("region", "channel", "product"):
        if col in df and "revenue" in df:
            top = df.groupby(col)["revenue"].sum().sort_values(ascending=False)
            lines.append(f"Revenue by {col}: " + "; ".join(f"{k}={v:,.0f}" for k, v in top.items()))
    if "anomaly" in df:
        lines.append(f"Anomalous records flagged: {int(df['anomaly'].sum())} of {len(df)}.")
    if "segment" in df and "revenue" in df:
        seg = df.groupby("segment")["revenue"].sum()
        lines.append("Revenue by customer segment: " + "; ".join(f"{k}={v:,.0f}" for k, v in seg.items()))
    if "date" in df and "revenue" in df:
        daily = df.groupby("date")["revenue"].sum().sort_index()
        if len(daily) > 3:
            half = len(daily) // 2
            first, last = daily.iloc[:half].mean(), daily.iloc[half:].mean()
            lines.append(f"Revenue trend: {'rising' if last > first else 'falling'} "
                         f"({first:,.0f} -> {last:,.0f}, first vs second half).")
    return "\n".join(lines)


def _call_llm(system: str, user: str, provider: str, api_key: str) -> str | None:
    if provider == "Groq" and api_key:
        from groq import Groq
        r = Groq(api_key=api_key).chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.2)
        return r.choices[0].message.content
    if provider == "OpenAI" and api_key:
        from openai import OpenAI
        r = OpenAI(api_key=api_key).chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.2)
        return r.choices[0].message.content
    if provider == "xAI" and api_key:
        # xAI's API is OpenAI-compatible; just point the client at their base_url
        from openai import OpenAI
        r = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1").chat.completions.create(
            model="grok-2-latest",
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.2)
        return r.choices[0].message.content
    return None


def generate_insights(df: pd.DataFrame, provider: str, api_key: str) -> str:
    summary = summarize(df)
    system = ("You are a sharp business analyst. From the data summary, write 4-6 concise, "
              "specific, actionable insights and recommendations for a business leader. "
              "Use ONLY the numbers provided, never invent data. Lead with the most important point.")
    out = _call_llm(system, f"Data summary:\n{summary}\n\nWrite the insights:", provider, api_key)
    if out:
        return out
    return ("_Add a Groq (free) or OpenAI key in the sidebar for AI-written insights. "
            "Here is the factual data summary the AI would analyse:_\n\n```\n" + summary + "\n```")


def ask_data(question: str, df: pd.DataFrame, provider: str, api_key: str) -> str:
    summary = summarize(df)
    system = ("You answer questions about a business dataset using ONLY the provided summary and "
              "aggregates. Cite the specific numbers. If the answer cannot be derived from the "
              "summary, say so plainly. Be concise.")
    out = _call_llm(system, f"Data summary:\n{summary}\n\nQuestion: {question}\n\nAnswer:", provider, api_key)
    if out:
        return out
    return ("_Add an API key in the sidebar for generated answers. Relevant data summary:_\n\n```\n"
            + summary + "\n```")
