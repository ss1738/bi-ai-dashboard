# ----------------------------
# AI Revenue Recovery Dashboard (Full Script)
# ----------------------------
import streamlit as st
import pandas as pd
import numpy as np
import sqlite3, os
from datetime import datetime
from prophet import Prophet
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import IsolationForest
from fpdf import FPDF

# Deep Learning libs
import torch
import torch.nn as nn
import torch.optim as optim
from tensorflow import keras
from tensorflow.keras import layers

# ----------------------------
# Config
# ----------------------------
st.set_page_config(page_title="AI Revenue Recovery", layout="wide")
DB_PATH = "sales.db"

# ----------------------------
# Database Functions
# ----------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sales_data (
            date TEXT,
            region TEXT,
            channel TEXT,
            segment TEXT,
            product TEXT,
            revenue REAL,
            customers INTEGER
        )
    """)
    conn.close()

def save_to_db(df):
    conn = sqlite3.connect(DB_PATH)
    df.to_sql("sales_data", conn, if_exists="replace", index=False)
    conn.close()

def load_from_db():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM sales_data", conn)
    conn.close()
    df["date"] = pd.to_datetime(df["date"])
    return df

# ----------------------------
# Forecasting (Prophet)
# ----------------------------
def make_forecast_prophet(df, days=30):
    daily = df.groupby("date")["revenue"].sum().reset_index()
    daily.columns = ["ds", "y"]
    model = Prophet()
    model.fit(daily)
    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)
    return forecast

# ----------------------------
# ML Models
# ----------------------------
def train_churn_xgboost(df):
    df = df.copy()
    df["avg_rev_per_cust"] = df["revenue"] / df["customers"].replace(0,1)
    df["churn_flag"] = (df["revenue"] < df["revenue"].median()).astype(int)

    X = df[["revenue","customers","avg_rev_per_cust"]]
    y = df["churn_flag"]

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    model = xgb.XGBClassifier(eval_metric="logloss", use_label_encoder=False)
    model.fit(X_train,y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test,preds)
    return model, acc

def train_upsell_lightgbm(df):
    df = df.copy()
    df["avg_rev_per_cust"] = df["revenue"] / df["customers"].replace(0,1)
    df["upsell_flag"] = (df["avg_rev_per_cust"] > df["avg_rev_per_cust"].median()).astype(int)

    X = df[["revenue","customers","avg_rev_per_cust"]]
    y = df["upsell_flag"]

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    model = lgb.LGBMClassifier()
    model.fit(X_train,y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test,preds)
    return model, acc

# ----------------------------
# Deep Learning - PyTorch (LSTM)
# ----------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=2, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out,_ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

def train_lstm_pytorch(series, epochs=5):
    data = torch.tensor(series.values, dtype=torch.float32).view(-1,1,1)
    X, y = data[:-1], data[1:]
    model = LSTMModel()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for _ in range(epochs):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

    return model, loss.item()

# ----------------------------
# Deep Learning - TensorFlow (Dense Model)
# ----------------------------
def train_tf_dense(df):
    X = df[["revenue","customers"]].values
    y = (df["revenue"] > df["revenue"].median()).astype(int).values

    model = keras.Sequential([
        layers.Dense(32, activation="relu", input_shape=(X.shape[1],)),
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(X, y, epochs=5, verbose=0)
    loss, acc = model.evaluate(X, y, verbose=0)
    return model, acc

# ----------------------------
# Anomaly Detection
# ----------------------------
def detect_anomalies(df):
    daily = df.groupby("date")["revenue"].sum().reset_index()
    clf = IsolationForest(contamination=0.1, random_state=42)
    daily["anomaly"] = clf.fit_predict(daily[["revenue"]])
    return daily[daily["anomaly"] == -1]

# ----------------------------
# PDF Export
# ----------------------------
def export_pdf(kpis, insights):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200,10,"AI Revenue Recovery Report",ln=True,align="C")
    pdf.set_font("Arial","",12)
    pdf.ln(10)
    pdf.cell(200,10,"KPIs",ln=True)
    for k,v in kpis.items():
        pdf.cell(200,10,f"{k}: {v}",ln=True)
    pdf.ln(10)
    pdf.cell(200,10,"Insights",ln=True)
    for ins in insights:
        pdf.multi_cell(0,10,f"- {ins}")
    return pdf.output(dest="S").encode("latin-1")

# ----------------------------
# UI
# ----------------------------
st.title("📊 AI Revenue Recovery Dashboard")
init_db()

uploaded = st.file_uploader("Upload sales CSV", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
    df["date"] = pd.to_datetime(df["date"])
    save_to_db(df)
    st.success("✅ Data saved")
elif os.path.exists(DB_PATH):
    df = load_from_db()
    st.info("ℹ️ Loaded saved data")
else:
    st.warning("Upload a CSV to begin")
    st.stop()

# ----------------------------
# Filters
# ----------------------------
regions = st.multiselect("Region", df["region"].unique(), default=list(df["region"].unique()))
channels = st.multiselect("Channel", df["channel"].unique(), default=list(df["channel"].unique()))
start, end = st.date_input("Date Range", [df["date"].min(), df["date"].max()])

mask = (df["region"].isin(regions)) & (df["channel"].isin(channels)) & (df["date"].between(start,end))
df = df[mask]

# ----------------------------
# KPIs
# ----------------------------
col1,col2,col3,col4 = st.columns(4)
with col1: st.metric("Total Revenue", f"${df['revenue'].sum():,.0f}")
with col2: st.metric("Customers", f"{df['customers'].sum():,.0f}")
with col3: st.metric("Avg Deal Size", f"${(df['revenue'].sum()/df['customers'].sum()):,.0f}")
with col4: st.metric("Days", df["date"].nunique())

# ----------------------------
# Models
# ----------------------------
st.subheader("📈 Forecast (Prophet)")
forecast = make_forecast_prophet(df)
st.line_chart(forecast.set_index("ds")[["yhat","yhat_lower","yhat_upper"]])

st.subheader("🤖 ML Models")
xgb_model, acc_xgb = train_churn_xgboost(df)
lgb_model, acc_lgb = train_upsell_lightgbm(df)
st.write(f"XGBoost Churn Model Accuracy: {acc_xgb:.2f}")
st.write(f"LightGBM Upsell Model Accuracy: {acc_lgb:.2f}")

st.subheader("🧠 Deep Learning")
lstm_model, lstm_loss = train_lstm_pytorch(df.groupby("date")["revenue"].sum())
tf_model, tf_acc = train_tf_dense(df)
st.write(f"PyTorch LSTM final loss: {lstm_loss:.4f}")
st.write(f"TensorFlow Dense Model Accuracy: {tf_acc:.2f}")

st.subheader("🚨 Anomaly Detection")
anoms = detect_anomalies(df)
st.dataframe(anoms)

# ----------------------------
# Export
# ----------------------------
kpis = {"Total Revenue": f"${df['revenue'].sum():,.0f}","Customers":df["customers"].sum()}
insights = ["Recover anomaly days","Upsell top customers","Expand Online channel"]

if st.button("📥 Export PDF Report"):
    pdf_bytes = export_pdf(kpis,insights)
    st.download_button("Download PDF",data=pdf_bytes,file_name="report.pdf")

# ----------------------------
# Assistant
# ----------------------------
st.subheader("💬 AI Assistant (demo)")
q = st.text_input("Ask me about revenue")
if q:
    st.write("🤖 (Demo AI): This is a placeholder. Future: RAG-powered answers.")
