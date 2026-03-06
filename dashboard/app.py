import json
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="AutoTriage Dashboard", layout="wide")
st.title("AutoTriage Dashboard")

API_BASE = "http://localhost:8000"
DATASET_PATH = Path("data/ticket_dataset_v1.json")
BASELINE_PATH = Path("evaluation/checkpoint2_baseline.json")

def load_tickets():
    try:
        r = requests.get(f"{API_BASE}/tickets", timeout=3)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict) and "tickets" in data:
            return pd.DataFrame(data["tickets"])
        return pd.DataFrame(data)
    except Exception:
        if DATASET_PATH.exists():
            with open(DATASET_PATH, "r") as f:
                return pd.DataFrame(json.load(f))
        return pd.DataFrame()

def classifier_status():
    try:
        r = requests.get("http://localhost:8001/health", timeout=2)
        if r.ok:
            return "running"
    except Exception:
        pass
    return "not reachable"

df = load_tickets()
status = classifier_status()

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Ticket Count by Category")
    if not df.empty and "category" in df.columns:
        counts = df["category"].value_counts()
        st.bar_chart(counts)
    else:
        st.info("No ticket data available yet.")

with col2:
    st.subheader("Classifier Status")
    if status == "running":
        st.success("Classifier running")
    else:
        st.warning("Classifier not reachable")

    if BASELINE_PATH.exists():
        try:
            with open(BASELINE_PATH, "r") as f:
                metrics = json.load(f)
            st.subheader("Latest Baseline")
            if "category_accuracy" in metrics:
                st.metric("Category Accuracy", f'{metrics["category_accuracy"]:.2%}')
            elif "accuracy" in metrics:
                st.metric("Accuracy", f'{metrics["accuracy"]:.2%}')
        except Exception:
            st.caption("Could not read baseline metrics.")

st.subheader("Recent Tickets")
if not df.empty:
    cols = [c for c in ["ticket_id", "title", "category", "severity", "source"] if c in df.columns]
    st.dataframe(df[cols].head(20), use_container_width=True)
else:
    st.info("No tickets to display.")