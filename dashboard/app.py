import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="AutoTriage Dashboard", layout="wide")
st.title("AutoTriage Dashboard")

API_BASE = "http://localhost:8000"
DATASET_PATH = Path("data/ticket_dataset_v2.json")
BASELINE_PATH = Path("evaluation/checkpoint2_baseline.json")
CP3_RESULTS_PATH = Path("evaluation/checkpoint3_results.json")


# ---------- Helpers ----------
def load_tickets() -> tuple[pd.DataFrame, str]:
    try:
        r = requests.get(f"{API_BASE}/tickets", timeout=3)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict) and "tickets" in data:
            return pd.DataFrame(data["tickets"]), "live intake API"
        return pd.DataFrame(data), "live intake API"
    except Exception:
        if DATASET_PATH.exists():
            with open(DATASET_PATH, "r") as f:
                return pd.DataFrame(json.load(f)), "local fallback dataset"
        return pd.DataFrame(), "none"


def service_status(url: str) -> str:
    try:
        r = requests.get(url, timeout=2)
        if r.ok:
            return "running"
    except Exception:
        pass
    return "not reachable"


def load_json_if_exists(path: Path) -> dict:
    if path.exists():
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def format_pct(value):
    if value is None:
        return "—"
    try:
        return f"{float(value):.2%}"
    except Exception:
        return "—"


def format_num(value):
    if value is None:
        return "—"
    try:
        return f"{float(value):.0f}"
    except Exception:
        return "—"


def build_trend_df(cp2: dict, cp3: dict) -> pd.DataFrame:
    rows = []

    if cp2:
        rows.append({
            "checkpoint": "CP2",
            "category_accuracy": cp2.get("category_accuracy", cp2.get("accuracy")),
            "severity_accuracy": cp2.get("severity_accuracy"),
            "routing_accuracy": cp2.get("routing_accuracy"),
        })

    if cp3:
        rows.append({
            "checkpoint": "CP3",
            "category_accuracy": cp3.get("category_accuracy", cp3.get("accuracy")),
            "severity_accuracy": cp3.get("severity_accuracy"),
            "routing_accuracy": cp3.get("routing_accuracy"),
        })

    return pd.DataFrame(rows)


# ---------- Controls ----------
control_left, control_right = st.columns([1, 5])
with control_left:
    if st.button("Refresh", use_container_width=True):
        st.rerun()

# ---------- Load Data ----------
df, data_source = load_tickets()
classifier = service_status("http://localhost:8001/health")
cp2_metrics = load_json_if_exists(BASELINE_PATH)
cp3_metrics = load_json_if_exists(CP3_RESULTS_PATH)
metrics = cp3_metrics if cp3_metrics else cp2_metrics

st.caption(f"Data source: {data_source}")

# ---------- Top Status / Metrics ----------
status_col1, status_col2, status_col3 = st.columns(3)

with status_col1:
    st.subheader("Intake API Status")
    if not df.empty:
        st.success("Ticket data available")
    else:
        st.warning("No ticket data available")

with status_col2:
    st.subheader("Classifier Status")
    if classifier == "running":
        st.success("Classifier running")
    else:
        st.warning("Classifier not reachable")

with status_col3:
    st.subheader("Metrics")
    if metrics:
        metric_a, metric_b, metric_c = st.columns(3)
        with metric_a:
            st.metric("Category Accuracy", format_pct(metrics.get("category_accuracy", metrics.get("accuracy"))))
        with metric_b:
            st.metric("Severity Accuracy", format_pct(metrics.get("severity_accuracy")))
        with metric_c:
            st.metric("Routing Accuracy", format_pct(metrics.get("routing_accuracy")))
    else:
        st.caption("No evaluation metrics available yet.")

# ---------- Quick Summary ----------
st.divider()
summary1, summary2, summary3, summary4 = st.columns(4)

with summary1:
    st.metric("Total Tickets", len(df) if not df.empty else 0)

with summary2:
    st.metric("Categories", df["category"].nunique() if not df.empty and "category" in df.columns else 0)

with summary3:
    st.metric("Severity Levels", df["severity"].nunique() if not df.empty and "severity" in df.columns else 0)

with summary4:
    if cp3_metrics:
        st.metric("Current Eval File", "CP3")
    elif cp2_metrics:
        st.metric("Current Eval File", "CP2")
    else:
        st.metric("Current Eval File", "None")

# ---------- Main Charts ----------
st.divider()
left_chart, right_chart = st.columns([2, 1])

with left_chart:
    st.subheader("Ticket Count by Category and Severity")
    if not df.empty and {"category", "severity"}.issubset(df.columns):
        grouped = (
            df.groupby(["category", "severity"])
            .size()
            .unstack(fill_value=0)
            .sort_index()
        )
        st.bar_chart(grouped, use_container_width=True)
    else:
        st.info("No category/severity data available yet.")

with right_chart:
    st.subheader("Routing Distribution")
    routing_col = None
    if "assigned_team" in df.columns:
        routing_col = "assigned_team"
    elif "assigned_queue" in df.columns:
        routing_col = "assigned_queue"

    if not df.empty and routing_col:
        routing_counts = (
            df[routing_col]
            .fillna("unassigned")
            .value_counts()
            .rename_axis("route")
            .reset_index(name="count")
        )
        st.bar_chart(routing_counts.set_index("route"), use_container_width=True)
    else:
        st.info("No routing data available yet.")

# ---------- Recent Tickets ----------
st.divider()
st.subheader("Recent Tickets")

if not df.empty:
    preferred_cols = [
        "ticket_id",
        "title",
        "category",
        "severity",
        "classification_status",
        "routing_status",
        "assigned_team",
        "assigned_queue",
        "source",
    ]
    cols = [c for c in preferred_cols if c in df.columns]
    display_df = df[cols].copy()

    if "ticket_id" in display_df.columns:
        display_df = display_df.sort_values("ticket_id", ascending=False)

    st.dataframe(
        display_df.head(20),
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("No tickets to display.")

# ---------- Trends ----------
st.divider()
st.subheader("Evaluation Trends")

trend_df = build_trend_df(cp2_metrics, cp3_metrics)

if not trend_df.empty:
    trend_plot_df = trend_df.set_index("checkpoint")
    numeric_cols = [c for c in ["category_accuracy", "severity_accuracy", "routing_accuracy"] if c in trend_plot_df.columns]
    trend_plot_df = trend_plot_df[numeric_cols]

    if not trend_plot_df.empty:
        st.line_chart(trend_plot_df, use_container_width=True)

    st.dataframe(
        trend_df,
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("No trend data available yet.")

# ---------- Confusion Matrix ----------
st.divider()
st.subheader("Confusion Matrix (Category)")

if not df.empty and {"category", "predicted_category"}.issubset(df.columns):
    cm = pd.crosstab(df["category"], df["predicted_category"])
    st.caption("Using live category vs predicted_category values")
else:
    labels = ["Auth", "API", "UI", "Backend"]
    matrix = np.array([
        [20, 2, 1, 0],
        [1, 18, 2, 1],
        [0, 3, 22, 2],
        [1, 0, 2, 19],
    ])
    cm = pd.DataFrame(matrix, index=labels, columns=labels)
    st.caption("Showing placeholder confusion matrix (no predictions yet)")

st.dataframe(
    cm.style.background_gradient(cmap="Blues"),
    use_container_width=True,
)

# ---------- Latency ----------
st.divider()
st.subheader("Latency Summary")

lat1, lat2, lat3 = st.columns(3)
with lat1:
    st.metric("P50 Latency (ms)", format_num(metrics.get("p50_latency_ms") if metrics else None))
with lat2:
    st.metric("P95 Latency (ms)", format_num(metrics.get("p95_latency_ms") if metrics else None))
with lat3:
    st.metric("P99 Latency (ms)", format_num(metrics.get("p99_latency_ms") if metrics else None))

# ---------- Raw Metrics Debug ----------
with st.expander("Show Raw Evaluation Metrics"):
    if metrics:
        st.json(metrics)
    else:
        st.write("No metrics file loaded.")