import json
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="AutoTriage Dashboard", layout="wide")
st.title("AutoTriage Dashboard")

API_BASE = "http://localhost:8000"
DATASET_PATH = Path("data/ticket_dataset_v2.json")
BASELINE_PATH = Path("evaluation/checkpoint2_baseline.json")
CP3_RESULTS_PATH = Path("evaluation/checkpoint3_results.json")


def _flatten_tickets(records: list[dict]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()

    rows: list[dict] = []
    for record in records:
        row = {k: v for k, v in record.items() if k not in {"classification", "routing", "latency"}}
        classification = record.get("classification") or {}
        routing = record.get("routing") or {}
        latency = record.get("latency") or {}
        row.update(classification)
        row.update(routing)
        row.update(latency)
        rows.append(row)
    return pd.DataFrame(rows)


def load_tickets() -> tuple[pd.DataFrame, str]:
    try:
        response = requests.get(f"{API_BASE}/tickets", timeout=3)
        response.raise_for_status()
        payload = response.json()
        records = payload.get("tickets", payload) if isinstance(payload, dict) else payload
        return _flatten_tickets(records), "live intake API"
    except Exception:
        if DATASET_PATH.exists():
            raw = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
            records = raw.get("tickets", raw) if isinstance(raw, dict) else raw
            return pd.DataFrame(records), "local fallback dataset"
        return pd.DataFrame(), "none"


def service_status(url: str) -> str:
    try:
        response = requests.get(url, timeout=2)
        if response.ok:
            return "running"
    except Exception:
        pass
    return "not reachable"


def load_json_if_exists(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def format_pct(value):
    if value is None:
        return "—"
    return f"{float(value):.2%}"


def format_num(value):
    if value is None:
        return "—"
    return f"{float(value):.0f}"


def build_trend_df(cp2: dict, cp3: dict) -> pd.DataFrame:
    rows = []
    if cp2:
        metrics = cp2.get("metrics", cp2)
        rows.append(
            {
                "checkpoint": "CP2",
                "category_accuracy": metrics.get("category_accuracy", metrics.get("accuracy")),
                "severity_accuracy": metrics.get("severity_accuracy"),
                "routing_accuracy": metrics.get("routing_accuracy"),
            }
        )
    if cp3:
        rows.append(
            {
                "checkpoint": "CP3",
                "category_accuracy": cp3.get("category_accuracy"),
                "severity_accuracy": cp3.get("severity_accuracy"),
                "routing_accuracy": cp3.get("routing_accuracy"),
            }
        )
    return pd.DataFrame(rows)


control_left, _ = st.columns([1, 5])
with control_left:
    if st.button("Refresh", use_container_width=True):
        st.rerun()

df, data_source = load_tickets()
classifier = service_status("http://localhost:8001/health")
cp2_metrics = load_json_if_exists(BASELINE_PATH)
cp3_metrics = load_json_if_exists(CP3_RESULTS_PATH)
metrics = cp3_metrics if cp3_metrics else cp2_metrics.get("metrics", cp2_metrics)

st.caption(f"Data source: {data_source}")

status_col1, status_col2, status_col3 = st.columns(3)
with status_col1:
    st.subheader("Intake API Status")
    st.success("Ticket data available") if not df.empty else st.warning("No ticket data available")
with status_col2:
    st.subheader("Classifier Status")
    st.success("Classifier running") if classifier == "running" else st.warning("Classifier not reachable")
with status_col3:
    st.subheader("Metrics")
    if metrics:
        metric_a, metric_b, metric_c = st.columns(3)
        with metric_a:
            st.metric("Category Accuracy", format_pct(metrics.get("category_accuracy", metrics.get("accuracy"))))
        with metric_b:
            st.metric("Severity Accuracy", format_pct(metrics.get("severity_accuracy")))
        with metric_c:
            st.metric("Routing Accuracy", format_pct(cp3_metrics.get("routing_accuracy") if cp3_metrics else metrics.get("routing_accuracy")))
    else:
        st.caption("No evaluation metrics available yet.")

st.divider()
summary1, summary2, summary3, summary4 = st.columns(4)
with summary1:
    st.metric("Total Tickets", len(df) if not df.empty else 0)
with summary2:
    st.metric("Categories", df["category"].nunique() if not df.empty and "category" in df.columns else 0)
with summary3:
    st.metric("Severity Levels", df["severity"].nunique() if not df.empty and "severity" in df.columns else 0)
with summary4:
    st.metric("Current Eval File", "CP3" if cp3_metrics else "CP2" if cp2_metrics else "None")

st.divider()
left_chart, right_chart = st.columns([2, 1])
with left_chart:
    st.subheader("Ticket Count by Category and Severity")
    if not df.empty and {"category", "severity"}.issubset(df.columns):
        grouped = df.groupby(["category", "severity"]).size().unstack(fill_value=0).sort_index()
        st.bar_chart(grouped, use_container_width=True)
    else:
        st.info("No category/severity data available yet.")

with right_chart:
    st.subheader("Routing Distribution")
    routing_col = "assigned_team" if "assigned_team" in df.columns else "assigned_queue" if "assigned_queue" in df.columns else None
    if not df.empty and routing_col:
        routing_counts = df[routing_col].fillna("unassigned").value_counts().rename_axis("route").to_frame("count")
        st.bar_chart(routing_counts, use_container_width=True)
    else:
        st.info("No routing data available yet.")

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
        "needs_review",
        "assigned_team",
        "assigned_queue",
        "source",
    ]
    cols = [c for c in preferred_cols if c in df.columns]
    display_df = df[cols].copy()
    st.dataframe(display_df.head(20), use_container_width=True, hide_index=True)
else:
    st.info("No tickets to display.")

st.divider()
st.subheader("Evaluation Trends")
trend_df = build_trend_df(cp2_metrics, cp3_metrics)
if not trend_df.empty:
    plot_df = trend_df.set_index("checkpoint")
    st.line_chart(plot_df[[c for c in ["category_accuracy", "severity_accuracy", "routing_accuracy"] if c in plot_df.columns]], use_container_width=True)
    st.dataframe(trend_df, use_container_width=True, hide_index=True)
else:
    st.info("No trend data available yet.")

st.divider()
st.subheader("Confusion Matrix (Category)")
category_cm = cp3_metrics.get("category_confusion_matrix", {}) if cp3_metrics else {}
if category_cm:
    labels = category_cm.get("labels", [])
    matrix = category_cm.get("matrix", [])
    cm_df = pd.DataFrame(matrix, index=labels, columns=labels)
    st.dataframe(cm_df.style.background_gradient(cmap="Blues"), use_container_width=True)
else:
    st.info("Run `python -m evaluation.run_eval` to generate the latest confusion matrix.")

st.divider()
st.subheader("Latency Summary")
lat1, lat2, lat3 = st.columns(3)
with lat1:
    st.metric("P50 Latency (ms)", format_num(cp3_metrics.get("p50_latency_ms") if cp3_metrics else None))
with lat2:
    st.metric("P95 Latency (ms)", format_num(cp3_metrics.get("p95_latency_ms") if cp3_metrics else None))
with lat3:
    st.metric("P99 Latency (ms)", format_num(cp3_metrics.get("p99_latency_ms") if cp3_metrics else None))

with st.expander("Show Raw Evaluation Metrics"):
    if cp3_metrics:
        st.json(cp3_metrics)
    elif cp2_metrics:
        st.json(cp2_metrics)
    else:
        st.write("No metrics file loaded.")
