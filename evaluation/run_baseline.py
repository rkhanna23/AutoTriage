"""CP2-SA-04 baseline benchmark runner for Dataset v1.

Usage:
  python -m evaluation.run_baseline \
    --dataset data/ticket_dataset_v1.json \
    --output evaluation/checkpoint2_baseline.json
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from services.classifier.classifier import LowConfidenceResult, TicketClassifier

VALID_CATEGORIES = ["Auth", "Billing", "Outage", "Performance", "Security", "Feature Request"]
VALID_SEVERITIES = ["P0", "P1", "P2", "P3"]


@dataclass
class DatasetRow:
    ticket_id: str
    title: str
    description: str
    category: str
    severity: str
    source: str


def _load_dataset(path: Path) -> list[DatasetRow]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    if path.suffix.lower() == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            items = raw.get("tickets", [])
        else:
            items = raw
    elif path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            items = list(csv.DictReader(f))
    else:
        raise ValueError("Unsupported dataset format. Use .json or .csv")

    rows: list[DatasetRow] = []
    for i, item in enumerate(items):
        ticket_id = str(item.get("ticket_id") or f"dataset-{i+1}")
        rows.append(
            DatasetRow(
                ticket_id=ticket_id,
                title=str(item.get("title", "")).strip(),
                description=str(item.get("description", "")).strip(),
                category=str(item.get("category", "")).strip(),
                severity=str(item.get("severity", "")).strip(),
                source=str(item.get("source", "unknown")).strip(),
            )
        )

    return rows


def _safe_div(n: int, d: int) -> float:
    return round((n / d), 4) if d else 0.0


def _per_class_pr(y_true: list[str], y_pred: list[str], labels: list[str]) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)

        metrics[label] = {
            "precision": _safe_div(tp, tp + fp),
            "recall": _safe_div(tp, tp + fn),
            "support": sum(1 for t in y_true if t == label),
        }
    return metrics


def run(dataset_path: Path, output_path: Path) -> dict[str, Any]:
    rows = _load_dataset(dataset_path)

    classifier = TicketClassifier()

    predictions: list[dict[str, Any]] = []
    gt_categories: list[str] = []
    gt_severities: list[str] = []
    pred_categories: list[str] = []
    pred_severities: list[str] = []

    low_confidence_count = 0

    for row in rows:
        result = classifier.classify(ticket_id=row.ticket_id, title=row.title, description=row.description)

        if isinstance(result, LowConfidenceResult):
            low_confidence_count += 1

        gt_categories.append(row.category)
        gt_severities.append(row.severity)
        pred_categories.append(result.category)
        pred_severities.append(result.severity)

        predictions.append(
            {
                "ticket_id": row.ticket_id,
                "source": row.source,
                "ground_truth": {
                    "category": row.category,
                    "severity": row.severity,
                },
                "prediction": {
                    "category": result.category,
                    "severity": result.severity,
                    "confidence": result.confidence,
                    "model_version": result.model_version,
                    "prompt_version": result.prompt_version,
                },
            }
        )

    category_correct = sum(1 for t, p in zip(gt_categories, pred_categories) if t == p)
    severity_correct = sum(1 for t, p in zip(gt_severities, pred_severities) if t == p)

    result_json = {
        "checkpoint": "CP2-SA-04",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "path": str(dataset_path),
            "total_tickets": len(rows),
        },
        "metrics": {
            "category_accuracy": _safe_div(category_correct, len(rows)),
            "severity_accuracy": _safe_div(severity_correct, len(rows)),
            "target_category_accuracy": 0.70,
            "meets_target": _safe_div(category_correct, len(rows)) >= 0.70,
            "low_confidence_count": low_confidence_count,
            "low_confidence_rate": _safe_div(low_confidence_count, len(rows)),
            "category_per_class": _per_class_pr(gt_categories, pred_categories, VALID_CATEGORIES),
            "severity_per_class": _per_class_pr(gt_severities, pred_severities, VALID_SEVERITIES),
        },
        "predictions": predictions,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result_json, indent=2), encoding="utf-8")

    return result_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CP2 baseline evaluation on Dataset v1")
    parser.add_argument("--dataset", default="data/ticket_dataset_v1.json", help="Path to labeled dataset (.json or .csv)")
    parser.add_argument("--output", default="evaluation/checkpoint2_baseline.json", help="Path to output JSON")
    args = parser.parse_args()

    result = run(Path(args.dataset), Path(args.output))
    print(
        json.dumps(
            {
                "category_accuracy": result["metrics"]["category_accuracy"],
                "severity_accuracy": result["metrics"]["severity_accuracy"],
                "total_tickets": result["dataset"]["total_tickets"],
                "output": args.output,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
