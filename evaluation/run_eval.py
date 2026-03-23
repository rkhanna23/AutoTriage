"""Checkpoint 3 evaluation harness for dataset v2, routing, and latency metrics."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from evaluation.run_baseline import _load_dataset, _safe_div
from services.classifier.classifier import PROMPT_VERSION, TicketClassifier
from services.router.main import _get_rule_map

DEFAULT_DATASET = Path("data/ticket_dataset_v2.json")
DEFAULT_OUTPUT = Path("evaluation/checkpoint3_results.json")


def _percentile(sorted_values: list[float], p: float) -> float | None:
    if not sorted_values:
        return None
    k = (len(sorted_values) - 1) * (p / 100)
    f = int(k)
    c = min(f + 1, len(sorted_values) - 1)
    return round(sorted_values[f] + (k - f) * (sorted_values[c] - sorted_values[f]), 3)


def _latency_summary(values: list[float]) -> dict[str, float | None]:
    ordered = sorted(values)
    return {
        "p50": _percentile(ordered, 50),
        "p95": _percentile(ordered, 95),
        "p99": _percentile(ordered, 99),
    }


def _route_assignment(category: str | None, severity: str | None) -> dict[str, str]:
    rule_map = _get_rule_map()
    fallback = rule_map.get(("Unknown", "Unknown"), {"team": "Manual-Triage", "queue": "manual-triage"})
    if category is None or severity is None:
        return {"assigned_team": fallback["team"], "assigned_queue": fallback["queue"]}
    rule = rule_map.get((category, severity), fallback)
    return {"assigned_team": rule["team"], "assigned_queue": rule["queue"]}


def _confusion_matrix(labels: list[str], truth: list[str], pred: list[str]) -> dict[str, Any]:
    index = {label: i for i, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for actual, predicted in zip(truth, pred):
        if actual in index and predicted in index:
            matrix[index[actual]][index[predicted]] += 1
    return {"labels": labels, "matrix": matrix}


def run(dataset_path: Path = DEFAULT_DATASET, output_path: Path = DEFAULT_OUTPUT, prompt_version: str = PROMPT_VERSION) -> dict[str, Any]:
    rows = _load_dataset(dataset_path)
    classifier = TicketClassifier(prompt_version=prompt_version, allow_fallback=False)

    gt_categories: list[str] = []
    gt_severities: list[str] = []
    pred_categories: list[str] = []
    pred_severities: list[str] = []
    classify_latencies: list[float] = []
    route_latencies: list[float] = []
    total_latencies: list[float] = []
    predictions: list[dict[str, Any]] = []

    routing_correct = 0
    predicted_routing_correct = 0

    for row in rows:
        start_total = time.perf_counter()

        start_classify = time.perf_counter()
        result = classifier.classify(
            ticket_id=row.ticket_id,
            title=row.title,
            description=row.description,
            prompt_version=prompt_version,
        )
        classify_ms = round((time.perf_counter() - start_classify) * 1000, 3)

        start_route = time.perf_counter()
        expected_route = _route_assignment(row.category, row.severity)
        routed_from_ground_truth = _route_assignment(row.category, row.severity)
        routed_from_prediction = _route_assignment(
            None if result.needs_review else result.category,
            None if result.needs_review else result.severity,
        )
        route_ms = round((time.perf_counter() - start_route) * 1000, 3)
        total_ms = round((time.perf_counter() - start_total) * 1000, 3)

        gt_categories.append(row.category)
        gt_severities.append(row.severity)
        pred_categories.append(result.category)
        pred_severities.append(result.severity)
        classify_latencies.append(classify_ms)
        route_latencies.append(route_ms)
        total_latencies.append(total_ms)

        routing_correct += int(routed_from_ground_truth == expected_route)
        predicted_routing_correct += int(routed_from_prediction == expected_route)

        predictions.append(
            {
                "ticket_id": row.ticket_id,
                "source": row.source,
                "ground_truth": {
                    "category": row.category,
                    "severity": row.severity,
                    "assigned_team": expected_route["assigned_team"],
                    "assigned_queue": expected_route["assigned_queue"],
                },
                "prediction": {
                    "category": result.category,
                    "severity": result.severity,
                    "confidence": result.confidence,
                    "model_version": result.model_version,
                    "prompt_version": result.prompt_version,
                    "needs_review": result.needs_review,
                    "assigned_team": routed_from_prediction["assigned_team"],
                    "assigned_queue": routed_from_prediction["assigned_queue"],
                },
                "latency_ms": {
                    "classify_ms": classify_ms,
                    "route_ms": route_ms,
                    "total_ms": total_ms,
                },
            }
        )

    category_accuracy = _safe_div(sum(1 for t, p in zip(gt_categories, pred_categories) if t == p), len(rows))
    severity_accuracy = _safe_div(sum(1 for t, p in zip(gt_severities, pred_severities) if t == p), len(rows))
    routing_accuracy = _safe_div(routing_correct, len(rows))
    predicted_routing_accuracy = _safe_div(predicted_routing_correct, len(rows))

    category_labels = ["Auth", "Billing", "Outage", "Performance", "Security", "Feature Request", "Unknown"]
    severity_labels = ["P0", "P1", "P2", "P3", "Unknown"]

    result_json = {
        "checkpoint": "CP3-KL-04",
        "dataset": str(dataset_path),
        "prompt_version": prompt_version,
        "total_tickets": len(rows),
        "category_accuracy": category_accuracy,
        "severity_accuracy": severity_accuracy,
        "routing_accuracy": routing_accuracy,
        "predicted_routing_accuracy": predicted_routing_accuracy,
        "p50_latency_ms": _latency_summary(total_latencies)["p50"],
        "p95_latency_ms": _latency_summary(total_latencies)["p95"],
        "p99_latency_ms": _latency_summary(total_latencies)["p99"],
        "latency_breakdown_ms": {
            "classify_ms": _latency_summary(classify_latencies),
            "route_ms": _latency_summary(route_latencies),
            "total_ms": _latency_summary(total_latencies),
        },
        "category_confusion_matrix": _confusion_matrix(category_labels, gt_categories, pred_categories),
        "severity_confusion_matrix": _confusion_matrix(severity_labels, gt_severities, pred_severities),
        "predictions": predictions,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result_json, indent=2), encoding="utf-8")
    return result_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Checkpoint 3 evaluation on Dataset v2")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET), help="Path to dataset v2 JSON")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output JSON path")
    parser.add_argument("--prompt-version", default=PROMPT_VERSION, help="Prompt version to evaluate")
    args = parser.parse_args()

    result = run(Path(args.dataset), Path(args.output), prompt_version=args.prompt_version)
    print(json.dumps({
        "dataset": result["dataset"],
        "prompt_version": result["prompt_version"],
        "category_accuracy": result["category_accuracy"],
        "severity_accuracy": result["severity_accuracy"],
        "routing_accuracy": result["routing_accuracy"],
        "predicted_routing_accuracy": result["predicted_routing_accuracy"],
        "p95_latency_ms": result["p95_latency_ms"],
        "output": args.output,
    }, indent=2))


if __name__ == "__main__":
    main()
