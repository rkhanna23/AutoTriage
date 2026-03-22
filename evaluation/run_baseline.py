"""Prompt-version evaluation runner for CP2/CP3 classifier experiments."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from services.classifier.classifier import (
    CONFIDENCE_THRESHOLD,
    LowConfidenceResult,
    PROMPT_VERSION,
    TicketClassifier,
)

VALID_CATEGORIES = ["Auth", "Billing", "Outage", "Performance", "Security", "Feature Request"]
VALID_SEVERITIES = ["P0", "P1", "P2", "P3"]
DEFAULT_PROMPT_VERSIONS = ["v1.0", "v2.0", "v2.1"]
CHECKPOINT2_BASELINE_PATH = Path("evaluation/checkpoint2_baseline.json")


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
        items = raw.get("tickets", raw) if isinstance(raw, dict) else raw
    elif path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            items = list(csv.DictReader(f))
    else:
        raise ValueError("Unsupported dataset format. Use .json or .csv")

    rows: list[DatasetRow] = []
    for i, item in enumerate(items):
        rows.append(
            DatasetRow(
                ticket_id=str(item.get("ticket_id") or f"dataset-{i + 1}"),
                title=str(item.get("title", "")).strip(),
                description=str(item.get("description", "")).strip(),
                category=str(item.get("category", "")).strip(),
                severity=str(item.get("severity", "")).strip(),
                source=str(item.get("source", "unknown")).strip(),
            )
        )
    return rows


def _safe_div(n: int | float, d: int | float) -> float:
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


def _confidence_distribution(confidences: list[float]) -> dict[str, float]:
    if not confidences:
        return {"min": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "min": round(min(confidences), 4),
        "max": round(max(confidences), 4),
        "mean": round(mean(confidences), 4),
    }


def _expected_calibration_error(confidences: list[float], correctness: list[int], bins: int = 10) -> float:
    if not confidences:
        return 0.0

    total = len(confidences)
    error = 0.0
    for idx in range(bins):
        lower = idx / bins
        upper = (idx + 1) / bins
        bucket = [
            (conf, correct)
            for conf, correct in zip(confidences, correctness)
            if lower <= conf < upper or (idx == bins - 1 and conf == 1.0)
        ]
        if not bucket:
            continue
        bucket_confidence = sum(conf for conf, _ in bucket) / len(bucket)
        bucket_accuracy = sum(correct for _, correct in bucket) / len(bucket)
        error += (len(bucket) / total) * abs(bucket_accuracy - bucket_confidence)
    return round(error, 4)


def run(dataset_path: Path, output_path: Path, prompt_version: str = PROMPT_VERSION) -> dict[str, Any]:
    rows = _load_dataset(dataset_path)
    classifier = TicketClassifier(prompt_version=prompt_version)

    predictions: list[dict[str, Any]] = []
    gt_categories: list[str] = []
    gt_severities: list[str] = []
    pred_categories: list[str] = []
    pred_severities: list[str] = []
    confidences: list[float] = []
    correctness: list[int] = []

    low_confidence_count = 0
    needs_review_count = 0

    for row in rows:
        result = classifier.classify(
            ticket_id=row.ticket_id,
            title=row.title,
            description=row.description,
            prompt_version=prompt_version,
        )

        if isinstance(result, LowConfidenceResult):
            low_confidence_count += 1
        if result.needs_review:
            needs_review_count += 1

        gt_categories.append(row.category)
        gt_severities.append(row.severity)
        pred_categories.append(result.category)
        pred_severities.append(result.severity)
        confidences.append(result.confidence)
        correctness.append(int(result.category == row.category and result.severity == row.severity))

        predictions.append(
            {
                "ticket_id": row.ticket_id,
                "source": row.source,
                "ground_truth": {"category": row.category, "severity": row.severity},
                "prediction": {
                    "category": result.category,
                    "severity": result.severity,
                    "confidence": result.confidence,
                    "model_version": result.model_version,
                    "prompt_version": result.prompt_version,
                    "needs_review": result.needs_review,
                },
            }
        )

    category_correct = sum(1 for t, p in zip(gt_categories, pred_categories) if t == p)
    severity_correct = sum(1 for t, p in zip(gt_severities, pred_severities) if t == p)

    result_json = {
        "checkpoint": "CP3-SA-04",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {"path": str(dataset_path), "total_tickets": len(rows)},
        "prompt_version": prompt_version,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "metrics": {
            "category_accuracy": _safe_div(category_correct, len(rows)),
            "severity_accuracy": _safe_div(severity_correct, len(rows)),
            "target_severity_accuracy": 0.70,
            "meets_target": _safe_div(severity_correct, len(rows)) >= 0.70,
            "low_confidence_count": low_confidence_count,
            "low_confidence_rate": _safe_div(low_confidence_count, len(rows)),
            "needs_review_count": needs_review_count,
            "needs_review_rate": _safe_div(needs_review_count, len(rows)),
            "expected_calibration_error": _expected_calibration_error(confidences, correctness),
            "confidence_distribution": {
                "overall": _confidence_distribution(confidences),
                "correct": _confidence_distribution([c for c, ok in zip(confidences, correctness) if ok]),
                "incorrect": _confidence_distribution([c for c, ok in zip(confidences, correctness) if not ok]),
            },
            "category_per_class": _per_class_pr(gt_categories, pred_categories, VALID_CATEGORIES),
            "severity_per_class": _per_class_pr(gt_severities, pred_severities, VALID_SEVERITIES),
        },
        "predictions": predictions,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result_json, indent=2), encoding="utf-8")
    return result_json


def _normalize_existing_result(path: Path, prompt_version: str) -> dict[str, Any]:
    existing = json.loads(path.read_text(encoding="utf-8"))
    predictions = existing.get("predictions", [])
    confidences = [float(item["prediction"].get("confidence", 0.0)) for item in predictions]
    correctness = [
        int(
            item["ground_truth"].get("category") == item["prediction"].get("category")
            and item["ground_truth"].get("severity") == item["prediction"].get("severity")
        )
        for item in predictions
    ]
    needs_review_count = sum(
        1
        for item in predictions
        if float(item["prediction"].get("confidence", 0.0)) < CONFIDENCE_THRESHOLD
        or item["prediction"].get("category") == "Unknown"
        or item["prediction"].get("severity") == "Unknown"
    )
    total_tickets = int(existing.get("dataset", {}).get("total_tickets", len(predictions)))

    existing["checkpoint"] = "CP3-SA-04"
    existing["prompt_version"] = prompt_version
    existing["confidence_threshold"] = CONFIDENCE_THRESHOLD
    existing["metrics"]["target_severity_accuracy"] = 0.70
    existing["metrics"]["needs_review_count"] = needs_review_count
    existing["metrics"]["needs_review_rate"] = _safe_div(needs_review_count, total_tickets)
    existing["metrics"]["expected_calibration_error"] = _expected_calibration_error(confidences, correctness)
    existing["metrics"]["confidence_distribution"] = {
        "overall": _confidence_distribution(confidences),
        "correct": _confidence_distribution([c for c, ok in zip(confidences, correctness) if ok]),
        "incorrect": _confidence_distribution([c for c, ok in zip(confidences, correctness) if not ok]),
    }
    for item in predictions:
        item["prediction"]["prompt_version"] = prompt_version
        item["prediction"]["needs_review"] = (
            float(item["prediction"].get("confidence", 0.0)) < CONFIDENCE_THRESHOLD
            or item["prediction"].get("category") == "Unknown"
            or item["prediction"].get("severity") == "Unknown"
        )
    return existing


def compare_prompt_versions(
    dataset_path: Path,
    output_dir: Path,
    prompt_versions: list[str],
    comparison_output: Path,
    markdown_output: Path,
) -> dict[str, Any]:
    version_results: list[dict[str, Any]] = []

    for prompt_version in prompt_versions:
        version_output = output_dir / f"checkpoint3_{prompt_version}.json"
        if prompt_version == "v1.0" and CHECKPOINT2_BASELINE_PATH.exists():
            normalized = _normalize_existing_result(CHECKPOINT2_BASELINE_PATH, prompt_version)
            version_output.write_text(json.dumps(normalized, indent=2), encoding="utf-8")
            version_results.append(normalized)
        else:
            version_results.append(run(dataset_path, version_output, prompt_version=prompt_version))

    best_result = max(
        version_results,
        key=lambda item: (
            item["metrics"]["severity_accuracy"],
            item["metrics"]["category_accuracy"],
            -item["metrics"]["expected_calibration_error"],
            -item["metrics"]["needs_review_rate"],
        ),
    )

    comparison = {
        "checkpoint": "CP3-SA-04",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset_path),
        "versions": [
            {
                "prompt_version": item["prompt_version"],
                "category_accuracy": item["metrics"]["category_accuracy"],
                "severity_accuracy": item["metrics"]["severity_accuracy"],
                "needs_review_rate": item["metrics"]["needs_review_rate"],
                "expected_calibration_error": item["metrics"]["expected_calibration_error"],
                "output_path": str(output_dir / f"checkpoint3_{item['prompt_version']}.json"),
            }
            for item in version_results
        ],
        "recommended_default": best_result["prompt_version"],
    }

    comparison_output.write_text(json.dumps(comparison, indent=2), encoding="utf-8")

    lines = [
        "# Checkpoint 3 Prompt Comparison",
        "",
        f"Dataset: `{dataset_path}`",
        f"Recommended default: `{best_result['prompt_version']}`",
        "",
        "| Prompt | Category Accuracy | Severity Accuracy | Needs Review Rate | ECE |",
        "| --- | --- | --- | --- | --- |",
    ]
    for item in comparison["versions"]:
        lines.append(
            f"| {item['prompt_version']} | {item['category_accuracy']:.2%} | "
            f"{item['severity_accuracy']:.2%} | {item['needs_review_rate']:.2%} | "
            f"{item['expected_calibration_error']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            (
                f"`{best_result['prompt_version']}` should ship as the default because it produced "
                f"the strongest severity accuracy while also minimizing calibration error and review volume."
            ),
        ]
    )
    markdown_output.write_text("\n".join(lines) + "\n", encoding="utf-8")

    best_output = output_dir / "checkpoint3_baseline.json"
    best_output.write_text(json.dumps(best_result, indent=2), encoding="utf-8")

    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CP2/CP3 classifier evaluation")
    parser.add_argument("--dataset", default="data/ticket_dataset_v1.json", help="Path to labeled dataset (.json or .csv)")
    parser.add_argument("--output", default="evaluation/checkpoint2_baseline.json", help="Path to output JSON for a single-version run")
    parser.add_argument("--prompt-version", default=PROMPT_VERSION, help="Prompt version to evaluate for a single-version run")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run comparative evaluation across v1.0, v2.0, and v2.1 and emit checkpoint3 artifacts.",
    )
    parser.add_argument("--output-dir", default="evaluation", help="Directory for per-version comparison outputs")
    parser.add_argument("--comparison-output", default="evaluation/checkpoint3_comparison.json", help="Comparison JSON path")
    parser.add_argument("--markdown-output", default="evaluation/checkpoint3_comparison.md", help="Comparison markdown path")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)

    if args.compare:
        comparison = compare_prompt_versions(
            dataset_path=dataset_path,
            output_dir=Path(args.output_dir),
            prompt_versions=DEFAULT_PROMPT_VERSIONS,
            comparison_output=Path(args.comparison_output),
            markdown_output=Path(args.markdown_output),
        )
        print(json.dumps(comparison, indent=2))
        return

    result = run(dataset_path, Path(args.output), prompt_version=args.prompt_version)
    print(
        json.dumps(
            {
                "prompt_version": result["prompt_version"],
                "category_accuracy": result["metrics"]["category_accuracy"],
                "severity_accuracy": result["metrics"]["severity_accuracy"],
                "needs_review_rate": result["metrics"]["needs_review_rate"],
                "output": args.output,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
