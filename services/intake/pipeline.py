"""
CP3-RK-01, CP3-RK-03: Pipeline to classify and route tickets.
Calls classifier service, then router service. Handles timeouts/errors gracefully.
"""
import os
import time
from typing import Any, Optional

import httpx

CLASSIFIER_URL = os.getenv("CLASSIFIER_SERVICE_URL", "http://localhost:8001")
ROUTER_URL = os.getenv("ROUTER_SERVICE_URL", "http://localhost:8002")
CLASSIFIER_TIMEOUT = float(os.getenv("CLASSIFIER_TIMEOUT", "65.0"))
ROUTER_TIMEOUT = float(os.getenv("ROUTER_TIMEOUT", "5.0"))


def _ms(seconds: float) -> int:
    """Convert seconds to milliseconds."""
    return int(round(seconds * 1000))


def classify_ticket(
    ticket_id: str,
    title: str,
    description: str,
) -> tuple[dict[str, Any], Optional[int], bool]:
    """
    Call classifier service. Returns (result_dict, classify_ms, success).
    On timeout/error: returns low-confidence result and success=False.
    """
    url = f"{CLASSIFIER_URL.rstrip('/')}/classify"
    payload = {"ticket_id": ticket_id, "title": title, "description": description}
    fallback = {
        "ticket_id": ticket_id,
        "category": None,
        "severity": None,
        "confidence": 0.0,
        "model_version": "",
        "prompt_version": "",
        "needs_review": True,
    }
    start = time.perf_counter()
    try:
        with httpx.Client(timeout=CLASSIFIER_TIMEOUT) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            elapsed_ms = _ms(time.perf_counter() - start)
            return (
                {
                    "ticket_id": data.get("ticket_id", ticket_id),
                    "category": data.get("category"),
                    "severity": data.get("severity"),
                    "confidence": data.get("confidence", 0.0),
                    "model_version": data.get("model_version", ""),
                    "prompt_version": data.get("prompt_version", ""),
                    "needs_review": bool(data.get("needs_review", False)),
                },
                elapsed_ms,
                True,
            )
    except (httpx.HTTPError, httpx.TimeoutException):
        elapsed_ms = _ms(time.perf_counter() - start)
        return fallback, elapsed_ms, False


def route_ticket(
    ticket_id: str,
    category: Optional[str],
    severity: Optional[str],
) -> tuple[dict[str, Any], Optional[int], bool]:
    """
    Call router service. Returns (result_dict, route_ms, success).
    On timeout/error: returns empty assignment and success=False.
    """
    url = f"{ROUTER_URL.rstrip('/')}/route"
    payload = {
        "ticket_id": ticket_id,
        "category": category or "Unknown",
        "severity": severity or "Unknown",
    }
    fallback = {"assigned_team": None, "assigned_queue": None}
    start = time.perf_counter()
    try:
        with httpx.Client(timeout=ROUTER_TIMEOUT) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            elapsed_ms = _ms(time.perf_counter() - start)
            return (
                {
                    "assigned_team": data.get("assigned_team"),
                    "assigned_queue": data.get("assigned_queue"),
                },
                elapsed_ms,
                True,
            )
    except (httpx.HTTPError, httpx.TimeoutException):
        elapsed_ms = _ms(time.perf_counter() - start)
        return fallback, elapsed_ms, False
