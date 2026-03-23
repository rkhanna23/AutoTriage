"""
CP3-RK-05: Integration tests for the end-to-end pipeline and router.
Covers: auto-classify, auto-route, classifier failure, routing combos, latency fields.
Uses mocks so tests pass in CI without classifier/router services.
"""
import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient

from tests.conftest import client

VALID_PAYLOAD = {
    "title": "Login page returns 500",
    "description": "Users are unable to log in. The /auth/login endpoint returns HTTP 500.",
    "source": "web",
    "timestamp": "2024-06-01T10:00:00Z",
}




# ---------------------------------------------------------------------------
# CP3-RK-05 (1): Submit ticket and verify it gets auto-classified
# ---------------------------------------------------------------------------

def test_submit_ticket_gets_auto_classified():
    """When classifier succeeds, ticket has classification_status=classified and category/severity."""
    classify_result = (
        {
            "ticket_id": "will-be-replaced",
            "category": "Auth",
            "severity": "P1",
            "confidence": 0.92,
            "model_version": "llama3.1:8b",
            "prompt_version": "v1.0",
            "needs_review": False,
        },
        150,
        True,
    )
    route_result = (
        {"assigned_team": "Identity-Oncall", "assigned_queue": "auth-p1"},
        5,
        True,
    )
    with patch("services.intake.main.pipeline_classify", return_value=classify_result):
        with patch("services.intake.main.pipeline_route", return_value=route_result):
            resp = client.post("/tickets", json=VALID_PAYLOAD)
    assert resp.status_code == 201
    body = resp.json()
    assert body["classification"]["classification_status"] == "classified"
    assert body["classification"]["category"] == "Auth"
    assert body["classification"]["severity"] == "P1"
    assert body["classification"]["confidence"] == 0.92
    assert body["classification"]["needs_review"] is False

    ticket_id = body["ticket_id"]
    get_resp = client.get(f"/tickets/{ticket_id}")
    assert get_resp.status_code == 200
    t = get_resp.json()
    assert t["classification"]["classification_status"] == "classified"
    assert t["classification"]["category"] == "Auth"
    assert t["classification"]["severity"] == "P1"


# ---------------------------------------------------------------------------
# CP3-RK-05 (2): Submit ticket and verify it gets auto-routed
# ---------------------------------------------------------------------------

def test_submit_ticket_gets_auto_routed():
    """When router succeeds, ticket has routing_status=routed and assigned_team/queue."""
    classify_result = (
        {
            "ticket_id": "x",
            "category": "Billing",
            "severity": "P2",
            "confidence": 0.88,
            "model_version": "llama3.1:8b",
            "prompt_version": "v1.0",
            "needs_review": False,
        },
        120,
        True,
    )
    route_result = (
        {"assigned_team": "Finance-Queue", "assigned_queue": "billing-p2"},
        3,
        True,
    )
    with patch("services.intake.main.pipeline_classify", return_value=classify_result):
        with patch("services.intake.main.pipeline_route", return_value=route_result):
            resp = client.post("/tickets", json=VALID_PAYLOAD)
    assert resp.status_code == 201
    body = resp.json()
    assert body["routing"]["routing_status"] == "routed"
    assert body["routing"]["assigned_team"] == "Finance-Queue"
    assert body["routing"]["assigned_queue"] == "billing-p2"

    ticket_id = body["ticket_id"]
    get_resp = client.get(f"/tickets/{ticket_id}")
    assert get_resp.json()["routing"]["assigned_team"] == "Finance-Queue"


# ---------------------------------------------------------------------------
# CP3-RK-05 (3): Classifier timeout/failure results in classification_status=pending
# ---------------------------------------------------------------------------

def test_classifier_failure_sets_classification_status_pending():
    """When classifier times out or errors, classification_status=pending for retry."""
    classify_result = (
        {
            "ticket_id": "x",
            "category": None,
            "severity": None,
            "confidence": 0.0,
            "model_version": "",
            "prompt_version": "",
            "needs_review": True,
        },
        65000,
        False,
    )
    route_result = (
        {"assigned_team": "Manual-Triage", "assigned_queue": "manual-triage"},
        2,
        True,
    )
    with patch("services.intake.main.pipeline_classify", return_value=classify_result):
        with patch("services.intake.main.pipeline_route", return_value=route_result):
            resp = client.post("/tickets", json=VALID_PAYLOAD)
    assert resp.status_code == 201
    body = resp.json()
    assert body["classification"] is None

    ticket_id = body["ticket_id"]
    get_resp = client.get(f"/tickets/{ticket_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["classification"]["classification_status"] == "pending"


# ---------------------------------------------------------------------------
# CP3-RK-05 (4): Routing with valid and invalid category/severity combos
# ---------------------------------------------------------------------------

def test_routing_valid_category_severity():
    """Valid (Auth, P0) routes to Identity-Oncall."""
    classify_result = (
        {"ticket_id": "x", "category": "Auth", "severity": "P0", "confidence": 0.95, "model_version": "m", "prompt_version": "v", "needs_review": False},
        100,
        True,
    )
    route_result = (
        {"assigned_team": "Identity-Oncall", "assigned_queue": "auth-p0"},
        1,
        True,
    )
    with patch("services.intake.main.pipeline_classify", return_value=classify_result):
        with patch("services.intake.main.pipeline_route", return_value=route_result):
            resp = client.post("/tickets", json=VALID_PAYLOAD)
    assert resp.status_code == 201
    assert resp.json()["routing"]["assigned_team"] == "Identity-Oncall"


def test_routing_unknown_category_severity_falls_back():
    """Unknown/Unknown routes to Manual-Triage (router fallback)."""
    classify_result = (
        {"ticket_id": "x", "category": "Unknown", "severity": "Unknown", "confidence": 0.3, "model_version": "m", "prompt_version": "v", "needs_review": True},
        80,
        True,
    )
    route_result = (
        {"assigned_team": "Manual-Triage", "assigned_queue": "manual-triage"},
        1,
        True,
    )
    with patch("services.intake.main.pipeline_classify", return_value=classify_result):
        with patch("services.intake.main.pipeline_route", return_value=route_result):
            resp = client.post("/tickets", json=VALID_PAYLOAD)
    assert resp.status_code == 201
    assert resp.json()["routing"]["assigned_team"] == "Manual-Triage"


# ---------------------------------------------------------------------------
# CP3-RK-05 (5): Latency fields are populated
# ---------------------------------------------------------------------------

def test_latency_fields_populated():
    """Latency fields (intake_ms, classify_ms, route_ms, total_ms) are populated."""
    classify_result = (
        {"ticket_id": "x", "category": "Auth", "severity": "P1", "confidence": 0.9, "model_version": "m", "prompt_version": "v", "needs_review": False},
        200,
        True,
    )
    route_result = (
        {"assigned_team": "Identity-Oncall", "assigned_queue": "auth-p1"},
        10,
        True,
    )
    with patch("services.intake.main.pipeline_classify", return_value=classify_result):
        with patch("services.intake.main.pipeline_route", return_value=route_result):
            resp = client.post("/tickets", json=VALID_PAYLOAD)
    assert resp.status_code == 201
    body = resp.json()
    assert "latency" in body
    assert body["latency"]["intake_ms"] is not None
    assert body["latency"]["classify_ms"] == 200
    assert body["latency"]["route_ms"] == 10
    assert body["latency"]["total_ms"] is not None
    # With mocks, total_ms is real elapsed time (can be small); classify_ms/route_ms come from mock
    assert isinstance(body["latency"]["total_ms"], (int, float))


# ---------------------------------------------------------------------------
# Additional CP3-RK-05 tests
# ---------------------------------------------------------------------------

def test_get_ticket_includes_routing_info():
    """GET /tickets/:id includes routing information."""
    classify_result = (
        {"ticket_id": "x", "category": "Security", "severity": "P0", "confidence": 0.99, "model_version": "m", "prompt_version": "v", "needs_review": False},
        100,
        True,
    )
    route_result = (
        {"assigned_team": "Security-Oncall", "assigned_queue": "security-p0"},
        2,
        True,
    )
    with patch("services.intake.main.pipeline_classify", return_value=classify_result):
        with patch("services.intake.main.pipeline_route", return_value=route_result):
            resp = client.post("/tickets", json=VALID_PAYLOAD)
    ticket_id = resp.json()["ticket_id"]
    get_resp = client.get(f"/tickets/{ticket_id}")
    assert get_resp.status_code == 200
    t = get_resp.json()
    assert t["routing"]["assigned_team"] == "Security-Oncall"
    assert t["routing"]["assigned_queue"] == "security-p0"
    assert t["latency"]["total_ms"] is not None


def test_list_tickets_includes_classification_and_routing():
    """GET /tickets list includes classification and routing for each ticket."""
    classify_result = (
        {"ticket_id": "x", "category": "Outage", "severity": "P0", "confidence": 0.95, "model_version": "m", "prompt_version": "v", "needs_review": False},
        100,
        True,
    )
    route_result = (
        {"assigned_team": "SRE-Oncall", "assigned_queue": "outage-p0"},
        2,
        True,
    )
    with patch("services.intake.main.pipeline_classify", return_value=classify_result):
        with patch("services.intake.main.pipeline_route", return_value=route_result):
            client.post("/tickets", json=VALID_PAYLOAD)
    resp = client.get("/tickets")
    assert resp.status_code == 200
    tickets = resp.json()["tickets"]
    assert len(tickets) >= 1
    t = tickets[0]
    assert "classification" in t
    assert "routing" in t
    assert t["classification"]["category"] == "Outage"
    assert t["routing"]["assigned_team"] == "SRE-Oncall"


def test_router_failure_sets_routing_status_failed():
    """When router fails, routing_status=failed."""
    classify_result = (
        {"ticket_id": "x", "category": "Auth", "severity": "P1", "confidence": 0.9, "model_version": "m", "prompt_version": "v", "needs_review": False},
        100,
        True,
    )
    route_result = (
        {"assigned_team": None, "assigned_queue": None},
        5000,
        False,
    )
    with patch("services.intake.main.pipeline_classify", return_value=classify_result):
        with patch("services.intake.main.pipeline_route", return_value=route_result):
            resp = client.post("/tickets", json=VALID_PAYLOAD)
    assert resp.status_code == 201
    assert resp.json()["routing"]["routing_status"] == "failed"


def test_metrics_latency_endpoint():
    """GET /metrics/latency returns p50, p95, p99 for timing fields."""
    classify_result = (
        {"ticket_id": "x", "category": "Auth", "severity": "P1", "confidence": 0.9, "model_version": "m", "prompt_version": "v", "needs_review": False},
        100,
        True,
    )
    route_result = (
        {"assigned_team": "Identity-Oncall", "assigned_queue": "auth-p1"},
        5,
        True,
    )
    with patch("services.intake.main.pipeline_classify", return_value=classify_result):
        with patch("services.intake.main.pipeline_route", return_value=route_result):
            client.post("/tickets", json=VALID_PAYLOAD)
    resp = client.get("/metrics/latency?n=100")
    assert resp.status_code == 200
    body = resp.json()
    assert "sample_size" in body
    assert body["sample_size"] >= 1
    assert "intake_ms" in body
    assert "p50" in body["intake_ms"]
    assert "p95" in body["intake_ms"]
    assert "p99" in body["intake_ms"]
    assert "total_ms" in body
    assert "p50" in body["total_ms"]


def test_metrics_latency_empty_when_no_tickets():
    """GET /metrics/latency returns empty percentiles when no tickets have timing data."""
    resp = client.get("/metrics/latency?n=100")
    assert resp.status_code == 200
    body = resp.json()
    assert body["sample_size"] == 0
    assert body["intake_ms"]["p50"] is None
    assert body["total_ms"]["p50"] is None
