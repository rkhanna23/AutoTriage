"""
CP3-RK-05: Integration tests for the Router Service.
Covers POST /route, GET /routes, valid/invalid category-severity combos.
"""
import pytest
from fastapi.testclient import TestClient

from services.router.main import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# POST /route — valid category/severity
# ---------------------------------------------------------------------------

def test_route_auth_p0():
    """Auth + P0 routes to Identity-Oncall."""
    resp = client.post(
        "/route",
        json={"ticket_id": "t-1", "category": "Auth", "severity": "P0"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["assigned_team"] == "Identity-Oncall"
    assert body["assigned_queue"] == "auth-p0"


def test_route_billing_p2():
    """Billing + P2 routes to Finance-Queue."""
    resp = client.post(
        "/route",
        json={"ticket_id": "t-2", "category": "Billing", "severity": "P2"},
    )
    assert resp.status_code == 200
    assert resp.json()["assigned_team"] == "Finance-Queue"
    assert resp.json()["assigned_queue"] == "billing-p2"


def test_route_security_p0():
    """Security + P0 routes to Security-Oncall."""
    resp = client.post(
        "/route",
        json={"ticket_id": "t-3", "category": "Security", "severity": "P0"},
    )
    assert resp.status_code == 200
    assert resp.json()["assigned_team"] == "Security-Oncall"


def test_route_unknown_unknown():
    """Unknown + Unknown routes to Manual-Triage."""
    resp = client.post(
        "/route",
        json={"ticket_id": "t-4", "category": "Unknown", "severity": "Unknown"},
    )
    assert resp.status_code == 200
    assert resp.json()["assigned_team"] == "Manual-Triage"
    assert resp.json()["assigned_queue"] == "manual-triage"


def test_route_invalid_combo_falls_back():
    """Unmapped (Foo, Bar) falls back to Manual-Triage."""
    resp = client.post(
        "/route",
        json={"ticket_id": "t-5", "category": "Foo", "severity": "Bar"},
    )
    assert resp.status_code == 200
    assert resp.json()["assigned_team"] == "Manual-Triage"


# ---------------------------------------------------------------------------
# GET /routes
# ---------------------------------------------------------------------------

def test_get_routes_lists_all_rules():
    """GET /routes returns all routing rules."""
    resp = client.get("/routes")
    assert resp.status_code == 200
    body = resp.json()
    assert "rules" in body
    assert len(body["rules"]) > 0
    rule = body["rules"][0]
    assert "category" in rule
    assert "severity" in rule
    assert "team" in rule
    assert "queue" in rule


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

def test_router_health():
    """GET /health returns ok."""
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    assert resp.json()["service"] == "router"


# ---------------------------------------------------------------------------
# POST /route — validation
# ---------------------------------------------------------------------------

def test_route_missing_ticket_id_returns_422():
    """Missing ticket_id returns 422."""
    resp = client.post(
        "/route",
        json={"category": "Auth", "severity": "P0"},
    )
    assert resp.status_code == 422


def test_route_empty_category_returns_422():
    """Empty category returns 422."""
    resp = client.post(
        "/route",
        json={"ticket_id": "t-1", "category": "", "severity": "P0"},
    )
    assert resp.status_code == 422
