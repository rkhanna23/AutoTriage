"""
Integration tests for the AutoTriage Intake Service — CP2-RK-04
Covers all endpoints: POST /tickets, GET /tickets, GET /tickets/{id}
Happy path + error cases. Requires at least 12 tests.
"""
import pytest
from datetime import datetime, timezone
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from services.intake.main import app
from services.intake.database import Base, get_db

# ---------------------------------------------------------------------------
# Test database setup — isolated in-memory SQLite for each test session
# ---------------------------------------------------------------------------

TEST_DB_URL = "sqlite:///./test_autotriage.db"
test_engine = create_engine(TEST_DB_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db

Base.metadata.create_all(bind=test_engine)

client = TestClient(app)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

VALID_PAYLOAD = {
    "title": "Login page returns 500",
    "description": "Users are unable to log in. The /auth/login endpoint returns HTTP 500.",
    "source": "web",
    "timestamp": "2024-06-01T10:00:00Z",
}


@pytest.fixture(autouse=True)
def reset_db():
    """Wipe and recreate tables before every test for isolation."""
    Base.metadata.drop_all(bind=test_engine)
    Base.metadata.create_all(bind=test_engine)
    yield


# ---------------------------------------------------------------------------
# Test 1 — POST /tickets happy path
# ---------------------------------------------------------------------------

def test_create_ticket_returns_201():
    response = client.post("/tickets", json=VALID_PAYLOAD)
    assert response.status_code == 201
    body = response.json()
    assert body["acknowledged"] is True
    assert "ticket_id" in body
    assert len(body["ticket_id"]) == 36  # UUID v4


# ---------------------------------------------------------------------------
# Test 2 — POST /tickets persists and is retrievable
# ---------------------------------------------------------------------------

def test_created_ticket_is_retrievable():
    create_resp = client.post("/tickets", json=VALID_PAYLOAD)
    ticket_id = create_resp.json()["ticket_id"]

    get_resp = client.get(f"/tickets/{ticket_id}")
    assert get_resp.status_code == 200
    body = get_resp.json()
    assert body["ticket_id"] == ticket_id
    assert body["title"] == VALID_PAYLOAD["title"]
    assert body["description"] == VALID_PAYLOAD["description"]
    assert body["source"] == VALID_PAYLOAD["source"]
    assert body["status"] == "open"


# ---------------------------------------------------------------------------
# Test 3 — POST /tickets missing required field → 422
# ---------------------------------------------------------------------------

def test_create_ticket_missing_title_returns_422():
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "title"}
    response = client.post("/tickets", json=payload)
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Test 4 — POST /tickets blank title → 422
# ---------------------------------------------------------------------------

def test_create_ticket_blank_title_returns_422():
    payload = {**VALID_PAYLOAD, "title": "   "}
    response = client.post("/tickets", json=payload)
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Test 5 — POST /tickets missing description → 422
# ---------------------------------------------------------------------------

def test_create_ticket_missing_description_returns_422():
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "description"}
    response = client.post("/tickets", json=payload)
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Test 6 — POST /tickets invalid timestamp → 422
# ---------------------------------------------------------------------------

def test_create_ticket_invalid_timestamp_returns_422():
    payload = {**VALID_PAYLOAD, "timestamp": "not-a-date"}
    response = client.post("/tickets", json=payload)
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Test 7 — GET /tickets returns empty list when no tickets exist
# ---------------------------------------------------------------------------

def test_list_tickets_empty():
    response = client.get("/tickets")
    assert response.status_code == 200
    body = response.json()
    assert body["tickets"] == []
    assert body["pagination"]["total"] == 0
    assert body["pagination"]["has_more"] is False


# ---------------------------------------------------------------------------
# Test 8 — GET /tickets returns all submitted tickets
# ---------------------------------------------------------------------------

def test_list_tickets_returns_all():
    for i in range(3):
        client.post("/tickets", json={**VALID_PAYLOAD, "title": f"Ticket {i}"})
    response = client.get("/tickets")
    assert response.status_code == 200
    body = response.json()
    assert body["pagination"]["total"] == 3
    assert len(body["tickets"]) == 3


# ---------------------------------------------------------------------------
# Test 9 — GET /tickets filter by source
# ---------------------------------------------------------------------------

def test_list_tickets_filter_by_source():
    client.post("/tickets", json={**VALID_PAYLOAD, "source": "web"})
    client.post("/tickets", json={**VALID_PAYLOAD, "source": "mobile"})
    client.post("/tickets", json={**VALID_PAYLOAD, "source": "mobile"})

    response = client.get("/tickets?source=mobile")
    assert response.status_code == 200
    body = response.json()
    assert body["pagination"]["total"] == 2
    assert all(t["source"] == "mobile" for t in body["tickets"])


# ---------------------------------------------------------------------------
# Test 10 — GET /tickets pagination (limit and offset)
# ---------------------------------------------------------------------------

def test_list_tickets_pagination():
    for i in range(5):
        client.post("/tickets", json={**VALID_PAYLOAD, "title": f"Ticket {i}"})

    page1 = client.get("/tickets?limit=2&offset=0").json()
    page2 = client.get("/tickets?limit=2&offset=2").json()

    assert len(page1["tickets"]) == 2
    assert page1["pagination"]["has_more"] is True
    assert len(page2["tickets"]) == 2

    ids_p1 = {t["ticket_id"] for t in page1["tickets"]}
    ids_p2 = {t["ticket_id"] for t in page2["tickets"]}
    assert ids_p1.isdisjoint(ids_p2)


# ---------------------------------------------------------------------------
# Test 11 — GET /tickets/{id} unknown ID → 404
# ---------------------------------------------------------------------------

def test_get_ticket_not_found_returns_404():
    response = client.get("/tickets/00000000-0000-0000-0000-000000000000")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Test 12 — GET /tickets filter by invalid status → 422
# ---------------------------------------------------------------------------

def test_list_tickets_invalid_status_returns_422():
    response = client.get("/tickets?status=banana")
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Test 13 — GET /health returns ok
# ---------------------------------------------------------------------------

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


# ---------------------------------------------------------------------------
# Test 14 — POST /tickets empty body → 422
# ---------------------------------------------------------------------------

def test_create_ticket_empty_body_returns_422():
    response = client.post("/tickets", json={})
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Test 15 — GET /tickets date range filter
# ---------------------------------------------------------------------------

def test_list_tickets_date_range_filter():
    client.post("/tickets", json={**VALID_PAYLOAD, "timestamp": "2024-01-01T00:00:00Z"})
    client.post("/tickets", json={**VALID_PAYLOAD, "timestamp": "2024-06-15T00:00:00Z"})
    client.post("/tickets", json={**VALID_PAYLOAD, "timestamp": "2024-12-31T00:00:00Z"})

    response = client.get("/tickets?date_from=2024-06-01T00:00:00Z&date_to=2024-07-01T00:00:00Z")
    assert response.status_code == 200
    body = response.json()
    assert body["pagination"]["total"] == 1
