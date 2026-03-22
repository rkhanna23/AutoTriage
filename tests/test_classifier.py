import json

import httpx
from fastapi.testclient import TestClient

from services.classifier.classifier import LowConfidenceResult, TicketClassifier
from services.classifier.main import app


class _FakeHTTPResponse:
    def __init__(self, response_text: str):
        self._response_text = response_text

    def raise_for_status(self):
        return None

    def json(self):
        return {"response": self._response_text}


def test_parse_valid_category_and_severity():
    classifier = TicketClassifier(model="test-model", prompt_version="v1.0")
    raw = json.dumps(
        {
            "ticket_id": "abc-123",
            "category": "Auth",
            "severity": "P1",
            "confidence": 0.91,
        }
    )

    result = classifier._parse(ticket_id="abc-123", raw=raw)

    assert result.ticket_id == "abc-123"
    assert result.category == "Auth"
    assert result.severity == "P1"
    assert result.confidence == 0.91
    assert result.prompt_version == "v1.0"
    assert result.needs_review is False


def test_parse_invalid_severity_returns_none():
    classifier = TicketClassifier(model="test-model", prompt_version="v1.0")
    raw = json.dumps(
        {
            "ticket_id": "abc-123",
            "category": "Auth",
            "severity": "P9",
            "confidence": 0.92,
        }
    )

    result = classifier._parse(ticket_id="abc-123", raw=raw)

    assert result is None


def test_classify_falls_back_locally_on_transport_error(monkeypatch):
    classifier = TicketClassifier(base_url="http://bad-host", model="test-model", prompt_version="v2.1")

    def _raise(*args, **kwargs):
        raise httpx.ConnectError("boom")

    monkeypatch.setattr("services.classifier.classifier.httpx.post", _raise)

    result = classifier.classify(
        ticket_id="t-1",
        title="API gateway unreachable",
        description="All customer requests return 503 from every region",
    )

    assert result.ticket_id == "t-1"
    assert result.category == "Outage"
    assert result.severity == "P0"
    assert result.model_version == "test-model+local-fallback"
    assert result.prompt_version == "v2.1"
    assert result.needs_review is False


def test_classify_api_returns_exact_schema(monkeypatch):
    payload = json.dumps(
        {
            "ticket_id": "ticket-7",
            "category": "Security",
            "severity": "P0",
            "confidence": 0.88,
        }
    )

    def _fake_post(*args, **kwargs):
        return _FakeHTTPResponse(payload)

    monkeypatch.setattr("services.classifier.classifier.httpx.post", _fake_post)

    client = TestClient(app)
    response = client.post(
        "/classify",
        json={
            "ticket_id": "ticket-7",
            "title": "Possible data leak",
            "description": "Public bucket exposed private records",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert sorted(body.keys()) == [
        "category",
        "confidence",
        "model_version",
        "needs_review",
        "prompt_version",
        "severity",
        "ticket_id",
    ]
    assert body["ticket_id"] == "ticket-7"
    assert body["category"] == "Security"
    assert body["severity"] == "P0"
    assert body["needs_review"] is False


def test_classify_api_accepts_prompt_version(monkeypatch):
    payload = json.dumps(
        {
            "ticket_id": "ticket-8",
            "category": "Auth",
            "severity": "P1",
            "confidence": 0.9,
        }
    )

    def _fake_post(*args, **kwargs):
        return _FakeHTTPResponse(payload)

    monkeypatch.setattr("services.classifier.classifier.httpx.post", _fake_post)

    client = TestClient(app)
    response = client.post(
        "/classify",
        json={
            "ticket_id": "ticket-8",
            "title": "SSO callback loop",
            "description": "Users cannot sign in",
            "prompt_version": "v2.0",
        },
    )

    assert response.status_code == 200
    assert response.json()["prompt_version"] == "v2.0"
