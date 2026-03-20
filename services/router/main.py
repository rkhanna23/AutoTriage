"""
AutoTriage — Routing Engine
CP3-RK-02: Data model and API for (category, severity) -> (queue, team) mapping.
"""
import json
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROUTER_DIR = Path(__file__).parent
RULES_FILE = ROUTER_DIR / "routing_rules.json"

# Default rules: (category, severity) -> {team, queue}
DEFAULT_RULES = [
    {"category": "Auth", "severity": "P0", "team": "Identity-Oncall", "queue": "auth-p0"},
    {"category": "Auth", "severity": "P1", "team": "Identity-Oncall", "queue": "auth-p1"},
    {"category": "Auth", "severity": "P2", "team": "Identity-Queue", "queue": "auth-p2"},
    {"category": "Auth", "severity": "P3", "team": "Identity-Queue", "queue": "auth-p3"},
    {"category": "Billing", "severity": "P0", "team": "Finance-Oncall", "queue": "billing-p0"},
    {"category": "Billing", "severity": "P1", "team": "Finance-Oncall", "queue": "billing-p1"},
    {"category": "Billing", "severity": "P2", "team": "Finance-Queue", "queue": "billing-p2"},
    {"category": "Billing", "severity": "P3", "team": "Finance-Queue", "queue": "billing-p3"},
    {"category": "Outage", "severity": "P0", "team": "SRE-Oncall", "queue": "outage-p0"},
    {"category": "Outage", "severity": "P1", "team": "SRE-Oncall", "queue": "outage-p1"},
    {"category": "Outage", "severity": "P2", "team": "SRE-Queue", "queue": "outage-p2"},
    {"category": "Outage", "severity": "P3", "team": "SRE-Queue", "queue": "outage-p3"},
    {"category": "Performance", "severity": "P0", "team": "SRE-Oncall", "queue": "perf-p0"},
    {"category": "Performance", "severity": "P1", "team": "SRE-Oncall", "queue": "perf-p1"},
    {"category": "Performance", "severity": "P2", "team": "SRE-Queue", "queue": "perf-p2"},
    {"category": "Performance", "severity": "P3", "team": "SRE-Queue", "queue": "perf-p3"},
    {"category": "Security", "severity": "P0", "team": "Security-Oncall", "queue": "security-p0"},
    {"category": "Security", "severity": "P1", "team": "Security-Oncall", "queue": "security-p1"},
    {"category": "Security", "severity": "P2", "team": "Security-Queue", "queue": "security-p2"},
    {"category": "Security", "severity": "P3", "team": "Security-Queue", "queue": "security-p3"},
    {"category": "Feature Request", "severity": "P0", "team": "Product-Oncall", "queue": "feature-p0"},
    {"category": "Feature Request", "severity": "P1", "team": "Product-Queue", "queue": "feature-p1"},
    {"category": "Feature Request", "severity": "P2", "team": "Product-Queue", "queue": "feature-p2"},
    {"category": "Feature Request", "severity": "P3", "team": "Product-Queue", "queue": "feature-p3"},
    {"category": "Unknown", "severity": "Unknown", "team": "Manual-Triage", "queue": "manual-triage"},
]


def _load_rules() -> list[dict[str, Any]]:
    if RULES_FILE.exists():
        try:
            data = json.loads(RULES_FILE.read_text(encoding="utf-8"))
            return data.get("rules", data) if isinstance(data, dict) else data
        except (json.JSONDecodeError, OSError):
            pass
    return DEFAULT_RULES


def _save_rules(rules: list[dict[str, Any]]) -> None:
    RULES_FILE.write_text(json.dumps({"rules": rules}, indent=2), encoding="utf-8")


def _get_rule_map() -> dict[tuple[str, str], dict[str, str]]:
    rules = _load_rules()
    return {(r["category"], r["severity"]): r for r in rules}


app = FastAPI(
    title="AutoTriage Router Service",
    description="Routes classified tickets to response queues/teams.",
    version="1.0.0",
)


class RouteRequest(BaseModel):
    """Classified ticket for routing."""
    ticket_id: str = Field(..., min_length=1)
    category: str = Field(..., min_length=1)
    severity: str = Field(..., min_length=1)


class RouteResponse(BaseModel):
    """Routing assignment."""
    ticket_id: str
    assigned_team: str
    assigned_queue: str


class RoutingRule(BaseModel):
    """Single routing rule."""
    category: str
    severity: str
    team: str
    queue: str


class RoutesListResponse(BaseModel):
    """All routing rules."""
    rules: list[RoutingRule]


@app.post(
    "/route",
    response_model=RouteResponse,
    summary="Route a classified ticket to a team/queue",
    tags=["Router"],
)
def route_ticket(req: RouteRequest):
    """
    Accept a classified ticket and return the assigned team and queue
    based on (category, severity) mapping.
    """
    rule_map = _get_rule_map()
    key = (req.category, req.severity)
    if key in rule_map:
        r = rule_map[key]
        return RouteResponse(
            ticket_id=req.ticket_id,
            assigned_team=r["team"],
            assigned_queue=r["queue"],
        )
    # Fallback for Unknown or unmapped combos
    fallback = rule_map.get(("Unknown", "Unknown"))
    if fallback:
        return RouteResponse(
            ticket_id=req.ticket_id,
            assigned_team=fallback["team"],
            assigned_queue=fallback["queue"],
        )
    # Last resort
    return RouteResponse(
        ticket_id=req.ticket_id,
        assigned_team="Manual-Triage",
        assigned_queue="manual-triage",
    )


@app.get(
    "/routes",
    response_model=RoutesListResponse,
    summary="List all routing rules",
    tags=["Router"],
)
def list_routes():
    """Return all current routing rules."""
    rules = _load_rules()
    return RoutesListResponse(
        rules=[RoutingRule(**r) for r in rules if "category" in r and "severity" in r]
    )


@app.get("/health", tags=["Meta"])
def health():
    return {"status": "ok", "service": "router"}
