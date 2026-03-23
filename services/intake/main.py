"""
AutoTriage — Ticket Intake Service
Implements CP2-RK-01, CP2-RK-02, CP2-RK-03, CP3-RK-01, CP3-RK-03, CP3-RK-04
"""
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Query
from sqlalchemy.orm import Session

from .database import Base, engine, get_db
from .models import Ticket
from .pipeline import classify_ticket as pipeline_classify, route_ticket as pipeline_route
from .schemas import (
    ClassificationResult,
    LatencyMetrics,
    PaginationMeta,
    RoutingResult,
    TicketCreate,
    TicketCreateResponse,
    TicketResponse,
    TicketsListResponse,
)

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="AutoTriage Intake Service",
    description="REST API for ticket submission, storage, and retrieval.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


def _build_create_response(ticket: Ticket) -> TicketCreateResponse:
    """Build TicketCreateResponse with classification, routing, latency when available."""
    classification = None
    if ticket.classification_status in ("classified", "failed"):
        classification = ClassificationResult(
            category=ticket.category,
            severity=ticket.severity,
            confidence=ticket.confidence,
            model_version=ticket.model_version,
            prompt_version=ticket.prompt_version,
            needs_review=bool(ticket.needs_review),
            classification_status=ticket.classification_status or "pending",
        )
    routing = None
    if ticket.routing_status in ("routed", "failed"):
        routing = RoutingResult(
            assigned_team=ticket.assigned_team,
            assigned_queue=ticket.assigned_queue,
            routing_status=ticket.routing_status or "pending",
        )
    latency = None
    if ticket.total_ms is not None:
        latency = LatencyMetrics(
            intake_ms=ticket.intake_ms,
            classify_ms=ticket.classify_ms,
            route_ms=ticket.route_ms,
            total_ms=ticket.total_ms,
        )
    return TicketCreateResponse(
        ticket_id=ticket.id,
        classification=classification,
        routing=routing,
        latency=latency,
    )


# ---------------------------------------------------------------------------
# CP2-RK-01 / CP3-RK-01 — POST /tickets (with auto-classify and auto-route)
# ---------------------------------------------------------------------------

@app.post(
    "/tickets",
    response_model=TicketCreateResponse,
    status_code=201,
    summary="Submit a new support ticket",
    tags=["Tickets"],
)
def create_ticket(payload: TicketCreate, db: Session = Depends(get_db)):
    """
    Accept a new ticket, persist it, auto-classify via classifier service,
    then auto-route via router service. Returns ticket_id and classification/routing when available.
    """
    t0 = time.perf_counter()
    ticket = Ticket(
        id=str(uuid.uuid4()),
        title=payload.title,
        description=payload.description,
        source=payload.source,
        timestamp=payload.timestamp,
        status="open",
        created_at=datetime.now(timezone.utc),
    )
    db.add(ticket)
    db.commit()
    db.refresh(ticket)
    intake_ms = int(round((time.perf_counter() - t0) * 1000))

    # CP3-RK-01: Call classifier
    class_result, classify_ms, class_ok = pipeline_classify(
        ticket.id, ticket.title, ticket.description
    )
    ticket.classification_status = "classified" if class_ok else "pending"
    ticket.category = class_result.get("category")
    ticket.severity = class_result.get("severity")
    ticket.confidence = class_result.get("confidence")
    ticket.model_version = class_result.get("model_version") or ""
    ticket.prompt_version = class_result.get("prompt_version") or ""
    ticket.needs_review = bool(class_result.get("needs_review", False))
    ticket.classify_ms = classify_ms
    ticket.intake_ms = intake_ms

    # CP3-RK-03: Call router
    route_result, route_ms, route_ok = pipeline_route(
        ticket.id,
        None if ticket.needs_review else ticket.category,
        None if ticket.needs_review else ticket.severity,
    )
    ticket.routing_status = "routed" if route_ok else "failed"
    ticket.assigned_team = route_result.get("assigned_team")
    ticket.assigned_queue = route_result.get("assigned_queue")
    ticket.route_ms = route_ms
    ticket.total_ms = int(round((time.perf_counter() - t0) * 1000))

    db.commit()
    db.refresh(ticket)
    return _build_create_response(ticket)


# ---------------------------------------------------------------------------
# CP2-RK-03 — GET /tickets  (list with filters + pagination)
# ---------------------------------------------------------------------------

@app.get(
    "/tickets",
    response_model=TicketsListResponse,
    summary="List tickets with optional filters",
    tags=["Tickets"],
)
def list_tickets(
    source: Optional[str] = Query(None, description="Filter by source system"),
    date_from: Optional[datetime] = Query(None, description="Include tickets at or after this ISO-8601 datetime"),
    date_to: Optional[datetime] = Query(None, description="Include tickets at or before this ISO-8601 datetime"),
    status: Optional[str] = Query(None, description="Filter by status (open, in_progress, resolved, closed)"),
    limit: int = Query(20, ge=1, le=100, description="Max records to return"),
    offset: int = Query(0, ge=0, description="Number of records to skip"),
    db: Session = Depends(get_db),
):
    """
    Return a paginated list of tickets.

    Supports optional filtering by `source`, `status`, and a `date_from`/`date_to`
    range applied to the ticket's `timestamp` field.
    """
    query = db.query(Ticket)

    if source:
        query = query.filter(Ticket.source == source)
    if status:
        valid_statuses = {"open", "in_progress", "resolved", "closed"}
        if status not in valid_statuses:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid status '{status}'. Must be one of: {sorted(valid_statuses)}",
            )
        query = query.filter(Ticket.status == status)
    if date_from:
        query = query.filter(Ticket.timestamp >= date_from)
    if date_to:
        query = query.filter(Ticket.timestamp <= date_to)

    total = query.count()
    tickets = query.order_by(Ticket.created_at.desc()).offset(offset).limit(limit).all()

    return TicketsListResponse(
        tickets=[TicketResponse.model_validate(t) for t in tickets],
        pagination=PaginationMeta(
            total=total,
            limit=limit,
            offset=offset,
            has_more=(offset + limit) < total,
        ),
    )


# ---------------------------------------------------------------------------
# CP2-RK-03 — GET /tickets/{ticket_id}  (single ticket retrieval)
# ---------------------------------------------------------------------------

@app.get(
    "/tickets/{ticket_id}",
    response_model=TicketResponse,
    summary="Retrieve a single ticket by ID",
    tags=["Tickets"],
)
def get_ticket(ticket_id: str, db: Session = Depends(get_db)):
    """
    Fetch a single ticket by its UUID.

    Returns 404 if the ticket does not exist.
    """
    ticket = db.query(Ticket).filter(Ticket.id == ticket_id).first()
    if not ticket:
        raise HTTPException(status_code=404, detail=f"Ticket '{ticket_id}' not found")
    return TicketResponse.model_validate(ticket)


# ---------------------------------------------------------------------------
# CP3-RK-04 — GET /metrics/latency (p50, p95, p99 over last N tickets)
# ---------------------------------------------------------------------------

def _percentile(sorted_values: list[float], p: float) -> float | None:
    """Compute percentile (0-100) from sorted list. Returns None if empty."""
    if not sorted_values:
        return None
    k = (len(sorted_values) - 1) * (p / 100)
    f = int(k)
    c = f + 1 if f + 1 < len(sorted_values) else f
    return sorted_values[f] + (k - f) * (sorted_values[c] - sorted_values[f])


@app.get(
    "/metrics/latency",
    summary="Latency percentiles (p50, p95, p99) over recent tickets",
    tags=["Metrics"],
)
def get_latency_metrics(
    n: int = Query(100, ge=1, le=1000, description="Number of recent tickets to aggregate"),
    db: Session = Depends(get_db),
):
    """
    Return p50, p95, p99 latencies (in ms) for intake_ms, classify_ms, route_ms, total_ms
    aggregated over the last N tickets that have timing data.
    """
    tickets = (
        db.query(Ticket)
        .filter(Ticket.total_ms.isnot(None))
        .order_by(Ticket.created_at.desc())
        .limit(n)
        .all()
    )
    intake_vals = sorted([t.intake_ms for t in tickets if t.intake_ms is not None])
    classify_vals = sorted([t.classify_ms for t in tickets if t.classify_ms is not None])
    route_vals = sorted([t.route_ms for t in tickets if t.route_ms is not None])
    total_vals = sorted([t.total_ms for t in tickets if t.total_ms is not None])

    def _pct(vals: list[float]) -> dict[str, float | None]:
        return {
            "p50": _percentile(vals, 50),
            "p95": _percentile(vals, 95),
            "p99": _percentile(vals, 99),
        }

    return {
        "sample_size": len(tickets),
        "intake_ms": _pct(intake_vals),
        "classify_ms": _pct(classify_vals),
        "route_ms": _pct(route_vals),
        "total_ms": _pct(total_vals),
    }


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health", tags=["Meta"])
def health():
    return {"status": "ok", "service": "intake"}
