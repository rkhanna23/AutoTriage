"""
AutoTriage — Ticket Intake Service
Implements CP2-RK-01, CP2-RK-02, CP2-RK-03
"""
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from .database import Base, engine, get_db
from .models import Ticket
from .schemas import (
    PaginationMeta,
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


# ---------------------------------------------------------------------------
# CP2-RK-01 — POST /tickets
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
    Accept a new ticket and persist it to the database.

    Returns the generated `ticket_id` and an acknowledgment flag.
    """
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
    return TicketCreateResponse(ticket_id=ticket.id)


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
# Health check
# ---------------------------------------------------------------------------

@app.get("/health", tags=["Meta"])
def health():
    return {"status": "ok", "service": "intake"}
