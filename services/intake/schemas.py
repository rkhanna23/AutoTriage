from datetime import datetime
from typing import List, Literal, Optional
from pydantic import BaseModel, Field, field_validator


TicketStatus = Literal["open", "in_progress", "resolved", "closed"]
ClassificationStatus = Literal["pending", "classified", "failed"]
RoutingStatus = Literal["pending", "routed", "failed"]


class ClassificationResult(BaseModel):
    """Classification result when available (CP3-RK-01)."""
    category: Optional[str] = None
    severity: Optional[str] = None
    confidence: Optional[float] = None
    model_version: Optional[str] = None
    prompt_version: Optional[str] = None
    needs_review: bool = False
    classification_status: str = "pending"

    model_config = {"protected_namespaces": ()}


class RoutingResult(BaseModel):
    """Routing result when available (CP3-RK-03)."""
    assigned_team: Optional[str] = None
    assigned_queue: Optional[str] = None
    routing_status: str = "pending"


class LatencyMetrics(BaseModel):
    """Per-stage timing in milliseconds (CP3-RK-04)."""
    intake_ms: Optional[int] = None
    classify_ms: Optional[int] = None
    route_ms: Optional[int] = None
    total_ms: Optional[int] = None


class TicketCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=255, description="Short title of the ticket")
    description: str = Field(..., min_length=1, description="Full description of the issue")
    source: str = Field(..., min_length=1, max_length=100, description="Origin system or team (e.g. 'web', 'mobile', 'ops')")
    timestamp: datetime = Field(..., description="ISO-8601 datetime when the ticket was originally created")

    @field_validator("title", "description", "source", mode="before")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        if isinstance(v, str):
            v = v.strip()
            if not v:
                raise ValueError("Field must not be blank")
        return v


class TicketResponse(BaseModel):
    ticket_id: str
    title: str
    description: str
    source: str
    timestamp: datetime
    status: TicketStatus
    created_at: datetime
    acknowledged: bool = True

    # CP3-RK-01: Classification when available
    classification: Optional[ClassificationResult] = None

    # CP3-RK-03: Routing when available
    routing: Optional[RoutingResult] = None

    # CP3-RK-04: Latency metrics
    latency: Optional[LatencyMetrics] = None

    model_config = {"from_attributes": True}


class TicketCreateResponse(BaseModel):
    """CP3-RK-01: Includes classification and routing when available."""
    ticket_id: str
    acknowledged: bool = True
    message: str = "Ticket received and stored successfully"
    classification: Optional[ClassificationResult] = None
    routing: Optional[RoutingResult] = None
    latency: Optional[LatencyMetrics] = None


class PaginationMeta(BaseModel):
    total: int
    limit: int
    offset: int
    has_more: bool


class TicketsListResponse(BaseModel):
    tickets: List[TicketResponse]
    pagination: PaginationMeta
