import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, Float, Integer, String, DateTime, Enum as SAEnum
from sqlalchemy.dialects.sqlite import TEXT
from .database import Base


def _new_uuid() -> str:
    return str(uuid.uuid4())


class Ticket(Base):
    __tablename__ = "tickets"

    id = Column(String(36), primary_key=True, default=_new_uuid, index=True)
    title = Column(String(255), nullable=False)
    description = Column(TEXT, nullable=False)
    source = Column(String(100), nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    status = Column(
        SAEnum("open", "in_progress", "resolved", "closed", name="ticket_status"),
        nullable=False,
        default="open",
    )
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    # CP3-RK-01: Classification fields
    category = Column(String(50), nullable=True)
    severity = Column(String(10), nullable=True)
    confidence = Column(Float, nullable=True)
    model_version = Column(String(100), nullable=True)
    prompt_version = Column(String(50), nullable=True)
    classification_status = Column(
        String(20), nullable=False, default="pending"
    )  # pending | classified | failed

    # CP3-RK-03: Routing fields
    assigned_team = Column(String(100), nullable=True)
    assigned_queue = Column(String(100), nullable=True)
    routing_status = Column(
        String(20), nullable=False, default="pending"
    )  # pending | routed | failed

    # CP3-RK-04: Latency instrumentation
    intake_ms = Column(Integer, nullable=True)
    classify_ms = Column(Integer, nullable=True)
    route_ms = Column(Integer, nullable=True)
    total_ms = Column(Integer, nullable=True)

    @property
    def ticket_id(self) -> str:
        return self.id

    # CP3: Properties for TicketResponse serialization (classification, routing, latency)
    @property
    def classification(self):
        from .schemas import ClassificationResult
        return ClassificationResult(
            category=self.category,
            severity=self.severity,
            confidence=self.confidence,
            model_version=self.model_version,
            prompt_version=self.prompt_version,
            classification_status=self.classification_status or "pending",
        )

    @property
    def routing(self):
        from .schemas import RoutingResult
        return RoutingResult(
            assigned_team=self.assigned_team,
            assigned_queue=self.assigned_queue,
            routing_status=self.routing_status or "pending",
        )

    @property
    def latency(self):
        from .schemas import LatencyMetrics
        if self.intake_ms is None and self.classify_ms is None and self.route_ms is None and self.total_ms is None:
            return None
        return LatencyMetrics(
            intake_ms=self.intake_ms,
            classify_ms=self.classify_ms,
            route_ms=self.route_ms,
            total_ms=self.total_ms,
        )
