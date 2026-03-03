import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, DateTime, Enum as SAEnum
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

    @property
    def ticket_id(self) -> str:
        return self.id
