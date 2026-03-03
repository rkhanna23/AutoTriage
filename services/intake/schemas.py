from datetime import datetime
from typing import List, Literal, Optional
from pydantic import BaseModel, Field, field_validator


TicketStatus = Literal["open", "in_progress", "resolved", "closed"]


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

    model_config = {"from_attributes": True}


class TicketCreateResponse(BaseModel):
    ticket_id: str
    acknowledged: bool = True
    message: str = "Ticket received and stored successfully"


class PaginationMeta(BaseModel):
    total: int
    limit: int
    offset: int
    has_more: bool


class TicketsListResponse(BaseModel):
    tickets: List[TicketResponse]
    pagination: PaginationMeta
