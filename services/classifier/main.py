"""
AutoTriage — Classifier Service HTTP API
CP2-SA-02/03: category + severity + strict structured schema.
"""

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .classifier import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    PROMPT_VERSION,
    TicketClassifier,
)

app = FastAPI(
    title="AutoTriage Classifier Service",
    description="Zero-shot ticket classifier returning category + severity as structured JSON.",
    version="1.0.0",
)

_classifier: TicketClassifier | None = None


def _get_classifier() -> TicketClassifier:
    global _classifier
    if _classifier is None:
        _classifier = TicketClassifier()
    return _classifier


class ClassifyRequest(BaseModel):
    ticket_id: str = Field(..., min_length=1, description="UUID or unique ID of the ticket")
    title: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)


class ClassifyResponse(BaseModel):
    ticket_id: str
    category: str
    severity: str
    confidence: float
    model_version: str
    prompt_version: str


@app.post(
    "/classify",
    response_model=ClassifyResponse,
    summary="Classify a ticket into category + severity",
    tags=["Classifier"],
)
def classify_ticket(req: ClassifyRequest):
    result = _get_classifier().classify(
        ticket_id=req.ticket_id,
        title=req.title,
        description=req.description,
    )

    return ClassifyResponse(
        ticket_id=result.ticket_id,
        category=result.category,
        severity=result.severity,
        confidence=result.confidence,
        model_version=result.model_version,
        prompt_version=result.prompt_version,
    )


@app.get("/health", tags=["Meta"])
def health():
    return {
        "status": "ok",
        "service": "classifier",
        "ollama_url": OLLAMA_BASE_URL,
        "model": OLLAMA_MODEL,
        "prompt_version": PROMPT_VERSION,
    }
