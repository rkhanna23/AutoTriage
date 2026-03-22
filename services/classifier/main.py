"""
AutoTriage — Classifier Service HTTP API
CP2-SA-02/03: category + severity + strict structured schema.
"""

from fastapi import FastAPI, HTTPException
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
    prompt_version: str | None = Field(
        default=None,
        description="Optional prompt version for A/B testing (for example: v1.0, v2.0, v2.1).",
    )


class ClassifyResponse(BaseModel):
    ticket_id: str
    category: str
    severity: str
    confidence: float
    model_version: str
    prompt_version: str
    needs_review: bool

    model_config = {"protected_namespaces": ()}


@app.post(
    "/classify",
    response_model=ClassifyResponse,
    summary="Classify a ticket into category + severity",
    tags=["Classifier"],
)
def classify_ticket(req: ClassifyRequest):
    try:
        result = _get_classifier().classify(
            ticket_id=req.ticket_id,
            title=req.title,
            description=req.description,
            prompt_version=req.prompt_version,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return ClassifyResponse(
        ticket_id=result.ticket_id,
        category=result.category,
        severity=result.severity,
        confidence=result.confidence,
        model_version=result.model_version,
        prompt_version=result.prompt_version,
        needs_review=result.needs_review,
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
