"""
AutoTriage — Classifier Service HTTP API (CP2-SA-01)
Backed by Ollama — free, local, no API key required.
"""
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .classifier import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    ClassificationResult,
    LowConfidenceResult,
    TicketClassifier,
)

app = FastAPI(
    title="AutoTriage Classifier Service",
    description=(
        "Zero-shot ticket category classifier powered by a local Ollama LLM. "
        "Free, no API key required. Model is configurable via OLLAMA_MODEL env var. "
        "(CP2-SA-01)"
    ),
    version="1.0.0",
)

_classifier: TicketClassifier | None = None


def _get_classifier() -> TicketClassifier:
    global _classifier
    if _classifier is None:
        _classifier = TicketClassifier()
    return _classifier


class ClassifyRequest(BaseModel):
    ticket_id: str = Field(..., description="UUID of the ticket being classified")
    title: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)


class ClassifyResponse(BaseModel):
    ticket_id: str
    category: str
    confidence: float
    reasoning: str
    model_version: str
    prompt_version: str
    low_confidence: bool = False


@app.post(
    "/classify",
    response_model=ClassifyResponse,
    summary="Classify a ticket into one of six categories",
    tags=["Classifier"],
)
def classify_ticket(req: ClassifyRequest):
    """
    Classify a ticket using the locally running Ollama LLM.

    The active model is controlled by the `OLLAMA_MODEL` environment variable
    (default: `llama3.1:8b`). Any model available in your Ollama instance works —
    e.g. `deepseek-r1:7b`, `mistral`, `qwen2.5:7b`, `llama3.3:70b`.
    """
    classifier = _get_classifier()
    result = classifier.classify(title=req.title, description=req.description)

    low = isinstance(result, LowConfidenceResult)
    return ClassifyResponse(
        ticket_id=req.ticket_id,
        category=result.category,
        confidence=result.confidence,
        reasoning=result.reasoning,
        model_version=result.model_version,
        prompt_version=result.prompt_version,
        low_confidence=low,
    )


@app.get("/health", tags=["Meta"])
def health():
    return {
        "status": "ok",
        "service": "classifier",
        "ollama_url": OLLAMA_BASE_URL,
        "model": OLLAMA_MODEL,
    }
