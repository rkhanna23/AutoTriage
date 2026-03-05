"""
AutoTriage — Zero-Shot Ticket Classifier
CP2-SA-02/03: category + severity classification with structured schema + prompt versioning.

Active prompt template: v1.0 (services/classifier/prompts/v1.0.txt)
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Literal

import httpx
from pydantic import BaseModel, Field

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
PROMPT_VERSION = "v1.0"
CONFIDENCE_THRESHOLD = 0.40

VALID_CATEGORIES = frozenset(
    ["Auth", "Billing", "Outage", "Performance", "Security", "Feature Request"]
)
VALID_SEVERITIES = frozenset(["P0", "P1", "P2", "P3"])

_PROMPTS_DIR = Path(__file__).parent / "prompts"

Category = Literal["Auth", "Billing", "Outage", "Performance", "Security", "Feature Request"]
Severity = Literal["P0", "P1", "P2", "P3"]


class ClassificationResult(BaseModel):
    ticket_id: str
    category: Category
    severity: Severity
    confidence: float = Field(..., ge=0.0, le=1.0)
    model_version: str
    prompt_version: str


class LowConfidenceResult(BaseModel):
    ticket_id: str
    category: Literal["Unknown"]
    severity: Literal["Unknown"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    model_version: str
    prompt_version: str


def _load_prompt_template(version: str) -> str:
    path = _PROMPTS_DIR / f"{version}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    return path.read_text(encoding="utf-8")


def _render_prompt(template: str, ticket_id: str, title: str, description: str) -> str:
    return (
        template.replace("{{ticket_id}}", ticket_id)
        .replace("{{title}}", title)
        .replace("{{description}}", description)
    )


class TicketClassifier:
    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        model: str = OLLAMA_MODEL,
        prompt_version: str = PROMPT_VERSION,
        timeout: float = 60.0,
    ):
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._prompt_version = prompt_version
        self._timeout = timeout
        self._template = _load_prompt_template(prompt_version)

    def classify(
        self,
        ticket_id: str,
        title: str,
        description: str,
    ) -> ClassificationResult | LowConfidenceResult:
        prompt = _render_prompt(
            self._template,
            ticket_id=ticket_id.strip(),
            title=title.strip(),
            description=description.strip(),
        )

        payload = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.0, "num_predict": 220},
        }

        try:
            response = httpx.post(
                f"{self._base_url}/api/generate",
                json=payload,
                timeout=self._timeout,
            )
            response.raise_for_status()
            raw = response.json().get("response", "").strip()
            return self._parse(ticket_id=ticket_id, raw=raw)
        except httpx.HTTPError:
            # Graceful low-confidence output when model is unavailable.
            return LowConfidenceResult(
                ticket_id=ticket_id,
                category="Unknown",
                severity="Unknown",
                confidence=0.0,
                model_version=self._model,
                prompt_version=self._prompt_version,
            )

    def _parse(self, ticket_id: str, raw: str) -> ClassificationResult | LowConfidenceResult:
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
        raw = re.sub(r"\s*```$", "", raw, flags=re.MULTILINE).strip()

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return LowConfidenceResult(
                ticket_id=ticket_id,
                category="Unknown",
                severity="Unknown",
                confidence=0.0,
                model_version=self._model,
                prompt_version=self._prompt_version,
            )

        out_ticket_id = str(data.get("ticket_id", "")).strip() or ticket_id
        category = str(data.get("category", "")).strip()
        severity = str(data.get("severity", "")).strip()

        try:
            confidence = float(data.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0

        if out_ticket_id != ticket_id:
            return LowConfidenceResult(
                ticket_id=ticket_id,
                category="Unknown",
                severity="Unknown",
                confidence=max(0.0, min(confidence, 1.0)),
                model_version=self._model,
                prompt_version=self._prompt_version,
            )

        if (
            category not in VALID_CATEGORIES
            or severity not in VALID_SEVERITIES
            or confidence < CONFIDENCE_THRESHOLD
        ):
            return LowConfidenceResult(
                ticket_id=ticket_id,
                category="Unknown",
                severity="Unknown",
                confidence=max(0.0, min(confidence, 1.0)),
                model_version=self._model,
                prompt_version=self._prompt_version,
            )

        return ClassificationResult(
            ticket_id=ticket_id,
            category=category,  # type: ignore[arg-type]
            severity=severity,  # type: ignore[arg-type]
            confidence=max(0.0, min(confidence, 1.0)),
            model_version=self._model,
            prompt_version=self._prompt_version,
        )
