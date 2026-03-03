"""
AutoTriage — Zero-Shot Ticket Classifier
CP2-SA-01: LLM-based category classifier via Ollama (local, free, no API key).

Default model : llama3.1:8b
Swap model    : set OLLAMA_MODEL env var to any model pulled in Ollama
                e.g. deepseek-r1:7b, mistral, qwen2.5:7b, llama3.3:70b

Prompt template: v1.0  (services/classifier/prompts/v1.0.txt)
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Literal

import httpx
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
PROMPT_VERSION = "v1.0"
CONFIDENCE_THRESHOLD = 0.40

VALID_CATEGORIES = frozenset(
    ["Auth", "Billing", "Outage", "Performance", "Security", "Feature Request"]
)

_PROMPTS_DIR = Path(__file__).parent / "prompts"

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

Category = Literal["Auth", "Billing", "Outage", "Performance", "Security", "Feature Request"]


class ClassificationResult(BaseModel):
    category: Category
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str
    model_version: str
    prompt_version: str


class LowConfidenceResult(BaseModel):
    category: Literal["Unknown"]
    confidence: float
    reasoning: str
    model_version: str
    prompt_version: str
    error: str = "Low confidence or unrecognised category — manual review recommended"


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

def _load_prompt_template(version: str) -> str:
    path = _PROMPTS_DIR / f"{version}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    return path.read_text(encoding="utf-8")


def _render_prompt(template: str, title: str, description: str) -> str:
    return template.replace("{{title}}", title).replace("{{description}}", description)


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class TicketClassifier:
    """
    Zero-shot ticket classifier backed by a locally running Ollama instance.

    Completely free — requires Ollama to be running with the chosen model pulled.
    Compatible with: llama3.1:8b, deepseek-r1:7b, mistral, qwen2.5:7b,
                     llama3.3:70b, mixtral, and any other Ollama model.
    """

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

    def classify(self, title: str, description: str) -> ClassificationResult | LowConfidenceResult:
        """
        Classify a support ticket into one of six categories.

        Sends the rendered prompt to Ollama and parses the JSON response.
        Returns LowConfidenceResult if the model responds with low confidence
        or an unrecognised category.
        """
        prompt = _render_prompt(self._template, title=title.strip(), description=description.strip())

        payload = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "format": "json",   # tells Ollama to enforce JSON output mode
            "options": {
                "temperature": 0.0,     # deterministic — classification needs consistency
                "num_predict": 200,
            },
        }

        try:
            response = httpx.post(
                f"{self._base_url}/api/generate",
                json=payload,
                timeout=self._timeout,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            return LowConfidenceResult(
                category="Unknown",
                confidence=0.0,
                reasoning=f"Ollama request failed: {exc}",
                model_version=self._model,
                prompt_version=self._prompt_version,
                error=str(exc),
            )

        raw = response.json().get("response", "").strip()
        return self._parse(raw)

    def _parse(self, raw: str) -> ClassificationResult | LowConfidenceResult:
        # Strip any accidental markdown fences some models still emit
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
        raw = re.sub(r"\s*```$", "", raw, flags=re.MULTILINE).strip()

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return LowConfidenceResult(
                category="Unknown",
                confidence=0.0,
                reasoning="Model did not return valid JSON.",
                model_version=self._model,
                prompt_version=self._prompt_version,
            )

        category: str = data.get("category", "").strip()
        confidence: float = float(data.get("confidence", 0.0))
        reasoning: str = data.get("reasoning", "")

        if category not in VALID_CATEGORIES or confidence < CONFIDENCE_THRESHOLD:
            return LowConfidenceResult(
                category="Unknown",
                confidence=confidence,
                reasoning=reasoning,
                model_version=self._model,
                prompt_version=self._prompt_version,
            )

        return ClassificationResult(
            category=category,  # type: ignore[arg-type]
            confidence=confidence,
            reasoning=reasoning,
            model_version=self._model,
            prompt_version=self._prompt_version,
        )
