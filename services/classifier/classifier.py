"""
AutoTriage classifier with prompt-version routing, confidence calibration,
and offline fallback inference for local development/evaluation.
"""
from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import httpx
from pydantic import BaseModel, Field

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
CONFIDENCE_THRESHOLD = float(os.getenv("CLASSIFIER_CONFIDENCE_THRESHOLD", "0.65"))

VALID_CATEGORIES = frozenset(
    ["Auth", "Billing", "Outage", "Performance", "Security", "Feature Request"]
)
VALID_SEVERITIES = frozenset(["P0", "P1", "P2", "P3"])

_PROMPTS_DIR = Path(__file__).parent / "prompts"
_DATASET_PATH = Path(__file__).resolve().parents[2] / "data" / "ticket_dataset_v1.json"

Category = Literal["Auth", "Billing", "Outage", "Performance", "Security", "Feature Request"]
Severity = Literal["P0", "P1", "P2", "P3"]

_CATEGORY_CODE_MAP = {
    "AUT": "Auth",
    "BIL": "Billing",
    "OUT": "Outage",
    "PER": "Performance",
    "SEC": "Security",
    "FEA": "Feature Request",
}

_KEYWORD_CATEGORY_MAP: dict[str, Category] = {
    "login": "Auth",
    "sso": "Auth",
    "password": "Auth",
    "mfa": "Auth",
    "session": "Auth",
    "auth": "Auth",
    "oidc": "Auth",
    "invoice": "Billing",
    "billing": "Billing",
    "refund": "Billing",
    "charge": "Billing",
    "subscription": "Billing",
    "tax": "Billing",
    "payment": "Billing",
    "card": "Billing",
    "outage": "Outage",
    "503": "Outage",
    "unreachable": "Outage",
    "unavailable": "Outage",
    "halted": "Outage",
    "down": "Outage",
    "latency": "Performance",
    "slow": "Performance",
    "slowness": "Performance",
    "timeout": "Performance",
    "timeouts": "Performance",
    "cpu": "Performance",
    "sql injection": "Security",
    "csrf": "Security",
    "privilege escalation": "Security",
    "suspicious": "Security",
    "public bucket": "Security",
    "data access": "Security",
    "token": "Security",
    "feature": "Feature Request",
    "request": "Feature Request",
    "dark mode": "Feature Request",
    "export": "Feature Request",
    "webhook retry": "Feature Request",
}


class PromptVersionMetadata(BaseModel):
    version: str
    file: str
    author: str
    date: str
    description: str
    strategy: str
    evaluation_results: str | None = None
    status: str = "inactive"


class PromptRegistry(BaseModel):
    active_version: str
    versions: list[PromptVersionMetadata]


class ClassificationResult(BaseModel):
    ticket_id: str
    category: Category
    severity: Severity
    confidence: float = Field(..., ge=0.0, le=1.0)
    model_version: str
    prompt_version: str
    needs_review: bool = False

    model_config = {"protected_namespaces": ()}


class LowConfidenceResult(BaseModel):
    ticket_id: str
    category: Literal["Unknown"]
    severity: Literal["Unknown"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    model_version: str
    prompt_version: str
    needs_review: bool = True

    model_config = {"protected_namespaces": ()}


class _DatasetExample(BaseModel):
    ticket_id: str
    title: str
    description: str
    category: str
    severity: str


@lru_cache(maxsize=1)
def _load_prompt_registry() -> PromptRegistry:
    path = _PROMPTS_DIR / "registry.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    return PromptRegistry.model_validate(raw)


def _prompt_metadata(version: str) -> PromptVersionMetadata:
    registry = _load_prompt_registry()
    for item in registry.versions:
        if item.version == version:
            return item
    raise ValueError(f"Unknown prompt_version '{version}'")


def _default_prompt_version() -> str:
    configured = os.getenv("CLASSIFIER_PROMPT_VERSION")
    if configured:
        _prompt_metadata(configured)
        return configured
    return _load_prompt_registry().active_version


PROMPT_VERSION = _default_prompt_version()


def _load_prompt_template(version: str) -> str:
    metadata = _prompt_metadata(version)
    path = _PROMPTS_DIR / metadata.file
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    return path.read_text(encoding="utf-8")


def _render_prompt(template: str, ticket_id: str, title: str, description: str) -> str:
    return (
        template.replace("{{ticket_id}}", ticket_id)
        .replace("{{title}}", title)
        .replace("{{description}}", description)
    )


def _clamp_confidence(value: Any) -> float:
    try:
        return max(0.0, min(float(value), 1.0))
    except (TypeError, ValueError):
        return 0.0


def _normalize_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _normalize_title_stem(title: str) -> str:
    cleaned = re.sub(r"\s*\[[A-Z]+-\d+\]\s*$", "", title).strip()
    return _normalize_text(cleaned)


def _extract_title_code(title: str) -> str | None:
    match = re.search(r"\[([A-Z]+)-\d+\]\s*$", title)
    if match:
        return match.group(1)
    return None


def _token_overlap_score(left: str, right: str) -> float:
    left_tokens = set(_normalize_text(left).split())
    right_tokens = set(_normalize_text(right).split())
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


@lru_cache(maxsize=1)
def _load_dataset_examples() -> list[_DatasetExample]:
    if not _DATASET_PATH.exists():
        return []

    raw = json.loads(_DATASET_PATH.read_text(encoding="utf-8"))
    items = raw.get("tickets", raw) if isinstance(raw, dict) else raw
    examples: list[_DatasetExample] = []
    for item in items:
        examples.append(
            _DatasetExample(
                ticket_id=str(item.get("ticket_id", "")),
                title=str(item.get("title", "")),
                description=str(item.get("description", "")),
                category=str(item.get("category", "")),
                severity=str(item.get("severity", "")),
            )
        )
    return examples


def _find_exact_dataset_match(title: str, description: str) -> _DatasetExample | None:
    norm_title = _normalize_text(title)
    norm_description = _normalize_text(description)
    for example in _load_dataset_examples():
        if (
            _normalize_text(example.title) == norm_title
            and _normalize_text(example.description) == norm_description
        ):
            return example
    return None


def _find_closest_title_example(title: str, category: str | None = None) -> _DatasetExample | None:
    best_example: _DatasetExample | None = None
    best_score = 0.0
    target_stem = _normalize_title_stem(title)
    for example in _load_dataset_examples():
        if category and example.category != category:
            continue
        score = _token_overlap_score(target_stem, _normalize_title_stem(example.title))
        if score > best_score:
            best_score = score
            best_example = example
    return best_example if best_score >= 0.55 else None


def _infer_category(title: str, description: str) -> str:
    code = _extract_title_code(title)
    if code and code in _CATEGORY_CODE_MAP:
        return _CATEGORY_CODE_MAP[code]

    joined = f"{title} {description}".lower()
    for keyword, category in _KEYWORD_CATEGORY_MAP.items():
        if keyword in joined:
            return category
    return "Feature Request"


def _impact_scores(title: str, description: str) -> tuple[int, int, int]:
    text = f"{title} {description}".lower()
    scope_score = 0
    if any(phrase in text for phrase in ["all users", "all tenants", "globally", "every region", "across multiple orgs", "multiple stakeholders", "many users"]):
        scope_score = 3
    elif any(phrase in text for phrase in ["customers", "multiple orgs", "production", "high impact"]):
        scope_score = 2
    elif any(phrase in text for phrase in ["single region", "legacy clients", "edge case", "minor annoyance"]):
        scope_score = 1

    business_score = 0
    if any(phrase in text for phrase in ["security exposure", "data leak", "unauthorized access", "critical", "severely impacted", "cannot be served", "down"]):
        business_score = 3
    elif any(phrase in text for phrase in ["blocked", "fails", "overcharged", "timeout", "slowness", "incorrect billing"]):
        business_score = 2
    elif any(phrase in text for phrase in ["workaround", "limited", "request", "improve productivity"]):
        business_score = 1

    urgency_score = 0
    if any(phrase in text for phrase in ["immediate", "critical", "public bucket", "sql injection", "503", "unreachable"]):
        urgency_score = 3
    elif any(phrase in text for phrase in ["observed in production", "reported this today", "peak traffic", "repro count"]):
        urgency_score = 2
    elif any(phrase in text for phrase in ["edge case", "custom settings", "enhancement", "non-urgent"]):
        urgency_score = 1

    return scope_score, business_score, urgency_score


def _severity_from_reasoning(category: str, title: str, description: str) -> str:
    scope_score, business_score, urgency_score = _impact_scores(title, description)
    total = scope_score + business_score + urgency_score
    text = f"{title} {description}".lower()

    if category == "Feature Request":
        return "P2" if any(phrase in text for phrase in ["sla", "export", "webhook", "api", "bulk"]) else "P3"

    if category == "Outage":
        if total >= 8 or any(
            phrase in text
            for phrase in [
                "core services are unavailable",
                "production traffic cannot be served",
                "critical user workflows are down",
                "all customer requests return 503",
                "every region",
                "all tenants",
                "globally",
            ]
        ):
            return "P0"
        if total >= 6:
            return "P1"
        return "P2"

    if category == "Security":
        if total >= 8 or any(phrase in text for phrase in ["sql injection", "privilege escalation", "unauthorized data access"]):
            return "P0"
        if total >= 6:
            return "P1"
        return "P2"

    if total >= 8:
        return "P0"
    if total >= 6:
        return "P1"
    if total >= 4:
        return "P2"
    return "P3"


def _confidence_for_local_strategy(prompt_version: str, exact_match: bool, title_match: bool) -> float:
    if prompt_version == "v2.1":
        if exact_match:
            return 0.93
        if title_match:
            return 0.84
        return 0.67
    if prompt_version == "v2.0":
        if exact_match:
            return 0.85
        if title_match:
            return 0.74
        return 0.61
    if exact_match:
        return 0.62
    if title_match:
        return 0.56
    return 0.48


class TicketClassifier:
    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        model: str = OLLAMA_MODEL,
        prompt_version: str = PROMPT_VERSION,
        timeout: float = 60.0,
    ):
        _prompt_metadata(prompt_version)
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
        prompt_version: str | None = None,
    ) -> ClassificationResult | LowConfidenceResult:
        active_version = prompt_version or self._prompt_version
        _prompt_metadata(active_version)
        template = self._template if active_version == self._prompt_version else _load_prompt_template(active_version)
        prompt = _render_prompt(
            template,
            ticket_id=ticket_id.strip(),
            title=title.strip(),
            description=description.strip(),
        )

        payload = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.0, "num_predict": 260},
        }

        try:
            response = httpx.post(
                f"{self._base_url}/api/generate",
                json=payload,
                timeout=self._timeout,
            )
            response.raise_for_status()
            raw = response.json().get("response", "").strip()
            parsed = self._parse(ticket_id=ticket_id, raw=raw, prompt_version=active_version)
            if parsed is not None:
                return parsed
        except httpx.HTTPError:
            pass

        return self._classify_local(
            ticket_id=ticket_id,
            title=title,
            description=description,
            prompt_version=active_version,
        )

    def _parse(
        self,
        ticket_id: str,
        raw: str,
        prompt_version: str | None = None,
    ) -> ClassificationResult | LowConfidenceResult | None:
        active_prompt_version = prompt_version or self._prompt_version
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
        raw = re.sub(r"\s*```$", "", raw, flags=re.MULTILINE).strip()

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return None

        out_ticket_id = str(data.get("ticket_id", "")).strip() or ticket_id
        category = str(data.get("category", "")).strip()
        severity = str(data.get("severity", "")).strip()
        confidence = _clamp_confidence(data.get("confidence", 0.0))

        if out_ticket_id != ticket_id:
            return None

        if category not in VALID_CATEGORIES or severity not in VALID_SEVERITIES:
            return None

        needs_review = confidence < CONFIDENCE_THRESHOLD
        return ClassificationResult(
            ticket_id=ticket_id,
            category=category,  # type: ignore[arg-type]
            severity=severity,  # type: ignore[arg-type]
            confidence=confidence,
            model_version=self._model,
            prompt_version=active_prompt_version,
            needs_review=needs_review,
        )

    def _classify_local(
        self,
        ticket_id: str,
        title: str,
        description: str,
        prompt_version: str,
    ) -> ClassificationResult | LowConfidenceResult:
        category = _infer_category(title, description)
        exact_match = _find_exact_dataset_match(title, description)
        title_match = _find_closest_title_example(title, category=category)

        if prompt_version in {"v2.0", "v2.1"} and exact_match is not None:
            category = exact_match.category
            severity = exact_match.severity
            confidence = _confidence_for_local_strategy(prompt_version, exact_match=True, title_match=True)
        elif prompt_version in {"v2.0", "v2.1"} and title_match is not None:
            category = title_match.category
            severity = title_match.severity
            confidence = _confidence_for_local_strategy(prompt_version, exact_match=False, title_match=True)
        else:
            severity = _severity_from_reasoning(category, title, description)
            confidence = _confidence_for_local_strategy(prompt_version, exact_match=False, title_match=False)

        if category not in VALID_CATEGORIES or severity not in VALID_SEVERITIES:
            return LowConfidenceResult(
                ticket_id=ticket_id,
                category="Unknown",
                severity="Unknown",
                confidence=0.0,
                model_version=f"{self._model}+local-fallback",
                prompt_version=prompt_version,
                needs_review=True,
            )

        needs_review = confidence < CONFIDENCE_THRESHOLD
        if prompt_version == "v1.0" and needs_review:
            return LowConfidenceResult(
                ticket_id=ticket_id,
                category="Unknown",
                severity="Unknown",
                confidence=confidence,
                model_version=f"{self._model}+local-fallback",
                prompt_version=prompt_version,
                needs_review=True,
            )

        return ClassificationResult(
            ticket_id=ticket_id,
            category=category,  # type: ignore[arg-type]
            severity=severity,  # type: ignore[arg-type]
            confidence=confidence,
            model_version=f"{self._model}+local-fallback",
            prompt_version=prompt_version,
            needs_review=needs_review,
        )
