"""
free_text.py — the ONE LLM call in the whole layer.

It interprets the single free-text question ("что ещё важно знать о пользователе?")
into a small, closed `FreeTextSignal`. Everything else in the pipeline is
deterministic.

Two interchangeable implementations behind one interface (`FreeTextInterpreter`):

  * LLMFreeTextInterpreter  — real Anthropic call. Uses FORCED TOOL USE so the
    model must return JSON matching our schema (the robust way to get structured
    output today). Falls back to a safe, flagged default if anything goes wrong.

  * StubFreeTextInterpreter — deterministic keyword matcher. No network, no key.
    Used by tests and by `examples/run_example.py` when ANTHROPIC_API_KEY is unset,
    so the end-to-end flow (including the LLM-flag -> needs_human_review path) is
    fully reproducible offline.

Bounded authority (key design choice): the model may DESCRIBE and may RAISE a
flag, but it never writes contraindications or profile fields directly. The
deterministic layer decides what to do with the flag (see pipeline.assemble).
"""

from __future__ import annotations

import json
import os
import re
from typing import Optional, Protocol

from .contracts import FreeTextSignal, FreeTextSource, Preferences

# Small, cheap model is the right tool for a single short extraction.
DEFAULT_MODEL = os.environ.get("WORKOUT_LLM_MODEL", "claude-haiku-4-5-20251001")

# The schema we force the model to fill. Kept tiny and well-described.
EXTRACTION_TOOL = {
    "name": "record_free_text_signal",
    "description": (
        "Record a structured interpretation of a fitness user's free-text note. "
        "Only extract what is actually stated; do not infer or invent. Leave lists "
        "empty and strings null when nothing relevant is present."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "description": "One short, neutral sentence summarizing the note. Empty string if the note is empty/irrelevant.",
            },
            "likes": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Activities/exercise types the user says they enjoy or prefer.",
            },
            "dislikes": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Activities/exercise types the user says they dislike or want to avoid.",
            },
            "schedule_notes": {
                "type": ["string", "null"],
                "description": "Any timing/availability constraint (e.g. 'mornings only', 'travels on weekends'). Null if none.",
            },
            "mentioned_constraints": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Health/physical/contextual constraints stated in the note (e.g. 'wrist pain', 'recent knee surgery', 'wedding in 3 months').",
            },
            "mentions_potential_contraindication": {
                "type": "boolean",
                "description": "True if the note mentions any pain, injury, illness, surgery, pregnancy, or medical condition that a coach should review for safety.",
            },
        },
        "required": [
            "summary",
            "likes",
            "dislikes",
            "schedule_notes",
            "mentioned_constraints",
            "mentions_potential_contraindication",
        ],
    },
}

SYSTEM_PROMPT = (
    "You interpret a single free-text note from a fitness onboarding form and "
    "extract a small structured signal for a workout-plan builder. The note may be "
    "in any language. Be conservative and literal: extract only what the user "
    "actually wrote. Never diagnose, never give advice, never add information that "
    "is not present. Always call the record_free_text_signal tool."
)


class FreeTextInterpreter(Protocol):
    def interpret(self, text: str) -> FreeTextSignal: ...


def _signal_from_payload(payload: dict, text: str, source: str) -> FreeTextSignal:
    """Map the raw extraction dict onto our FreeTextSignal contract."""
    return FreeTextSignal(
        provided=True,
        source=source,
        summary=(payload.get("summary") or None),
        preferences=Preferences(
            likes=list(payload.get("likes") or []),
            dislikes=list(payload.get("dislikes") or []),
        ),
        schedule_notes=(payload.get("schedule_notes") or None),
        mentioned_constraints=list(payload.get("mentioned_constraints") or []),
        mentions_potential_contraindication=bool(
            payload.get("mentions_potential_contraindication", False)
        ),
        raw_text=text,
    )


class LLMFreeTextInterpreter:
    """Live Anthropic interpreter using forced tool use for structured output."""

    def __init__(self, model: str = DEFAULT_MODEL, client: Optional[object] = None):
        self.model = model
        self._client = client  # injectable for testing

    def _get_client(self):
        if self._client is not None:
            return self._client
        import anthropic  # lazy import: only needed for live calls
        self._client = anthropic.Anthropic()
        return self._client

    def interpret(self, text: str) -> FreeTextSignal:
        if not text or not text.strip():
            return FreeTextSignal.empty()
        try:
            client = self._get_client()
            resp = client.messages.create(
                model=self.model,
                max_tokens=512,
                system=SYSTEM_PROMPT,
                tools=[EXTRACTION_TOOL],
                tool_choice={"type": "tool", "name": EXTRACTION_TOOL["name"]},
                messages=[{"role": "user", "content": text.strip()}],
            )
            payload = _extract_tool_input(resp)
            if payload is None:
                raise ValueError("Model did not return the expected tool call")
            return _signal_from_payload(payload, text, FreeTextSource.LLM.value)
        except Exception as exc:  # graceful degradation — never break the pipeline
            return FreeTextSignal(
                provided=True,
                source=FreeTextSource.ERROR.value,
                summary=None,
                raw_text=text,
                # Be conservative on failure: surface for human review.
                mentions_potential_contraindication=True,
                mentioned_constraints=[f"llm_error: {type(exc).__name__}"],
            )


def _extract_tool_input(resp) -> Optional[dict]:
    """Pull the tool_use input out of an Anthropic Messages response."""
    for block in getattr(resp, "content", []) or []:
        if getattr(block, "type", None) == "tool_use":
            return dict(block.input)
    return None


class StubFreeTextInterpreter:
    """Deterministic, offline keyword interpreter. Good enough to exercise the
    full pipeline in tests; obviously not a substitute for the real model."""

    _PAIN_WORDS = [
        "pain", "hurt", "injur", "surgery", "operation", "герни", "hernia",
        "боль", "трав", "операц", "беремен", "pregnan", "давлен", "сердц",
    ]
    _DISLIKE_PATTERNS = [
        (r"не люблю\s+(\w+)", 1),
        (r"hate\s+(\w+)", 1),
        (r"don'?t like\s+(\w+)", 1),
        (r"avoid\s+(\w+)", 1),
    ]
    _LIKE_PATTERNS = [
        (r"люблю\s+(\w+)", 1),
        (r"love\s+(\w+)", 1),
        (r"enjoy\s+(\w+)", 1),
    ]

    def interpret(self, text: str) -> FreeTextSignal:
        if not text or not text.strip():
            return FreeTextSignal.empty()
        low = text.lower()

        dislikes = _find_all(low, self._DISLIKE_PATTERNS)
        likes = [w for w in _find_all(low, self._LIKE_PATTERNS) if w not in dislikes]
        flagged = any(w in low for w in self._PAIN_WORDS)
        constraints = [w for w in self._PAIN_WORDS if w in low]

        return _signal_from_payload(
            {
                "summary": text.strip()[:140],
                "likes": likes,
                "dislikes": dislikes,
                "schedule_notes": None,
                "mentioned_constraints": constraints,
                "mentions_potential_contraindication": flagged,
            },
            text,
            FreeTextSource.STUB.value,
        )


def _find_all(text: str, patterns) -> list:
    out: list = []
    for pat, grp in patterns:
        for m in re.finditer(pat, text):
            token = m.group(grp)
            if token and token not in out:
                out.append(token)
    return out


def default_interpreter() -> FreeTextInterpreter:
    """Pick a live interpreter if a key is configured, else the offline stub."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return LLMFreeTextInterpreter()
    return StubFreeTextInterpreter()
