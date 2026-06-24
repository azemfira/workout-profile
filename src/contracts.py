"""
contracts.py — the input/output contract between the survey and the plan builder.

This module is the authoritative, machine-checkable definition of:
  * SurveyAnswers — the INPUT shape
  * UserProfile   — the OUTPUT shape (what downstream consumes)

Everything is plain stdlib (dataclasses + enums) to keep dependencies minimal.
`UserProfile.to_dict()` produces the JSON the plan builder receives.

----------------------------------------------------------------------
INPUT — survey_answers
----------------------------------------------------------------------
A JSON array of answer blocks, exactly as produced by the onboarding form:

    [
      {"question": "<display text>", "answers": ["<label>", ...]},
      ...
    ]

  * question : the human label of the question (matched to a question id
               via config/questions.yaml; matching is whitespace-insensitive).
  * answers  : list of selected option labels. For `single` questions this is
               expected to hold one item; for `text` questions one free string.

----------------------------------------------------------------------
OUTPUT — user_profile  (see UserProfile.to_dict for the exact JSON)
----------------------------------------------------------------------
  schema_version : str
  goal           : { code, focus }
  intensity      : { level, score, drivers{experience,frequency,session_minutes} }
  environment    : { locations[], equipment_level }
  constraints    : { limitations[], contraindicated_exercise_types[] }
  flags          : { <flag_name>: bool, ... }
  free_text_signal : { provided, summary, preferences{likes[],dislikes[]},
                       schedule_notes, mentioned_constraints[],
                       mentions_potential_contraindication, source, raw_text }
  meta           : { unmatched_questions[], warnings[] }
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = "1.0"


# ----------------------------- Enums (closed value sets) -----------------------------
# Mirrors the codes in config/*.yaml. Kept here so the contract is self-describing
# and so typos in code are caught early. (Config remains the source of truth for
# *which* labels map to these; the enums just pin the allowed output vocabulary.)

class GoalCode(str, Enum):
    LOSE_WEIGHT = "lose_weight"
    BUILD_MUSCLE = "build_muscle"
    ENDURANCE = "endurance"
    MAINTAIN = "maintain"
    INJURY_RECOVERY = "injury_recovery"


class IntensityLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


class EquipmentLevel(str, Enum):
    NONE = "none"
    MINIMAL = "minimal"
    FULL = "full"


class FreeTextSource(str, Enum):
    EMPTY = "empty"      # no free text was provided -> LLM not called
    LLM = "llm"          # produced by a live LLM call
    STUB = "stub"        # produced by the offline keyword stub
    ERROR = "error"      # LLM call failed -> safe default returned


# ----------------------------- Output sub-objects -----------------------------

@dataclass
class Goal:
    code: str
    focus: str


@dataclass
class Intensity:
    level: str
    score: int
    drivers: Dict[str, Optional[str]]  # {experience, frequency, session_minutes}


@dataclass
class Environment:
    locations: List[str]
    equipment_level: str


@dataclass
class Constraints:
    limitations: List[str]
    contraindicated_exercise_types: List[str]


@dataclass
class Preferences:
    likes: List[str] = field(default_factory=list)
    dislikes: List[str] = field(default_factory=list)


@dataclass
class FreeTextSignal:
    """Interpreted signal from the single free-text question.

    This is the ONLY part of the profile influenced by the LLM. Its authority
    is deliberately bounded: it never writes contraindications directly — it can
    only *raise a flag* (`mentions_potential_contraindication`) that the
    deterministic layer turns into `needs_human_review`.
    """
    provided: bool
    source: str = FreeTextSource.EMPTY.value
    summary: Optional[str] = None
    preferences: Preferences = field(default_factory=Preferences)
    schedule_notes: Optional[str] = None
    mentioned_constraints: List[str] = field(default_factory=list)
    mentions_potential_contraindication: bool = False
    raw_text: Optional[str] = None

    @staticmethod
    def empty() -> "FreeTextSignal":
        return FreeTextSignal(provided=False, source=FreeTextSource.EMPTY.value)


@dataclass
class Meta:
    unmatched_questions: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


# ----------------------------- Top-level profile -----------------------------

@dataclass
class UserProfile:
    goal: Goal
    intensity: Intensity
    environment: Environment
    constraints: Constraints
    flags: Dict[str, bool]
    free_text_signal: FreeTextSignal
    meta: Meta
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to the exact JSON dict the plan builder consumes."""
        d = asdict(self)
        # asdict already recurses through nested dataclasses; just order keys nicely.
        return {
            "schema_version": d["schema_version"],
            "goal": d["goal"],
            "intensity": d["intensity"],
            "environment": d["environment"],
            "constraints": d["constraints"],
            "flags": d["flags"],
            "free_text_signal": d["free_text_signal"],
            "meta": d["meta"],
        }
