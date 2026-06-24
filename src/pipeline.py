"""
pipeline.py — orchestration: survey_answers -> user_profile.

Flow (see docs/design.md for the diagram):

    survey_answers
        -> normalize            (labels -> codes)               [deterministic]
        -> classify_*           (goal, intensity, contras, ...) [deterministic]
        -> interpret free text  (one bounded LLM call)          [LLM]
        -> assemble             (merge + cross-cutting flags)    [deterministic]
        -> user_profile

`build_profile` is the single public entry point.
"""

from __future__ import annotations

from typing import List, Optional

from .classify import (
    classify_goal,
    classify_intensity,
    derive_contraindications,
    derive_equipment_level,
    evaluate_flags,
)
from .config_loader import Config, load_config
from .contracts import Environment, Meta, UserProfile
from .free_text import FreeTextInterpreter, default_interpreter
from .normalize import normalize


def build_profile(
    survey_answers: List[dict],
    interpreter: Optional[FreeTextInterpreter] = None,
    cfg: Optional[Config] = None,
) -> UserProfile:
    """Turn raw survey answers into a structured UserProfile.

    `interpreter` is injectable (tests pass a stub; production lets it default to
    a live LLM when ANTHROPIC_API_KEY is set, else the offline stub).
    """
    cfg = cfg or load_config()
    interpreter = interpreter or default_interpreter()

    # 1) Deterministic normalization.
    answers = normalize(survey_answers, cfg)

    # 2) Deterministic classification.
    goal = classify_goal(answers, cfg)
    intensity = classify_intensity(answers, cfg)
    constraints = derive_contraindications(answers, cfg)
    flags = evaluate_flags(answers, cfg)
    environment = Environment(
        locations=answers.codes("locations"),
        equipment_level=derive_equipment_level(answers, cfg),
    )

    # 3) The single LLM call (skipped internally if free text is empty).
    free_text_signal = interpreter.interpret(answers.free_text)

    # 4) Deterministic assembly + cross-cutting safety flag.
    flags["needs_human_review"] = _needs_human_review(flags, free_text_signal)

    return UserProfile(
        goal=goal,
        intensity=intensity,
        environment=environment,
        constraints=constraints,
        flags=flags,
        free_text_signal=free_text_signal,
        meta=Meta(
            unmatched_questions=answers.unmatched_questions,
            warnings=answers.warnings,
        ),
    )


def _needs_human_review(flags, free_text_signal) -> bool:
    """Cross-cutting safety flag, computed in CODE (not config) because it
    combines a deterministic signal with the bounded LLM signal:

      * medical clearance already recommended (heart/pressure or pregnancy), OR
      * the free-text note flagged a potential contraindication the checkbox
        question didn't capture.

    This is exactly where we convert the LLM's *advisory* flag into a
    deterministic, conservative action — the model never gets to silently add a
    contraindication itself.
    """
    return bool(
        flags.get("medical_clearance_recommended")
        or free_text_signal.mentions_potential_contraindication
    )
