"""
config_loader.py — load and lightly validate the YAML config bundle.

The loader is intentionally thin: it reads the YAML files in `config/`, does a
few sanity checks (so a broken config fails loudly at startup rather than
silently mis-classifying a user), and hands back a `Config` object the rest of
the pipeline reads from. No business logic lives here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List

import yaml

CONFIG_DIR = os.environ.get(
    "WORKOUT_CONFIG_DIR",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "config"),
)


def _read(name: str) -> Dict[str, Any]:
    path = os.path.join(CONFIG_DIR, name)
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        raise ValueError(f"Config file {name} is empty")
    return data


@dataclass
class Config:
    questions: List[Dict[str, Any]]
    goals: Dict[str, Any]
    intensity: Dict[str, Any]
    contraindications: Dict[str, Any]
    flags: Dict[str, Any]

    # --- convenience accessors used across the pipeline ---

    def question_by_id(self, qid: str) -> Dict[str, Any]:
        for q in self.questions:
            if q["id"] == qid:
                return q
        raise KeyError(f"No question with id={qid!r}")

    def text_to_id(self) -> Dict[str, str]:
        """Map normalized question display text -> question id."""
        return {_norm(q["text"]): q["id"] for q in self.questions}

    def label_to_code(self, qid: str) -> Dict[str, str]:
        """Map normalized option label -> code for one question."""
        q = self.question_by_id(qid)
        return {_norm(opt["label"]): opt["code"] for opt in q.get("options", [])}

    def question_type(self, qid: str) -> str:
        return self.question_by_id(qid)["type"]


def _norm(text: str) -> str:
    """Whitespace-insensitive normalization for robust text matching."""
    return " ".join(str(text).split()).strip().lower()


def _validate(cfg: Config) -> None:
    """Fail fast on the most damaging kinds of config drift."""
    ids = [q["id"] for q in cfg.questions]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate question ids in questions.yaml")

    valid_types = {"single", "multi", "text"}
    for q in cfg.questions:
        if q["type"] not in valid_types:
            raise ValueError(f"Question {q['id']}: invalid type {q['type']!r}")

    # Every contraindication tag must be in the declared vocabulary.
    vocab = set(cfg.contraindications["exercise_types"])
    for code, tags in cfg.contraindications["map"].items():
        unknown = set(tags) - vocab
        if unknown:
            raise ValueError(
                f"contraindications.map[{code}] uses tags outside the vocabulary: {unknown}"
            )

    # Intensity point tables must reference real codes (best-effort check).
    for dim in ("experience", "frequency", "session_minutes"):
        if dim not in cfg.intensity["points"]:
            raise ValueError(f"intensity.yaml missing points for {dim!r}")


@lru_cache(maxsize=1)
def load_config() -> Config:
    cfg = Config(
        questions=_read("questions.yaml")["questions"],
        goals=_read("goals.yaml"),
        intensity=_read("intensity.yaml"),
        contraindications=_read("contraindications.yaml"),
        flags=_read("flags.yaml"),
    )
    _validate(cfg)
    return cfg
