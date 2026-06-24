"""
normalize.py — turn raw `survey_answers` into clean, coded answers.

Output is a `NormalizedAnswers` object holding:
  * by_id    : {question_id: [code, ...]}   — normalized, code-level answers
  * free_text: str                          — the raw free-text answer ("" if none)
  * unmatched_questions / warnings          — surfaced into profile.meta

This step is 100% deterministic and config-driven. It does NOT make any
classification decisions — it only translates labels to codes and records
anomalies. All edge-case handling (unknown labels, multi-answers on a single
question, the "Нет" + real-limitation contradiction, etc.) is documented in
docs/edge_cases.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from .config_loader import Config, _norm


@dataclass
class NormalizedAnswers:
    by_id: Dict[str, List[str]] = field(default_factory=dict)
    free_text: str = ""
    unmatched_questions: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def codes(self, qid: str) -> List[str]:
        return self.by_id.get(qid, [])

    def single(self, qid: str):
        vals = self.by_id.get(qid, [])
        return vals[0] if vals else None


def normalize(survey_answers: List[dict], cfg: Config) -> NormalizedAnswers:
    out = NormalizedAnswers()
    text_to_id = cfg.text_to_id()
    seen_ids = set()

    for block in survey_answers:
        q_text = block.get("question", "")
        answers = block.get("answers", []) or []
        qid = text_to_id.get(_norm(q_text))

        if qid is None:
            out.unmatched_questions.append(q_text)
            out.warnings.append(f"Unknown question ignored: {q_text!r}")
            continue
        if qid in seen_ids:
            out.warnings.append(f"Duplicate question block for {qid!r}; later one ignored")
            continue
        seen_ids.add(qid)

        qtype = cfg.question_type(qid)

        # Free-text question: keep the raw string verbatim (LLM handles it later).
        if qtype == "text":
            out.free_text = (answers[0].strip() if answers else "")
            continue

        # Map each selected label -> code, dropping unknown labels with a warning.
        label_map = cfg.label_to_code(qid)
        codes: List[str] = []
        for label in answers:
            code = label_map.get(_norm(label))
            if code is None:
                out.warnings.append(f"Unknown answer for {qid!r} ignored: {label!r}")
                continue
            if code not in codes:  # de-dup
                codes.append(code)

        # A `single` question should yield exactly one code; be defensive.
        if qtype == "single" and len(codes) > 1:
            out.warnings.append(
                f"Question {qid!r} is single-select but got {len(codes)} answers; "
                f"keeping the first ({codes[0]!r})"
            )
            codes = codes[:1]

        out.by_id[qid] = codes

    return out
