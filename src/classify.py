"""
classify.py — the deterministic brain of the layer.

Pure functions that turn NormalizedAnswers + Config into the structured pieces
of the profile:
  * classify_goal              -> Goal
  * classify_intensity         -> Intensity
  * derive_contraindications   -> Constraints
  * evaluate_flags             -> {flag: bool}
  * derive_equipment_level     -> EquipmentLevel value

The *procedure* is here; the *data* (tables, weights, rules) lives in config/.
A small rule engine (`_match`) interprets the declarative flag/equipment rules,
so new flags need only a config edit.
"""

from __future__ import annotations

from typing import Dict, List

from .config_loader import Config
from .contracts import (
    Constraints,
    EquipmentLevel,
    Goal,
    Intensity,
)
from .normalize import NormalizedAnswers


# --------------------------------------------------------------------------
# Generic declarative rule engine (shared by flags + equipment level).
# A condition is {field, op, values}; codes are the normalized answer codes
# for that field.
# --------------------------------------------------------------------------

def _match(condition: dict, answers: NormalizedAnswers) -> bool:
    codes = set(answers.codes(condition["field"]))
    values = set(condition.get("values", []))
    op = condition["op"]

    if op == "any_of":
        return bool(codes & values)
    if op == "all_of":
        return values.issubset(codes)
    if op == "none_of":
        return not (codes & values)
    if op == "equals":
        # single-select equality
        return len(codes) == 1 and next(iter(codes)) == next(iter(values))
    raise ValueError(f"Unknown condition op: {op!r}")


def _all_match(conditions: List[dict], answers: NormalizedAnswers) -> bool:
    return all(_match(c, answers) for c in conditions)


# --------------------------------------------------------------------------
# Goal
# --------------------------------------------------------------------------

def classify_goal(answers: NormalizedAnswers, cfg: Config) -> Goal:
    code = answers.single("goal")
    goals_cfg = cfg.goals["goals"]
    if code is None or code not in goals_cfg:
        if code is not None:
            answers.warnings.append(f"Unrecognized goal {code!r}; using default")
        else:
            answers.warnings.append("Goal missing; using default")
        code = cfg.goals["default_goal"]
    return Goal(code=code, focus=goals_cfg[code]["focus"])


# --------------------------------------------------------------------------
# Intensity (weighted scoring -> band)
# --------------------------------------------------------------------------

def classify_intensity(answers: NormalizedAnswers, cfg: Config) -> Intensity:
    points = cfg.intensity["points"]
    drivers = {
        "experience": answers.single("experience"),
        "frequency": answers.single("frequency"),
        "session_minutes": answers.single("session_minutes"),
    }

    score = 0
    missing = []
    for dim, code in drivers.items():
        table = points[dim]
        if code is None or code not in table:
            missing.append(dim)
            continue  # missing dimension contributes 0 (conservative)
        score += table[code]

    if missing:
        answers.warnings.append(
            f"Intensity drivers missing/unknown: {missing}; treated as 0 points each"
        )

    level = cfg.intensity["default_level"]
    for band in cfg.intensity["bands"]:
        if band["min"] <= score <= band["max"]:
            level = band["level"]
            break

    return Intensity(level=level, score=score, drivers=drivers)


# --------------------------------------------------------------------------
# Contraindications + cleaned limitations list
# --------------------------------------------------------------------------

def _clean_limitations(answers: NormalizedAnswers) -> List[str]:
    """Resolve the 'Нет' + real-limitation contradiction: if any real
    limitation is selected, drop the 'none' sentinel."""
    codes = list(answers.codes("limitations"))
    real = [c for c in codes if c != "none"]
    if real and "none" in codes:
        answers.warnings.append("Both 'none' and real limitations selected; ignoring 'none'")
        return real
    return codes


def derive_contraindications(answers: NormalizedAnswers, cfg: Config) -> Constraints:
    limitations = _clean_limitations(answers)
    mapping = cfg.contraindications["map"]

    tags: List[str] = []
    for code in limitations:
        for tag in mapping.get(code, []):
            if tag not in tags:
                tags.append(tag)

    # Stable, deterministic ordering for reproducible output.
    return Constraints(
        limitations=limitations,
        contraindicated_exercise_types=sorted(tags),
    )


# --------------------------------------------------------------------------
# Structural flags
# --------------------------------------------------------------------------

def evaluate_flags(answers: NormalizedAnswers, cfg: Config) -> Dict[str, bool]:
    result: Dict[str, bool] = {}
    for flag_name, conditions in cfg.flags["flags"].items():
        result[flag_name] = _all_match(conditions, answers)
    return result


def derive_equipment_level(answers: NormalizedAnswers, cfg: Config) -> str:
    spec = cfg.flags["equipment_level"]
    for rule in spec["rules"]:
        if _match(rule, answers):
            return rule["level"]
    return spec.get("default", EquipmentLevel.NONE.value)
