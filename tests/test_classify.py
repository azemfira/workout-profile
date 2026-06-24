"""Tests for the deterministic classification functions."""

import unittest

from src.classify import (
    classify_goal,
    classify_intensity,
    derive_contraindications,
    derive_equipment_level,
    evaluate_flags,
)
from src.config_loader import load_config
from src.normalize import normalize


def _ans(cfg, **blocks):
    """Helper: build NormalizedAnswers from {question_text: [answers]}."""
    survey = [{"question": q, "answers": a} for q, a in blocks.items()]
    return normalize(survey, cfg)


class TestGoal(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config()

    def test_goal_focus_mapping(self):
        a = _ans(self.cfg, **{"Главная цель": ["Выносливость"]})
        g = classify_goal(a, self.cfg)
        self.assertEqual(g.code, "endurance")
        self.assertEqual(g.focus, "cardiovascular")

    def test_missing_goal_falls_back_to_default(self):
        a = _ans(self.cfg, **{"Опыт": ["Средний"]})
        g = classify_goal(a, self.cfg)
        self.assertEqual(g.code, "maintain")
        self.assertTrue(any("Goal missing" in w for w in a.warnings))


class TestIntensity(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config()

    def test_low(self):
        a = _ans(
            self.cfg,
            **{"Опыт": ["Новичок"], "Сколько раз в неделю": ["1–2"], "Минут на тренировку": ["15–20"]},
        )
        i = classify_intensity(a, self.cfg)
        self.assertEqual((i.score, i.level), (0, "low"))

    def test_moderate(self):
        a = _ans(
            self.cfg,
            **{"Опыт": ["Средний"], "Сколько раз в неделю": ["3–4"], "Минут на тренировку": ["30–40"]},
        )
        i = classify_intensity(a, self.cfg)
        self.assertEqual((i.score, i.level), (3, "moderate"))

    def test_high(self):
        a = _ans(
            self.cfg,
            **{"Опыт": ["Продвинутый"], "Сколько раз в неделю": ["5+"], "Минут на тренировку": ["60+"]},
        )
        i = classify_intensity(a, self.cfg)
        self.assertEqual((i.score, i.level), (6, "high"))

    def test_novice_can_never_reach_high(self):
        # Even maxed-out frequency + duration, a novice caps at "moderate".
        a = _ans(
            self.cfg,
            **{"Опыт": ["Новичок"], "Сколько раз в неделю": ["5+"], "Минут на тренировку": ["60+"]},
        )
        i = classify_intensity(a, self.cfg)
        self.assertEqual(i.score, 4)
        self.assertEqual(i.level, "moderate")

    def test_missing_driver_contributes_zero(self):
        a = _ans(self.cfg, **{"Опыт": ["Продвинутый"]})  # freq + minutes missing
        i = classify_intensity(a, self.cfg)
        self.assertEqual(i.score, 2)  # only experience counts
        self.assertTrue(any("drivers missing" in w for w in a.warnings))


class TestContraindications(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config()

    def test_knees_mapping(self):
        a = _ans(self.cfg, **{"Ограничения / травмы": ["Колени"]})
        c = derive_contraindications(a, self.cfg)
        self.assertEqual(c.limitations, ["knees"])
        self.assertEqual(
            c.contraindicated_exercise_types,
            sorted(["deep_knee_flexion", "plyometric", "high_impact"]),
        )

    def test_multiple_limitations_union_no_duplicates(self):
        a = _ans(self.cfg, **{"Ограничения / травмы": ["Спина", "Колени"]})
        c = derive_contraindications(a, self.cfg)
        # high_impact is shared by both -> must appear once
        self.assertEqual(c.contraindicated_exercise_types.count("high_impact"), 1)

    def test_none_with_real_limitation_drops_none(self):
        a = _ans(self.cfg, **{"Ограничения / травмы": ["Нет", "Спина"]})
        c = derive_contraindications(a, self.cfg)
        self.assertEqual(c.limitations, ["back"])
        self.assertTrue(any("ignoring 'none'" in w for w in a.warnings))

    def test_none_only_yields_no_contraindications(self):
        a = _ans(self.cfg, **{"Ограничения / травмы": ["Нет"]})
        c = derive_contraindications(a, self.cfg)
        self.assertEqual(c.limitations, ["none"])
        self.assertEqual(c.contraindicated_exercise_types, [])


class TestFlagsAndEquipment(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config()

    def test_gym_flags_and_full_equipment(self):
        a = _ans(self.cfg, **{"Где занимаешься": ["В зале"]})
        flags = evaluate_flags(a, self.cfg)
        self.assertTrue(flags["has_gym_access"])
        self.assertFalse(flags["bodyweight_only"])
        self.assertEqual(derive_equipment_level(a, self.cfg), "full")

    def test_bodyweight_only_and_none_equipment(self):
        a = _ans(self.cfg, **{"Где занимаешься": ["Дома без оборудования", "На улице"]})
        flags = evaluate_flags(a, self.cfg)
        self.assertTrue(flags["bodyweight_only"])
        self.assertTrue(flags["outdoor_available"])
        self.assertEqual(derive_equipment_level(a, self.cfg), "none")

    def test_dumbbells_minimal_equipment(self):
        a = _ans(self.cfg, **{"Где занимаешься": ["Дома с гантелями"]})
        self.assertEqual(derive_equipment_level(a, self.cfg), "minimal")

    def test_pregnancy_sets_medical_clearance(self):
        a = _ans(self.cfg, **{"Ограничения / травмы": ["Беременность"]})
        flags = evaluate_flags(a, self.cfg)
        self.assertTrue(flags["is_pregnant"])
        self.assertTrue(flags["medical_clearance_recommended"])

    def test_rehab_focus_flag(self):
        a = _ans(self.cfg, **{"Главная цель": ["Восстановиться после травмы"]})
        flags = evaluate_flags(a, self.cfg)
        self.assertTrue(flags["is_rehab_focus"])


if __name__ == "__main__":
    unittest.main()
