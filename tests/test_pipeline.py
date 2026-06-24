"""End-to-end tests for build_profile, using the offline stub interpreter so the
whole flow (including the LLM-flag -> needs_human_review path) is deterministic."""

import unittest

from src.contracts import FreeTextSource, UserProfile
from src.free_text import FreeTextInterpreter, StubFreeTextInterpreter
from src.pipeline import build_profile


FULL_SURVEY = [
    {"question": "Главная цель", "answers": ["Набрать мышцы"]},
    {"question": "Где занимаешься", "answers": ["В зале", "Дома с гантелями"]},
    {"question": "Сколько раз в неделю", "answers": ["3–4"]},
    {"question": "Опыт", "answers": ["Средний"]},
    {"question": "Ограничения / травмы", "answers": ["Колени"]},
    {"question": "Минут на тренировку", "answers": ["30–40"]},
    {"question": "Свободное поле", "answers": ["Готовлюсь к свадьбе через 3 месяца, не люблю бег"]},
]


class _Always(FreeTextInterpreter):
    """Test double returning a fixed signal."""
    def __init__(self, signal):
        self._signal = signal
    def interpret(self, text):
        return self._signal


class TestPipeline(unittest.TestCase):
    def _build(self, survey):
        return build_profile(survey, interpreter=StubFreeTextInterpreter())

    def test_full_profile_shape_and_values(self):
        p = self._build(FULL_SURVEY)
        self.assertIsInstance(p, UserProfile)
        d = p.to_dict()

        # contract top-level keys
        self.assertEqual(
            set(d.keys()),
            {
                "schema_version", "goal", "intensity", "environment",
                "constraints", "flags", "free_text_signal", "meta",
            },
        )
        self.assertEqual(d["goal"], {"code": "build_muscle", "focus": "hypertrophy"})
        self.assertEqual(d["intensity"]["level"], "moderate")
        self.assertEqual(d["environment"]["equipment_level"], "full")
        self.assertIn("deep_knee_flexion", d["constraints"]["contraindicated_exercise_types"])
        self.assertTrue(d["flags"]["has_gym_access"])

    def test_free_text_signal_extracted_by_stub(self):
        p = self._build(FULL_SURVEY)
        sig = p.free_text_signal
        self.assertTrue(sig.provided)
        self.assertEqual(sig.source, FreeTextSource.STUB.value)
        self.assertIn("бег", sig.preferences.dislikes)

    def test_empty_free_text_skips_llm(self):
        survey = [b for b in FULL_SURVEY if b["question"] != "Свободное поле"]
        p = self._build(survey)
        self.assertFalse(p.free_text_signal.provided)
        self.assertEqual(p.free_text_signal.source, FreeTextSource.EMPTY.value)

    def test_llm_flag_triggers_needs_human_review(self):
        # Free text mentions surgery -> stub flags potential contraindication
        # -> deterministic layer must set needs_human_review, even though the
        # checkbox limitations are "Нет".
        survey = [
            {"question": "Главная цель", "answers": ["Поддерживать форму"]},
            {"question": "Ограничения / травмы", "answers": ["Нет"]},
            {"question": "Свободное поле", "answers": ["Недавно была операция на плече"]},
        ]
        p = self._build(survey)
        self.assertTrue(p.free_text_signal.mentions_potential_contraindication)
        self.assertTrue(p.flags["needs_human_review"])
        # ...but the deterministic contraindication list stays empty (LLM has no
        # authority to add tags directly).
        self.assertEqual(p.constraints.contraindicated_exercise_types, [])

    def test_medical_limitation_sets_needs_human_review(self):
        survey = [{"question": "Ограничения / травмы", "answers": ["Сердце/давление"]}]
        p = self._build(survey)
        self.assertTrue(p.flags["needs_human_review"])

    def test_deterministic_output_is_reproducible(self):
        a = build_profile(FULL_SURVEY, interpreter=StubFreeTextInterpreter()).to_dict()
        b = build_profile(FULL_SURVEY, interpreter=StubFreeTextInterpreter()).to_dict()
        self.assertEqual(a, b)

    def test_unknown_question_surfaced_in_meta(self):
        survey = FULL_SURVEY + [{"question": "Любимая музыка", "answers": ["Рок"]}]
        p = self._build(survey)
        self.assertIn("Любимая музыка", p.meta.unmatched_questions)


if __name__ == "__main__":
    unittest.main()
