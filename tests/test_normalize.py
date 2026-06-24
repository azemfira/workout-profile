"""Tests for the deterministic normalization step."""

import unittest

from src.config_loader import load_config
from src.normalize import normalize


class TestNormalize(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config()

    def test_basic_label_to_code(self):
        ans = normalize(
            [
                {"question": "Главная цель", "answers": ["Набрать мышцы"]},
                {"question": "Опыт", "answers": ["Средний"]},
            ],
            self.cfg,
        )
        self.assertEqual(ans.single("goal"), "build_muscle")
        self.assertEqual(ans.single("experience"), "intermediate")

    def test_multi_select_and_dedup(self):
        ans = normalize(
            [{"question": "Где занимаешься", "answers": ["В зале", "Дома с гантелями", "В зале"]}],
            self.cfg,
        )
        self.assertEqual(ans.codes("locations"), ["gym", "home_dumbbells"])

    def test_whitespace_insensitive_question_match(self):
        # extra/odd whitespace in the question text must still match
        ans = normalize(
            [{"question": "  Главная   цель ", "answers": ["Похудеть"]}],
            self.cfg,
        )
        self.assertEqual(ans.single("goal"), "lose_weight")

    def test_unknown_question_is_recorded(self):
        ans = normalize(
            [{"question": "Любимый цвет", "answers": ["Синий"]}],
            self.cfg,
        )
        self.assertIn("Любимый цвет", ans.unmatched_questions)
        self.assertTrue(ans.warnings)

    def test_unknown_answer_label_dropped_with_warning(self):
        ans = normalize(
            [{"question": "Опыт", "answers": ["Полубог"]}],
            self.cfg,
        )
        self.assertEqual(ans.codes("experience"), [])
        self.assertTrue(any("Unknown answer" in w for w in ans.warnings))

    def test_single_select_with_multiple_answers_keeps_first(self):
        ans = normalize(
            [{"question": "Опыт", "answers": ["Новичок", "Продвинутый"]}],
            self.cfg,
        )
        self.assertEqual(ans.codes("experience"), ["novice"])
        self.assertTrue(any("single-select" in w for w in ans.warnings))

    def test_free_text_captured_verbatim(self):
        ans = normalize(
            [{"question": "Свободное поле", "answers": ["  не люблю бег  "]}],
            self.cfg,
        )
        self.assertEqual(ans.free_text, "не люблю бег")

    def test_missing_free_text_is_empty(self):
        ans = normalize([{"question": "Опыт", "answers": ["Средний"]}], self.cfg)
        self.assertEqual(ans.free_text, "")


if __name__ == "__main__":
    unittest.main()
