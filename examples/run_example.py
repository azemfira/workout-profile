#!/usr/bin/env python3
"""
Run the interpretation layer on a survey JSON file and print the user_profile.

Usage:
    python examples/run_example.py [path/to/survey.json]

If ANTHROPIC_API_KEY is set, the free-text field is interpreted by a live LLM
call; otherwise an offline keyword stub is used (so this always runs).
"""

import json
import os
import sys

# Make `src` importable when run from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import build_profile  # noqa: E402


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(__file__), "sample_input.json"
    )
    with open(path, "r", encoding="utf-8") as fh:
        survey_answers = json.load(fh)

    profile = build_profile(survey_answers)  # interpreter auto-selected

    mode = "LLM" if os.environ.get("ANTHROPIC_API_KEY") else "STUB (offline)"
    print(f"# free-text interpreter: {mode}\n")
    print(json.dumps(profile.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
