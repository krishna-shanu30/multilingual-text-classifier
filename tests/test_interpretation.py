
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.language_intelligence.interpretation import interpret_text


def test_empty_text_is_rejected():
    try:
        interpret_text("", "Hindi")
    except ValueError as exc:
        assert "enter some text" in str(exc).lower()
    else:
        raise AssertionError("Expected ValueError for empty text")
