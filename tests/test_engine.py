from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pytest
from language_intelligence.engine import detect_scripts, load_artifacts, predict_text


def test_original_artifacts_load():
    model, vectorizer, encoder = load_artifacts(ROOT)
    assert len(encoder.classes_) == 20
    assert len(vectorizer.vocabulary_) > 0
    assert hasattr(model, "predict_proba")


def test_prediction_returns_candidates():
    model, vectorizer, encoder = load_artifacts(ROOT)
    result = predict_text("This is a longer English sentence for language identification.", model, vectorizer, encoder)
    assert result["language"] == "English"
    assert 0 <= result["confidence"] <= 1
    assert len(result["candidates"]) == 5
    assert result["candidates"][0]["confidence"] >= result["candidates"][-1]["confidence"]


def test_empty_input_is_rejected():
    model, vectorizer, encoder = load_artifacts(ROOT)
    with pytest.raises(ValueError):
        predict_text("   ", model, vectorizer, encoder)


def test_script_profiler_detects_multiple_scripts():
    counts = detect_scripts("Hello नमस्ते")
    assert "Latin" in counts
    assert "Devanagari" in counts
