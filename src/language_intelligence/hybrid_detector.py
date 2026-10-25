from pathlib import Path
from functools import lru_cache
import re

import fasttext
import joblib
import numpy as np


ROOT = Path(__file__).resolve().parents[2]

SUPPORTED_LANGUAGES = {
    "ar", "bg", "de", "el", "en",
    "es", "fr", "hi", "it", "ja",
    "nl", "pl", "pt", "ru", "sw",
    "th", "tr", "ur", "vi", "zh",
}


@lru_cache(maxsize=1)
def load_fasttext():
    path = ROOT / "models" / "lid.176.ftz"

    if not path.exists():
        raise FileNotFoundError(f"Missing fastText model: {path}")

    return fasttext.load_model(str(path))


@lru_cache(maxsize=1)
def load_svm():
    path = (
        ROOT
        / "experiments"
        / "candidates"
        / "language_pipeline_svm.pkl"
    )

    if not path.exists():
        return None

    return joblib.load(path)


def detect_language(text):
    text = re.sub(r"\s+", " ", str(text or "")).strip()

    if not text:
        return {
            "language": None,
            "confidence": None,
            "status": "Empty input",
            "fasttext_language": None,
            "svm_language": None,
            "candidates": [],
        }

    ft_model = load_fasttext()

    # Get all 176 labels, then restrict to our supported languages.
    labels, probabilities = ft_model.predict(
        text,
        k=len(ft_model.get_labels()),
    )

    ranked = sorted(
        [
            (label.replace("__label__", ""), float(prob))
            for label, prob in zip(labels, probabilities)
            if label.replace("__label__", "") in SUPPORTED_LANGUAGES
        ],
        key=lambda item: item[1],
        reverse=True,
    )

    if not ranked:
        return {
            "language": None,
            "confidence": None,
            "status": "No supported language detected",
            "fasttext_language": None,
            "svm_language": None,
            "candidates": [],
        }

    ft_language, ft_confidence = ranked[0]

    svm_model = load_svm()
    svm_language = None

    if svm_model is not None:
        svm_language = str(svm_model.predict([text])[0])

    # Conservative initial rules; validate these on held-out data
    # before interpreting them as reliable uncertainty thresholds.
    if len(text) < 12:
        status = "Short text — prediction may be unreliable"
    elif ft_confidence < 0.50:
        status = "Low fastText confidence"
    elif svm_language and svm_language != ft_language:
        status = "Models disagree — review prediction"
    else:
        status = "Prediction available"

    return {
        "language": ft_language,
        "confidence": ft_confidence,
        "status": status,
        "fasttext_language": ft_language,
        "svm_language": svm_language,
        "candidates": [
            {
                "language": code,
                "confidence": probability,
            }
            for code, probability in ranked[:5]
        ],
    }
