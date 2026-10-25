"""Inference helpers wrapping the original serialized classifier."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import re
import unicodedata
from typing import Any

import joblib
import numpy as np

from .languages import LANGUAGES

ROOT = Path(__file__).resolve().parents[2]


def load_artifacts(
    model_dir: str | Path = ROOT,
    use_candidate: bool = False,
) -> tuple[Any, Any, Any]:
    """Load legacy artifacts by default, or the SVM pipeline when explicitly requested."""
    model_dir = Path(model_dir)
    candidate = model_dir / "experiments" / "candidates" / "language_pipeline_svm.pkl"
    if use_candidate:
        if not candidate.exists():
            raise FileNotFoundError(f"Candidate model not found: {candidate}")
        return joblib.load(candidate), None, None

    required = [
        model_dir / "language_model.pkl",
        model_dir / "tfidf_vectorizer.pkl",
        model_dir / "label_encoder.pkl",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing model artifact(s): " + ", ".join(missing))
    return tuple(joblib.load(p) for p in required)  # type: ignore[return-value]


def detect_scripts(text: str) -> dict[str, int]:
    """Count Unicode script families heuristically; this is not language detection."""
    families = {
        "Latin": ((0x0041, 0x024F), (0x1E00, 0x1EFF)),
        "Arabic": ((0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF)),
        "Devanagari": ((0x0900, 0x097F),),
        "Cyrillic": ((0x0400, 0x052F),),
        "Greek": ((0x0370, 0x03FF),),
        "Han": ((0x4E00, 0x9FFF),),
        "Hiragana": ((0x3040, 0x309F),),
        "Katakana": ((0x30A0, 0x30FF),),
        "Thai": ((0x0E00, 0x0E7F),),
    }
    counts: Counter[str] = Counter()
    for char in text:
        if char.isspace() or unicodedata.category(char).startswith("P") or unicodedata.category(char).startswith("N"):
            continue
        cp = ord(char)
        found = False
        for family, ranges in families.items():
            if any(start <= cp <= end for start, end in ranges):
                counts[family] += 1
                found = True
                break
        if not found and unicodedata.category(char).startswith("L"):
            counts["Other"] += 1
    return dict(counts.most_common())


def _code_to_name(code: str) -> str:
    return LANGUAGES.get(str(code), str(code).upper())


def predict_text(text: str, model: Any, vectorizer: Any, encoder: Any, top_k: int = 5) -> dict[str, Any]:
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("Enter some text before running language detection.")

    # The candidate is a Pipeline with a LinearSVC classifier.
    if vectorizer is None and encoder is None and hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function([cleaned]))[0]
        classes = np.asarray(model.classes_)
        order = np.argsort(scores)[::-1][:max(1, min(top_k, len(scores)))]
        predicted_code = str(classes[int(order[0])])
        top_score = float(scores[int(order[0])])
        gap = float(scores[int(order[0])] - scores[int(order[1])]) if len(order) > 1 else top_score
        candidates = [
            {
                "language_code": str(classes[int(idx)]),
                "language": _code_to_name(str(classes[int(idx)])),
                "confidence": float(scores[int(idx)]),
            }
            for idx in order
        ]
        scripts = detect_scripts(cleaned)
        if len(cleaned) < 12:
            status = "Short input — treat prediction cautiously"
        elif gap < 0.10:
            status = "Ambiguous — SVM decision scores are close"
        else:
            status = "Prediction available — SVM scores are not probabilities"
        return {
            "text": cleaned,
            "language_code": predicted_code,
            "language": _code_to_name(predicted_code),
            "confidence": top_score,
            "score_type": "SVM decision score (not probability)",
            "top_two_gap": gap,
            "status": status,
            "candidates": candidates,
            "character_count": len(cleaned),
            "word_count": len(re.findall(r"\\S+", cleaned)),
            "script_counts": scripts,
            "script_summary": ", ".join(f"{k}: {v}" for k, v in scripts.items()) or "No letters detected",
        }

    features = vectorizer.transform([cleaned])
    probabilities = model.predict_proba(features)[0]
    order = np.argsort(probabilities)[::-1][:max(1, min(top_k, len(probabilities)))]
    model_class_ids = np.asarray(model.classes_)[order]
    label_codes = encoder.inverse_transform(model_class_ids.astype(int))
    candidates = [
        {"language_code": str(code), "language": _code_to_name(str(code)), "confidence": float(probabilities[idx])}
        for idx, code in zip(order, label_codes)
    ]
    prediction_id = int(model.classes_[int(np.argmax(probabilities))])
    predicted_code = str(encoder.inverse_transform([prediction_id])[0])
    confidence = float(np.max(probabilities))
    sorted_probabilities = np.sort(probabilities)
    top_two_gap = (
        float(sorted_probabilities[-1] - sorted_probabilities[-2])
        if len(sorted_probabilities) >= 2
        else confidence
    )

    # Heuristic warnings, not calibrated probability guarantees.
    if len(cleaned) < 12:
        status = "Short input — treat prediction cautiously"
    elif confidence < 0.55 or top_two_gap < 0.10:
        status = "Ambiguous — consider more context"
    else:
        status = "Prediction available"
    scripts = detect_scripts(cleaned)
    return {
        "text": cleaned,
        "language_code": predicted_code,
        "language": _code_to_name(predicted_code),
        "confidence": confidence,
        "top_two_gap": top_two_gap,
        "status": status,
        "candidates": candidates,
        "character_count": len(cleaned),
        "word_count": len(re.findall(r"\S+", cleaned)),
        "script_counts": scripts,
        "script_summary": ", ".join(f"{k}: {v}" for k, v in scripts.items()) or "No letters detected",
    }


def explain_prediction(text: str, result: dict[str, Any], model: Any, vectorizer: Any, encoder: Any, limit: int = 8) -> list[dict[str, Any]]:
    """Return influential active character n-grams when legacy artifacts support it."""
    if vectorizer is None or encoder is None:
        return []
    if not hasattr(model, "coef_") or not hasattr(vectorizer, "get_feature_names_out"):
        return []
    matrix = vectorizer.transform([text])
    if matrix.nnz == 0:
        return []
    code_to_id = {str(code): int(idx) for idx, code in enumerate(encoder.classes_)}
    class_id = code_to_id.get(result["language_code"])
    if class_id is None or class_id >= model.coef_.shape[0]:
        return []
    names = vectorizer.get_feature_names_out()
    active_indices = matrix.indices
    weights = model.coef_[class_id, active_indices] * matrix.data
    order = np.argsort(weights)[::-1][:limit]
    return [{"ngram": str(names[active_indices[i]]), "contribution": float(weights[i])} for i in order if weights[i] > 0]
