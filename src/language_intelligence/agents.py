"""Lightweight, local-first analysis agents. LLM support is optional and never required for prediction."""
from __future__ import annotations
import json
import os
import urllib.error
import urllib.request
from typing import Any


def run_quality_agent(text: str, prediction: dict[str, Any]) -> dict[str, Any]:
    notes = []
    if len(text.strip()) < 12:
        notes.append("The sample is short; add a longer sentence for a more dependable prediction.")
    if prediction["confidence"] < 0.55:
        notes.append("The model's top class is not strongly separated; review the alternative candidates.")
    if len(prediction.get("script_counts", {})) > 1:
        notes.append("Multiple writing systems were detected. This can indicate mixed-script text, names, or borrowed words.")
    if not notes:
        notes.append("No basic input-quality warning was triggered. This does not guarantee the prediction is correct.")
    return {"agent": "Input & Confidence Agent", "status": "review" if len(notes) > 1 or prediction["confidence"] < .55 else "ok", "findings": notes}


def run_script_agent(prediction: dict[str, Any]) -> dict[str, Any]:
    scripts = prediction.get("script_counts", {})
    dominant = max(scripts, key=scripts.get) if scripts else "None detected"
    return {"agent": "Script Profiling Agent", "dominant_script": dominant, "script_counts": scripts,
            "note": "Unicode script profiling is supporting evidence, not a substitute for language identification."}


def run_report_agent(text: str, prediction: dict[str, Any], quality: dict[str, Any], script: dict[str, Any]) -> dict[str, Any]:
    return {
        "summary": f"The baseline classifier predicts {prediction['language']} ({prediction['language_code']}) with a model score of {prediction['confidence']:.1%}.",
        "input_profile": {"characters": prediction["character_count"], "whitespace_separated_tokens": prediction["word_count"], "scripts": script["script_counts"]},
        "quality_findings": quality["findings"],
        "candidates": prediction["candidates"],
        "limitations": ["Confidence scores are model outputs and may be miscalibrated.", "The baseline model predicts one label for the full input and is not a word-level code-switching detector."],
    }


def run_local_llm_explanation(text: str, report: dict[str, Any], model_name: str | None = None, timeout: int = 45) -> str:
    """Call a locally running Ollama instance. Raises a clear error if unavailable."""
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    model_name = model_name or os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
    prompt = (
        "Explain the following language-identification result for a developer. Use only the supplied evidence; "
        "do not claim certainty or invent linguistic facts. Keep it under 120 words.\n\n"
        f"Text sample: {text[:1500]}\nResult JSON: {json.dumps(report, ensure_ascii=False)}"
    )
    payload = json.dumps({"model": model_name, "prompt": prompt, "stream": False}).encode("utf-8")
    request = urllib.request.Request(base_url + "/api/generate", data=payload, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
        return str(data.get("response", "")).strip() or "The local model returned an empty explanation."
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not reach Ollama at {base_url}. Start Ollama and pull '{model_name}'. Details: {exc}") from exc
