from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from src.language_intelligence.hybrid_detector import (
    load_fasttext,
    load_svm,
    SUPPORTED_LANGUAGES,
)

df = pd.read_csv(ROOT / "data" / "train.csv")
df = df.dropna(subset=["text", "labels"]).copy()
df["text"] = df["text"].astype(str).str.strip()
df = df[df["text"] != ""]
df["labels"] = df["labels"].astype(str)
df["normalized"] = (
    df["text"].str.casefold().str.replace(r"\s+", " ", regex=True)
)

# Remove text that has conflicting labels, then deduplicate.
conflicts = df.groupby("normalized")["labels"].nunique()
bad_texts = set(conflicts[conflicts > 1].index)
df = df[~df["normalized"].isin(bad_texts)]
df = df.drop_duplicates(subset=["normalized"])

_, test = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["labels"],
)

# Up to 50 examples per language, using the same examples for both models.
sample = test.groupby("labels", group_keys=False).sample(
    n=50,
    random_state=42,
)

texts = sample["text"].tolist()
expected = sample["labels"].tolist()

print(f"Evaluating {len(texts)} examples...")
print("Loading models...")

ft_model = load_fasttext()
svm_model = load_svm()

fasttext_predictions = []
for text in texts:
    labels, probs = ft_model.predict(text, k=176)
    ranked = [
        (label.replace("__label__", ""), float(prob))
        for label, prob in zip(labels, probs)
        if label.replace("__label__", "") in SUPPORTED_LANGUAGES
    ]
    fasttext_predictions.append(
        max(ranked, key=lambda item: item[1])[0] if ranked else "unknown"
    )

svm_predictions = (
    [str(x) for x in svm_model.predict(texts)]
    if svm_model is not None
    else None
)

print("\nFASTTEXT ACCURACY:", f"{accuracy_score(expected, fasttext_predictions):.2%}")
print(classification_report(expected, fasttext_predictions, zero_division=0))

if svm_predictions is not None:
    print("\nSVM ACCURACY:", f"{accuracy_score(expected, svm_predictions):.2%}")
    print(classification_report(expected, svm_predictions, zero_division=0))

results = pd.DataFrame({
    "text": texts,
    "expected": expected,
    "fasttext": fasttext_predictions,
})
if svm_predictions is not None:
    results["svm"] = svm_predictions

out = ROOT / "experiments" / "results" / "detector_comparison.csv"
out.parent.mkdir(parents=True, exist_ok=True)
results.to_csv(out, index=False, encoding="utf-8")
print("\nSaved example-by-example results to:", out)
