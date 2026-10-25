from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from src.language_intelligence.hybrid_detector import load_svm

df = pd.read_csv(ROOT / "data" / "train.csv")
df = df.dropna(subset=["text", "labels"]).copy()
df["text"] = df["text"].astype(str).str.strip()
df["labels"] = df["labels"].astype(str)
df = df[df["text"] != ""]
df["normalized"] = (
    df["text"].str.casefold().str.replace(r"\s+", " ", regex=True)
)

conflicts = df.groupby("normalized")["labels"].nunique()
df = df[~df["normalized"].isin(set(conflicts[conflicts > 1].index))]
df = df.drop_duplicates(subset=["normalized"])

_, test = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["labels"],
)

model = load_svm()
texts = test["text"].tolist()
expected = test["labels"].tolist()
predicted = [str(x) for x in model.predict(texts)]

print(f"Test rows: {len(test)}")
print(f"Accuracy: {accuracy_score(expected, predicted):.4%}")
print(classification_report(expected, predicted, zero_division=0))

errors = test[["text", "labels"]].copy()
errors["predicted"] = predicted
errors = errors[errors["labels"] != errors["predicted"]]
errors.to_csv(
    ROOT / "experiments" / "results" / "svm_full_test_errors.csv",
    index=False,
    encoding="utf-8",
)

print(f"\nMisclassified examples: {len(errors)}")
print("Saved to experiments/results/svm_full_test_errors.csv")
print(errors.head(30).to_string(index=False))
