from pathlib import Path
import json
import time

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "train.csv"
OUTPUT_DIR = ROOT / "experiments" / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=["text", "labels"]).copy()
df["text"] = df["text"].astype(str).str.strip()
df["labels"] = df["labels"].astype(str).str.strip()
df = df[df["text"].str.len() > 0].copy()

# Normalize whitespace for duplicate detection.
df["normalized_text"] = df["text"].str.replace(r"\s+", " ", regex=True).str.casefold()

# Remove ambiguous text groups with more than one label.
label_counts = df.groupby("normalized_text")["labels"].nunique()
conflicting_texts = set(label_counts[label_counts > 1].index)
conflicts = df[df["normalized_text"].isin(conflicting_texts)].copy()
conflicts.to_csv(OUTPUT_DIR / "conflicting_labels.csv", index=False)

clean = df[~df["normalized_text"].isin(conflicting_texts)].copy()

# One row per normalized text prevents duplicate leakage.
clean = clean.drop_duplicates(subset="normalized_text", keep="first")
clean = clean[["text", "labels"]].reset_index(drop=True)

print(f"Original rows: {len(df):,}")
print(f"Conflicting text groups: {len(conflicting_texts):,}")
print(f"Rows in conflicting groups: {len(conflicts):,}")
print(f"Unique clean texts: {len(clean):,}")
print(f"Languages: {clean['labels'].nunique()}")

X_train, X_test, y_train, y_test = train_test_split(
    clean["text"],
    clean["labels"],
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=clean["labels"],
)

def combined_features():
    return FeatureUnion([
        ("char", TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 5),
            min_df=2,
            max_features=200_000,
            sublinear_tf=True,
            dtype="float32",
        )),
        ("word", TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=2,
            max_features=100_000,
            sublinear_tf=True,
            dtype="float32",
        )),
    ])

experiments = {
    "baseline_char_logreg": Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 5),
            max_features=5_000,
            dtype="float32",
        )),
        ("classifier", LogisticRegression(max_iter=1000)),
    ]),
    "expanded_char_logreg": Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 6),
            min_df=2,
            max_features=200_000,
            sublinear_tf=True,
            dtype="float32",
        )),
        ("classifier", LogisticRegression(max_iter=1500, C=2.0)),
    ]),
    "char_word_logreg": Pipeline([
        ("features", combined_features()),
        ("classifier", LogisticRegression(max_iter=1500, C=2.0)),
    ]),
    "char_word_linear_svm": Pipeline([
        ("features", combined_features()),
        ("classifier", LinearSVC(C=1.0)),
    ]),
}

results = []

for name, pipeline in experiments.items():
    print(f"\n{'=' * 65}\nTraining: {name}")
    started = time.time()

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(y_test, predictions, average="macro")
    weighted_f1 = f1_score(y_test, predictions, average="weighted")
    elapsed = time.time() - started

    print(f"Accuracy:   {accuracy:.4%}")
    print(f"Macro-F1:   {macro_f1:.4%}")
    print(f"Weighted-F1:{weighted_f1:.4%}")
    print(f"Fit + test: {elapsed:.1f} seconds")

    report = classification_report(
        y_test, predictions, output_dict=True, zero_division=0
    )
    pd.DataFrame(report).transpose().to_csv(
        OUTPUT_DIR / f"{name}_report.csv"
    )

    results.append({
        "model": name,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "seconds": elapsed,
    })

summary = pd.DataFrame(results).sort_values("macro_f1", ascending=False)
summary.to_csv(OUTPUT_DIR / "comparison.csv", index=False)

print("\n\nFINAL COMPARISON (sorted by macro-F1)")
print(summary.to_string(index=False))
print(f"\nReports saved in: {OUTPUT_DIR}")
print("No production model artifacts were overwritten.")
