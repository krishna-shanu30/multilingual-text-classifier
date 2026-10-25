from pathlib import Path
import json
import time

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "train.csv"
OUT_DIR = ROOT / "experiments" / "candidates"
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH).dropna(subset=["text", "labels"]).copy()
df["text"] = df["text"].astype(str).str.strip()
df["labels"] = df["labels"].astype(str).str.strip()
df = df[df["text"].str.len() > 0].copy()

# Remove conflicting labels and normalized duplicates to reduce leakage.
df["normalized_text"] = (
    df["text"].str.replace(r"\s+", " ", regex=True).str.casefold()
)
label_counts = df.groupby("normalized_text")["labels"].nunique()
conflicts = set(label_counts[label_counts > 1].index)
df = df[~df["normalized_text"].isin(conflicts)]
df = df.drop_duplicates("normalized_text", keep="first")

X_train, X_test, y_train, y_test = train_test_split(
    df["text"],
    df["labels"],
    test_size=0.20,
    random_state=42,
    stratify=df["labels"],
)

features = FeatureUnion([
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

model = Pipeline([
    ("features", features),
    ("classifier", LinearSVC(C=1.0)),
])

print(f"Training rows: {len(X_train):,}")
print(f"Test rows: {len(X_test):,}")
print("Training combined character + word TF-IDF LinearSVC...")

started = time.time()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
macro_f1 = f1_score(y_test, predictions, average="macro")
weighted_f1 = f1_score(y_test, predictions, average="weighted")

report = classification_report(
    y_test, predictions, output_dict=True, zero_division=0
)

joblib.dump(model, OUT_DIR / "language_pipeline_svm.pkl")
pd.DataFrame(report).transpose().to_csv(OUT_DIR / "classification_report.csv")

summary = {
    "accuracy": accuracy,
    "macro_f1": macro_f1,
    "weighted_f1": weighted_f1,
    "train_rows": len(X_train),
    "test_rows": len(X_test),
    "conflicting_text_groups_removed": len(conflicts),
    "normalized_unique_rows": len(df),
    "training_seconds": round(time.time() - started, 2),
}

(OUT_DIR / "metrics.json").write_text(
    json.dumps(summary, indent=2), encoding="utf-8"
)

print("\nCandidate results:")
print(f"Accuracy:   {accuracy:.4%}")
print(f"Macro-F1:   {macro_f1:.4%}")
print(f"Weighted-F1:{weighted_f1:.4%}")
print(f"Elapsed:    {summary['training_seconds']} seconds")
print(f"\nSaved candidate to: {OUT_DIR / 'language_pipeline_svm.pkl'}")
print("Existing production model files were not changed.")
