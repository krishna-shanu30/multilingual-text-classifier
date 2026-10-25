from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "results"

df = pd.read_csv(ROOT / "data" / "train.csv").dropna(
    subset=["text", "labels"]
).copy()
df["text"] = df["text"].astype(str).str.strip()
df["labels"] = df["labels"].astype(str).str.strip()
df = df[df["text"].str.len() > 0].copy()
df["normalized_text"] = df["text"].str.replace(
    r"\s+", " ", regex=True
).str.casefold()

label_counts = df.groupby("normalized_text")["labels"].nunique()
conflicts = set(label_counts[label_counts > 1].index)
df = df[~df["normalized_text"].isin(conflicts)]
df = df.drop_duplicates("normalized_text").reset_index(drop=True)

train, test = train_test_split(
    df, test_size=0.20, random_state=42, stratify=df["labels"]
)

features = FeatureUnion([
    ("char", TfidfVectorizer(
        analyzer="char", ngram_range=(2, 5), min_df=2,
        max_features=200_000, sublinear_tf=True, dtype="float32",
    )),
    ("word", TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2), min_df=2,
        max_features=100_000, sublinear_tf=True, dtype="float32",
    )),
])

model = Pipeline([
    ("features", features),
    ("classifier", LinearSVC(C=1.0)),
])

print("Training the selected candidate...")
model.fit(train["text"], train["labels"])
pred = model.predict(test["text"])

report = classification_report(
    test["labels"], pred, output_dict=True, zero_division=0
)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv(OUT / "svm_recheck_report.csv")

labels = sorted(df["labels"].unique())
matrix = confusion_matrix(test["labels"], pred, labels=labels)
pd.DataFrame(
    matrix, index=labels, columns=labels
).rename_axis("Actual").to_csv(OUT / "svm_confusion_matrix.csv")

errors = test[["text", "labels"]].copy()
errors["predicted"] = pred
errors["correct"] = errors["labels"] == errors["predicted"]
errors["text_length"] = errors["text"].str.len()
errors.loc[~errors["correct"]].to_csv(
    OUT / "svm_misclassified_examples.csv", index=False
)

errors["length_bucket"] = pd.cut(
    errors["text_length"],
    bins=[-1, 20, 50, 100, 250, float("inf")],
    labels=["0-20", "21-50", "51-100", "101-250", "251+"],
)
bucket_results = errors.groupby(
    "length_bucket", observed=False
).agg(
    samples=("correct", "size"),
    accuracy=("correct", "mean"),
)
bucket_results.to_csv(OUT / "svm_length_performance.csv")

print("\nPerformance by text length:")
print(bucket_results.to_string())
print("\nMost frequent misclassifications:")
wrong = errors.loc[~errors["correct"]]
if wrong.empty:
    print("No errors on this test split.")
else:
    print(
        wrong.groupby(["labels", "predicted"])
        .size()
        .sort_values(ascending=False)
        .head(15)
        .to_string()
    )

print(f"\nSaved detailed reports to {OUT}")
