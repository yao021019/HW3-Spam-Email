"""Train a simple baseline classifier (Logistic Regression) and optionally an SVM.
Saves model and vectorizer to `models/` and writes metrics to `outputs/metrics.json`.

Usage:
    python scripts/train_baseline.py --data data/sms_spam.csv --out models/ --max-rows 5000
"""
import argparse
import json
import os
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from sklearn.utils import resample
try:
    from imblearn.over_sampling import SMOTE
except Exception:
    SMOTE = None
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC


def load_data(path: str, max_rows: int = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if max_rows:
        df = df.head(max_rows)
    df = df.dropna()
    if "label" not in df.columns or "message" not in df.columns:
        raise SystemExit("Expected columns 'label' and 'message' in dataset")
    return df


def preprocess_text(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    s = s.lower()
    # replace urls, emails, phones, numbers
    s = re.sub(r"https?://\S+|www\.\S+", " <URL> ", s)
    s = re.sub(r"\S+@\S+", " <EMAIL> ", s)
    s = re.sub(r"\+?\d[\d\-() ]{6,}\d", " <PHONE> ", s)
    s = re.sub(r"\d+", " <NUM> ", s)
    # remove excessive punctuation
    s = re.sub(r"[^\w\s<>]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def preprocess(df: pd.DataFrame) -> Tuple:
    X_text = df["message"].astype(str).tolist()
    y = (df["label"].str.lower() == "spam").astype(int).to_numpy()
    # Use a slightly stricter vectorizer to reduce noise (min_df to ignore rare tokens)
    vec = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=2)
    X = vec.fit_transform(X_text)
    return X, y, vec


def train_and_eval(X_train, X_test, y_train, y_test, model, model_name: str, out_dir: str):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    report = classification_report(y_test, preds, output_dict=True, zero_division=0)
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, f"{model_name}.joblib")
    joblib.dump(model, model_path)
    print(f"Saved {model_name} to {model_path}")
    return report, model_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/sms_spam.csv")
    p.add_argument("--out", default="models")
    p.add_argument("--metrics-out", default="outputs/metrics.json")
    p.add_argument("--max-rows", type=int, default=0)
    p.add_argument("--preprocess", action="store_true", help="Apply lightweight text preprocessing before vectorization")
    p.add_argument("--smote", action="store_true", help="Apply SMOTE to X_train after vectorization (may convert to dense arrays)")
    p.add_argument("--oversample", action="store_true", help="Simple upsample minority class in training data before vectorization")
    p.add_argument("--train-svm", action="store_true")
    args = p.parse_args()
    df = load_data(args.data, max_rows=(args.max_rows or None))
    if args.preprocess:
        print("Applying text preprocessing to messages...")
        df['message'] = df['message'].astype(str).map(preprocess_text)
    # Report class distribution and optionally upsample minority class
    print("Label distribution before any resampling:\n", df['label'].value_counts())
    if args.oversample:
        # simple pandas upsample minority class to match majority
        counts = df['label'].value_counts()
        if len(counts) > 1:
            majority_label = counts.idxmax()
            majority_count = counts.max()
            parts = []
            for lab, cnt in counts.items():
                part = df[df['label'] == lab]
                if lab == majority_label:
                    parts.append(part)
                else:
                    # upsample with replacement
                    part_up = part.sample(n=majority_count, replace=True, random_state=42)
                    parts.append(part_up)
            df = pd.concat(parts).sample(frac=1, random_state=42).reset_index(drop=True)
            print("Performed simple oversampling to balance classes. New distribution:\n", df['label'].value_counts())
    X, y, vec = preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Optionally apply SMOTE on vectorized features (converts to dense arrays)
    if args.smote:
        if SMOTE is None:
            print("SMOTE requested but imbalanced-learn is not available. Install imbalanced-learn.")
        else:
            print("Applying SMOTE to training data (converting to dense array)...")
            # convert to dense (be careful with memory on large data)
            X_train_dense = X_train.toarray() if hasattr(X_train, "toarray") else X_train
            sm = SMOTE(random_state=42)
            X_res, y_res = sm.fit_resample(X_train_dense, y_train)
            # convert back to sparse if needed
            X_train = X_res
            y_train = y_res

    os.makedirs(args.out, exist_ok=True)
    vec_path = os.path.join(args.out, "vectorizer.joblib")
    joblib.dump(vec, vec_path)
    print(f"Saved vectorizer to {vec_path}")

    metrics = {}

    # Logistic Regression baseline
    # Use class_weight='balanced' to address class imbalance and liblinear solver for small datasets
    lr = LogisticRegression(max_iter=1000, class_weight='balanced', solver='liblinear')
    lr_report, lr_path = train_and_eval(X_train, X_test, y_train, y_test, lr, "logistic_regression", args.out)
    metrics["logistic_regression"] = lr_report

    # Optional SVM comparator
    if args.train_svm:
        svm = LinearSVC()
        svm_report, svm_path = train_and_eval(X_train, X_test, y_train, y_test, svm, "svm_linear", args.out)
        metrics["svm_linear"] = svm_report

    os.makedirs(os.path.dirname(args.metrics_out), exist_ok=True)
    with open(args.metrics_out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Wrote metrics to {args.metrics_out}")


if __name__ == "__main__":
    main()
