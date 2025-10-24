"""
Check for a high-confidence spam example (prob >= 0.9) using current models in models/.
Prints summary and one example if found.
"""
import os
import json
import sys
from joblib import load
import pandas as pd
import numpy as np

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "sms_spam.csv")

VEC_PATH = os.path.join(MODEL_DIR, "vectorizer.joblib")

# Prefer SVM artifact if present, else logistic
svm_candidates = ["svm_linear.joblib", "svm.joblib", "linear_svc.joblib"]
logistic_candidates = ["logistic_regression.joblib", "logistic.joblib", "model.joblib"]

MODEL_PATH = None
for c in svm_candidates + logistic_candidates:
    p = os.path.join(MODEL_DIR, c)
    if os.path.exists(p):
        MODEL_PATH = p
        break

if MODEL_PATH is None or not os.path.exists(VEC_PATH):
    print("Model or vectorizer not found in models/. Please run training first.")
    sys.exit(2)

model = load(MODEL_PATH)
vec = load(VEC_PATH)

# load dataset, be flexible about headers
try:
    df = pd.read_csv(DATA_PATH)
except Exception as e:
    print(f"Failed to read data at {DATA_PATH}: {e}")
    sys.exit(3)

# detect label/text columns
def detect_cols(df):
    if 'label' in df.columns and 'message' in df.columns:
        return 'label','message'
    if df.shape[1] >= 2:
        return df.columns[0], df.columns[1]
    raise RuntimeError('Could not detect columns')

label_col, text_col = detect_cols(df)

# map labels to binary similarly to training: prefer explicit 'spam' else minority->1
vals = df[label_col].dropna().unique()
lower = [str(v).lower() for v in vals]
if 'spam' in lower:
    df['y'] = df[label_col].map(lambda v: 1 if str(v).lower()=='spam' else 0)
elif set(lower) <= {'0','1'}:
    df['y'] = df[label_col].astype(int)
else:
    counts = df[label_col].value_counts()
    if len(counts) >= 2:
        minority = counts.idxmin()
        df['y'] = df[label_col].map(lambda v: 1 if v==minority else 0)
    else:
        df['y'] = df[label_col].map(lambda v: 1 if str(v).strip() not in ['', '0', 'false', 'none'] else 0)

# compute probs
texts = df[text_col].astype(str).tolist()
X = vec.transform(texts)
if hasattr(model, 'predict_proba'):
    probs_all = model.predict_proba(X)
    # find index of positive class if possible
    try:
        pos_idx = list(model.classes_).index(1)
    except Exception:
        pos_idx = 1 if probs_all.shape[1] > 1 else 0
    probs = probs_all[:, pos_idx]
elif hasattr(model, 'decision_function'):
    scores = model.decision_function(X)
    probs = 1/(1+np.exp(-scores))
else:
    preds = model.predict(X)
    probs = preds.astype(float)

# find spam rows (y==1)
spam_mask = df['y']==1
spam_probs = probs[spam_mask.values]

if len(spam_probs)==0:
    print('No spam rows detected in dataset after mapping - cannot verify high-confidence spam example')
    sys.exit(4)

max_prob = float(spam_probs.max())
print(f"Found {int(spam_mask.sum())} spam rows. Highest predicted spam probability among them: {max_prob:.4f}")

if max_prob >= 0.9:
    # print one example
    idxs = np.where(spam_mask.values)[0]
    best_idx = idxs[int(np.argmax(spam_probs))]
    print("Example text with prob >= 0.9:\n")
    print(df.iloc[best_idx][text_col])
    sys.exit(0)
else:
    print("No spam example with prob >= 0.9 found. Consider further tuning (char n-grams, stronger sampling).")
    # show top-5 spam candidates
    idxs = np.where(spam_mask.values)[0]
    order = np.argsort(-spam_probs)[:5]
    for o in order:
        i = idxs[int(o)]
        print(f"p={spam_probs[int(o)]:.4f} --> {df.iloc[i][text_col][:200]!s}")
    sys.exit(5)
