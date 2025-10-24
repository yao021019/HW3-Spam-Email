## Why

Spam and unwanted messages remain a key nuisance and security vector. This change proposes adding a reproducible machine learning baseline for spam email/sms classification so we can measure improvements and compare models over time.

## What Changes

- Add a new capability `spam-classification` providing a baseline ML pipeline and reproducible evaluation artifacts.
- Phase 1 (baseline): build and evaluate a basic classifier using the public dataset linked below. The baseline model will use logistic regression as the primary algorithm; the original plan mentioned an SVM baseline — implementation may include both so we can compare results.
- Phase 2+: placeholders for additional experiments (hyperparameter tuning, feature engineering, model export, productionization).

## Data Source

Use the public dataset hosted in the Packt repo:

https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/refs/heads/master/Chapter03/datasets/sms_spam_no_header.csv

## Impact

- Affected specs: new capability `spam-classification` (adds requirements and scenarios).
- Affected code: scripts/notebooks under `scripts/` or `notebooks/` (implementation will be in a follow-up change).
- Breaking changes: None.

## Phases

Phase 1 — Baseline (this change focuses on planning/specs):
- Download and validate dataset
- Minimal preprocessing (tokenize, lower-case, remove punctuation; optional TF-IDF)
- Train a logistic regression baseline (scikit-learn) and optionally train an SVM for comparison
- Evaluate using precision/recall/F1 and report baseline metrics
- Provide a reproducible notebook or script that others can run locally

Phase 2 — Future experiments (placeholders):
- Feature engineering & n-grams
- Hyperparameter tuning (grid search / randomized)
- Model serialization & simple scoring API
- CI integration for training and evaluation

## Acceptance Criteria

1. Spec delta under `openspec/changes/add-spam-classifier/specs/spam-classification/spec.md` exists and uses `## ADDED Requirements` with at least one `#### Scenario:` per requirement.
2. Baseline plan includes dataset reference and expected evaluation metrics (precision, recall, F1) and reproducible instructions.
3. Implementation is not required in this change; implementation will be created as a follow-up change after this proposal is reviewed.

---

Notes:
- The user indicated both "logistic regression" and "SVM" in the plan. This proposal sets logistic regression as the primary baseline, and suggests optionally training SVM as a comparator during Phase 1.
