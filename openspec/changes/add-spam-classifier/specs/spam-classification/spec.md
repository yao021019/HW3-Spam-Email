## ADDED Requirements

### Requirement: Spam classification baseline
The project SHALL provide a reproducible baseline for spam message classification that trains a classifier on the referenced public dataset and reports precision, recall, and F1-score.

#### Scenario: Download and verify dataset
- **WHEN** the dataset URL is downloaded
- **THEN** the script verifies the file is readable, contains two columns (label and message), and >1000 rows

#### Scenario: Train logistic regression baseline
- **WHEN** the baseline training script or notebook is run on the dataset
- **THEN** the system trains a logistic regression model (scikit-learn or similar) with a documented preprocessing pipeline (tokenization, lowercasing, TF-IDF or similar)
- **AND** the training produces a saved model artifact and an evaluation report containing precision, recall, and F1 for the spam class

#### Scenario: Optional SVM comparison
- **WHEN** requested in Phase 1 experiments
- **THEN** an SVM model can be trained with the same preprocessing and evaluation pipeline and the results compared to logistic regression in the report

#### Scenario: Reproducible run
- **WHEN** a contributor follows the README steps (download, preprocess, train)
- **THEN** they can reproduce the baseline evaluation results locally and the steps are documented clearly in the change README
