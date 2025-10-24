# Spam classification baseline (HW3)

This project contains an OpenSpec-driven proposal and a minimal reproducible baseline for spam message classification using the public dataset referenced in the proposal.

Quick steps (Windows PowerShell):

1. Create a virtual environment and install dependencies

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Download dataset

```powershell
python scripts/download_dataset.py --out data/sms_spam.csv
```

3. Train baseline (logistic regression)

```powershell
python scripts/train_baseline.py --data data/sms_spam.csv --out models --metrics-out outputs/metrics.json
```

Use `--train-svm` to also train an SVM comparator and `--max-rows` to limit rows during development.

4. Run the Streamlit demo

```powershell
streamlit run app/streamlit_app.py
```

Notes:
- Model artifacts and data are ignored by `.gitignore`. Keep models in `models/` and outputs in `outputs/`.
- This baseline is meant to be small and reproducible; follow-up changes should add tests, CI, and model export conventions.
