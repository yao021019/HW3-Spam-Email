## 1. Proposal & planning
- [x] Create `proposal.md` describing goals, data source, and phases
- [ ] Create `specs/spam-classification/spec.md` delta (this file)

## 2. Phase 1 - Baseline (implementation in follow-up change)
- [ ] Download dataset and add a small verification script to `scripts/verify-dataset.js` or `scripts/verify-dataset.py`
- [ ] Implement preprocessing and baseline training notebook/script (`notebooks/baseline-spam-classification.ipynb` or `scripts/train_baseline.py`)
- [ ] Train logistic regression baseline and record precision/recall/F1
- [ ] Optionally train SVM for comparison
- [ ] Add evaluation outputs and a small README describing reproducible steps

## 3. Testing & CI
- [ ] Add unit tests for preprocessing helpers
- [ ] Add integration test that runs the baseline quickly on a small subset
- [ ] Add CI workflow to run `openspec validate` and tests on PRs (optional follow-up)

## 4. Documentation
- [ ] Add `README.md` in `openspec/changes/add-spam-classifier/` describing how to reproduce Phase 1 results
