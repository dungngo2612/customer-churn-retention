# M2.2: Logistic calibration

Base: `origin/feat/ml-experiments@1a3b674`. Logistic uses the existing M2.1
preprocessing and defaults, with `class_weight=None`, seed 42 and the same prepared
train/validation split. Test data is never opened.

Sigmoid (Platt) and isotonic use five shuffled stratified folds on train only.
`CalibratedClassifierCV(ensemble=False)` learns calibration from out-of-fold train
scores and refits the complete preprocessing/model pipeline on all train rows.
Validation is reserved for evaluation and threshold proposals.

Apply calibration only if its validation Brier score decreases by at least 0.005
from the uncalibrated baseline AND ROC-AUC decreases by no more than 0.005.
Comparisons allow 1e-12 numerical tolerance. Among eligible methods choose lowest
Brier, then highest ROC-AUC, then method name; otherwise keep no calibration.
This is a proposal and does not change a saved model or configuration.

The first measured validation results propose **không áp dụng**: sigmoid improves
Brier by only 0.0000096326; isotonic increases Brier. Neither reaches 0.005.

Reliability CSVs in `docs/calibration/` contain ten-bin aggregates only.
The shared evaluation module is absent from both main and this branch's base;
classification metrics reuse `training.metrics.compute_metrics`.
