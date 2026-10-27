# M2.2: Logistic calibration

Base: `origin/feat/ml-experiments@1a3b674` (main did not contain M2.1).
Logistic, class_weight=None, C=1.0, max_iter=1000, seed=42, folds=5.
Existing M2.1 split: train=4930, val=1056; only data/processed/train.csv and val.csv are opened. Test data is never read.

## Train-only calibration

The baseline reuses the existing logistic preprocessing pipeline. Sigmoid (Platt) and isotonic wrap the complete pipeline in CalibratedClassifierCV with shuffled StratifiedKFold and ensemble=False. Each preprocessing fit sees only its training fold. Calibration learns from out-of-fold train scores; the final base pipeline is refitted on all train rows. Validation is never supplied to fit, including calibrator fit.

## Validation results

| State | Brier | Log loss | ROC-AUC | PR-AUC (average precision) | ECE |
| --- | --- | --- | --- | --- | --- |
| none | 0.1366473120 | 0.4186778354 | 0.8454321613 | 0.6309479730 | 0.0243680897 |
| sigmoid | 0.1366376794 | 0.4186707369 | 0.8454321613 | 0.6309479730 | 0.0221565481 |
| isotonic | 0.1370516608 | 0.4216341961 | 0.8444955817 | 0.6206940059 | 0.0241541895 |

ECE = sum_b (n_b/N) * abs(mean_probability_b - churn_rate_b), with ten equal-width bins [i/10, (i+1)/10); the final bin includes probability 1. This is positive-class probability ECE. Empty bins contribute zero; their undefined means/rates are empty CSV cells.

Reliability data (bin, bounds, count, mean probability, observed churn): [none](calibration/reliability_none.csv), [sigmoid](calibration/reliability_sigmoid.csv), [isotonic](calibration/reliability_isotonic.csv). These files contain aggregates only, without IDs or individual predictions.

## Calibration proposal

Apply calibration only if validation Brier decreases by at least 0.005 AND ROC-AUC decreases by no more than 0.005 versus none. Inclusive comparisons allow 1e-12 numerical tolerance. Among eligible methods choose lowest Brier, then highest ROC-AUC, then method name; otherwise retain none. This is a proposal; no saved model is replaced.

Đề xuất: không áp dụng hiệu chỉnh; giữ logistic không class_weight.

## Validation threshold proposal

Scan: 0.05 through 0.95 inclusive, step 0.01; churn when proba >= threshold. Selected state: none. Max-F1 ties choose the lower threshold. Display every 0.05 increment plus max F1; [all 91 thresholds](calibration/thresholds.csv) are retained in CSV.

Temporary default only: threshold=0.27, precision=0.5112107623, recall=0.8142857143, F1=0.6280991736. This threshold is NOT locked and is NOT written to config.

| Threshold | Precision | Recall | F1 | TP | FP | FN | TN |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.05 | 0.351662 | 0.982143 | 0.517891 | 275 | 507 | 5 | 269 |
| 0.10 | 0.398801 | 0.950000 | 0.561774 | 266 | 401 | 14 | 375 |
| 0.15 | 0.439523 | 0.921429 | 0.595156 | 258 | 329 | 22 | 447 |
| 0.20 | 0.472062 | 0.875000 | 0.613267 | 245 | 274 | 35 | 502 |
| 0.25 | 0.495763 | 0.835714 | 0.622340 | 234 | 238 | 46 | 538 |
| 0.27 | 0.511211 | 0.814286 | 0.628099 | 228 | 218 | 52 | 558 |
| 0.30 | 0.518779 | 0.789286 | 0.626062 | 221 | 205 | 59 | 571 |
| 0.35 | 0.533163 | 0.746429 | 0.622024 | 209 | 183 | 71 | 593 |
| 0.40 | 0.557746 | 0.707143 | 0.623622 | 198 | 157 | 82 | 619 |
| 0.45 | 0.601974 | 0.653571 | 0.626712 | 183 | 121 | 97 | 655 |
| 0.50 | 0.645914 | 0.592857 | 0.618250 | 166 | 91 | 114 | 685 |
| 0.55 | 0.683962 | 0.517857 | 0.589431 | 145 | 67 | 135 | 709 |
| 0.60 | 0.723270 | 0.410714 | 0.523918 | 115 | 44 | 165 | 732 |
| 0.65 | 0.747899 | 0.317857 | 0.446115 | 89 | 30 | 191 | 746 |
| 0.70 | 0.733333 | 0.196429 | 0.309859 | 55 | 20 | 225 | 756 |
| 0.75 | 0.714286 | 0.071429 | 0.129870 | 20 | 8 | 260 | 768 |
| 0.80 | 0.666667 | 0.014286 | 0.027972 | 4 | 2 | 276 | 774 |
| 0.85 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 280 | 776 |
| 0.90 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 280 | 776 |
| 0.95 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 280 | 776 |

The shared evaluation module is absent from both main and the selected base. Metrics reuse training.metrics.compute_metrics; confusion counts and reliability/ECE supplement it locally.

Reproduce from the repository root:

```sh
python3 -m churn_retention.entrypoints.cli calibrate --config configs/training.yaml
```

The command replaces only this report and aggregate CSVs. It does not write MLflow or modify docs/experiments.md. Repeated runs with the same inputs and library versions produce identical docs and CSV bytes.
