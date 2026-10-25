# ML experiments

Smoke only: 2 folds, first 500 train rows, at most 2 candidates. The full model comparison has not been run.

The team's primary model is logistic regression. Candidates are ranked by mean train CV AP; these results do not automatically change that decision.

Limits: max_trials=2, time_budget_seconds=300, seed=42, threshold=0.5 (churn when probability >= 0.5), folds=2.

All candidates use identical shuffled stratified folds. Each fold uses a fresh pipeline with preprocessing and class weights fitted only on its training rows. The grid clones the existing CatBoost pipeline and sets l2_leaf_reg before fitting. Defaults come from configs/train_config.yaml.

The top three completed CV candidates are refitted on all available train rows and evaluated once on val, if budget remains. In smoke mode, train means the first 500 rows. No test dataset is read. Std uses ddof=0.

| Candidate | Configuration | CV AP mean±std | CV ROC-AUC mean±std | Brier mean±std | Log loss mean±std | Val AP | Time |
| --- | --- | --- | --- | --- | --- | --- | --- |
| logistic-balanced | `{"C": 1.0, "class_weight": true, "max_iter": 1000}` | 0.5171±0.0156 | 0.7799±0.0138 | 0.1838±0.0065 | 0.5658±0.0451 | 0.6155 | 12.13s; fit/fold 0.0539±0.0073s |
| logistic-unweighted | `{"C": 1.0, "class_weight": false, "max_iter": 1000}` | 0.5146±0.0205 | 0.7802±0.0173 | 0.1548±0.0033 | 0.4958±0.0368 | 0.6270 | 13.14s; fit/fold 0.0960±0.0504s |

Stop reason: configured_trials_completed; elapsed: 27.87s. The shared computation deadline terminates active workers. Tracking and report finalization may finish after that deadline. Time includes worker startup; fit/fold measures training only.

MLflow records CV mean/std for AP, ROC-AUC, Brier, log loss, precision, recall, F1@0.5 and fit time; shortlisted runs also record val metrics. A dash means validation was not performed or CV did not finish.

- `logistic-unweighted`: run `6b21f89aeec044dcbf62786919cd3bd8`, completed.
- `logistic-balanced`: run `236fab11edc74c6f86eda41232442e41`, completed.

Run the full experiment from the repository root:

```sh
PYTHONPATH=src python3 -m churn_retention.entrypoints.cli tune --config configs/training.yaml
```

Append `--smoke` for a quick check. The command replaces this report; MLflow retains previous runs. Both CSV paths must already point to prepared train/val files; this command does not create dataset splits.
