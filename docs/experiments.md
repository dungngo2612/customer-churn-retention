# ML experiments

Full experiment with the configured trial and time limits.

The team's primary model is logistic regression. Candidates are ranked by mean train CV AP; these results do not automatically change that decision.

Limits: max_trials=12, time_budget_seconds=300, seed=42, threshold=0.5 (churn when probability >= 0.5), folds=5.

Trials: trials_completed=12, trials_planned=12. Completed counts candidates with all CV folds finished, even if val is skipped.

Thời gian đọc CSV và MLflow không nằm trong ngân sách thời gian.

All candidates use identical shuffled stratified folds. Each fold uses a fresh pipeline with preprocessing and class weights fitted only on its training rows. The grid clones the existing CatBoost pipeline and sets l2_leaf_reg before fitting. Defaults come from configs/train_config.yaml.

The top three completed CV candidates are refitted on all available train rows and evaluated once on val, if budget remains. In smoke mode, train means the first 500 rows. No test dataset is read. Std uses ddof=0.

| Candidate | Configuration | CV AP mean±std | CV ROC-AUC mean±std | Brier mean±std | Log loss mean±std | Val AP | Time |
| --- | --- | --- | --- | --- | --- | --- | --- |
| catboost-grid-d4-lr0.1-l210 | `{"class_weight": false, "depth": 4, "iterations": 200, "l2_leaf_reg": 10, "learning_rate": 0.1}` | 0.6733±0.0284 | 0.8478±0.0154 | 0.1341±0.0064 | 0.4128±0.0177 | 0.6597 | 18.05s; fit/fold 0.9522±0.0704s |
| catboost-grid-d6-lr0.03-l23 | `{"class_weight": false, "depth": 6, "iterations": 200, "l2_leaf_reg": 3, "learning_rate": 0.03}` | 0.6720±0.0308 | 0.8477±0.0172 | 0.1340±0.0069 | 0.4138±0.0187 | 0.6558 | 21.23s; fit/fold 1.4638±0.1045s |
| catboost_default-balanced | `{"class_weight": true, "depth": 6, "iterations": 200, "learning_rate": 0.05}` | 0.6706±0.0310 | 0.8478±0.0159 | 0.1607±0.0095 | 0.4788±0.0238 | 0.6510 | 21.64s; fit/fold 1.6843±0.1850s |
| catboost-grid-d6-lr0.03-l210 | `{"class_weight": false, "depth": 6, "iterations": 200, "l2_leaf_reg": 10, "learning_rate": 0.03}` | 0.6703±0.0301 | 0.8479±0.0177 | 0.1340±0.0069 | 0.4136±0.0186 | val: không thuộc top 3 | 12.47s; fit/fold 1.4062±0.0347s |
| catboost-grid-d4-lr0.03-l23 | `{"class_weight": false, "depth": 4, "iterations": 200, "l2_leaf_reg": 3, "learning_rate": 0.03}` | 0.6702±0.0299 | 0.8482±0.0173 | 0.1339±0.0068 | 0.4134±0.0185 | val: không thuộc top 3 | 12.03s; fit/fold 0.9840±0.1261s |
| catboost-grid-d4-lr0.1-l23 | `{"class_weight": false, "depth": 4, "iterations": 200, "l2_leaf_reg": 3, "learning_rate": 0.1}` | 0.6694±0.0219 | 0.8465±0.0143 | 0.1347±0.0059 | 0.4143±0.0171 | val: không thuộc top 3 | 10.53s; fit/fold 0.9190±0.0553s |
| catboost-grid-d4-lr0.03-l210 | `{"class_weight": false, "depth": 4, "iterations": 200, "l2_leaf_reg": 10, "learning_rate": 0.03}` | 0.6688±0.0296 | 0.8475±0.0174 | 0.1343±0.0068 | 0.4141±0.0181 | val: không thuộc top 3 | 11.39s; fit/fold 0.8706±0.0310s |
| catboost_default-unweighted | `{"class_weight": false, "depth": 6, "iterations": 200, "learning_rate": 0.05}` | 0.6677±0.0272 | 0.8463±0.0161 | 0.1345±0.0068 | 0.4146±0.0184 | val: không thuộc top 3 | 15.05s; fit/fold 1.6956±0.1299s |
| catboost-grid-d6-lr0.1-l210 | `{"class_weight": false, "depth": 6, "iterations": 200, "l2_leaf_reg": 10, "learning_rate": 0.1}` | 0.6652±0.0261 | 0.8449±0.0145 | 0.1354±0.0065 | 0.4165±0.0180 | val: không thuộc top 3 | 15.83s; fit/fold 1.7549±0.2790s |
| catboost-grid-d6-lr0.1-l23 | `{"class_weight": false, "depth": 6, "iterations": 200, "l2_leaf_reg": 3, "learning_rate": 0.1}` | 0.6592±0.0271 | 0.8430±0.0157 | 0.1367±0.0070 | 0.4195±0.0197 | val: không thuộc top 3 | 14.01s; fit/fold 1.5481±0.0430s |
| logistic-unweighted | `{"C": 1.0, "class_weight": false, "max_iter": 1000}` | 0.6563±0.0311 | 0.8429±0.0158 | 0.1364±0.0067 | 0.4202±0.0173 | val: không thuộc top 3 | 7.79s; fit/fold 0.1027±0.0108s |
| logistic-balanced | `{"C": 1.0, "class_weight": true, "max_iter": 1000}` | 0.6551±0.0311 | 0.8428±0.0161 | 0.1667±0.0090 | 0.4941±0.0219 | val: không thuộc top 3 | 7.64s; fit/fold 0.1614±0.0714s |

Stop reason: configured_trials_completed; elapsed: 171.94s. The shared worker budget terminates active workers when exhausted. Elapsed wall time also includes CSV, MLflow and setup overhead. Trial time includes worker startup; fit/fold measures training only.

MLflow records CV mean/std for AP, ROC-AUC, Brier, log loss, precision, recall, F1@0.5 and fit time; shortlisted runs also record val metrics. Each run also records trials_completed, trials_planned and val_status. A dash means CV did not finish; skipped val includes its reason.

- `logistic-unweighted`: run `60a469681bc04a649dbdb9024cf145a6`, completed; val: không thuộc top 3.
- `logistic-balanced`: run `28d6719d2b7f45f1b9ea2a8e1c60e4a7`, completed; val: không thuộc top 3.
- `catboost_default-unweighted`: run `2c6464f7b8854b85a1ab5d34ed20eeec`, completed; val: không thuộc top 3.
- `catboost_default-balanced`: run `6c1e694205c64fe98f87a4ad026f77dd`, completed; val: đã đánh giá.
- `catboost-grid-d4-lr0.03-l23`: run `648b4681233443c2b0faff6c906e924c`, completed; val: không thuộc top 3.
- `catboost-grid-d4-lr0.03-l210`: run `21b8e55f14104a949f90d4a303d5a3e1`, completed; val: không thuộc top 3.
- `catboost-grid-d4-lr0.1-l23`: run `aabc222082904f918f71172ec4abf809`, completed; val: không thuộc top 3.
- `catboost-grid-d4-lr0.1-l210`: run `986fdb7f9a674748844ffa374d820441`, completed; val: đã đánh giá.
- `catboost-grid-d6-lr0.03-l23`: run `f61e99960b1045f68424ef5cf13ca69e`, completed; val: đã đánh giá.
- `catboost-grid-d6-lr0.03-l210`: run `5b92e1c78bbf495196aa199aedf09a00`, completed; val: không thuộc top 3.
- `catboost-grid-d6-lr0.1-l23`: run `2d7ae8b7a3b44282b746c8d7a5282269`, completed; val: không thuộc top 3.
- `catboost-grid-d6-lr0.1-l210`: run `ae95e48b00024b32b1492677f4b3cdd7`, completed; val: không thuộc top 3.

Run the full experiment from the repository root:

```sh
PYTHONPATH=src python3 -m churn_retention.entrypoints.cli tune --config configs/training.yaml
```

Append `--smoke` for a quick check. The command replaces this report; MLflow retains previous runs. Both CSV paths must already point to prepared train/val files; this command does not create dataset splits.
