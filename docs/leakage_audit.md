# Leakage audit (Task 2.1)

## Scope and evidence

Audit date: 2026-10-28

Reviewed `configs/train_config.yaml`, `data/telco_churn.csv`, `artifacts/splits/dataset_splits.csv`, `notebooks/explore_and_split.py`, and the available dataset preprocessing/validation code under `src/churn_retention/modules/datasets/`. A repository search found no implemented model training pipeline or preprocessing transformer with `fit`/`fit_transform` calls; the `ml/training` and `ml/preprocessing` packages currently contain only package initializers.

## Feature and label inventory

| Role | Columns |
| --- | --- |
| Identifier (excluded by training config) | `customerID` |
| Target | `Churn` (positive label: `Yes`) |
| Numeric features | `SeniorCitizen`, `tenure`, `MonthlyCharges`, `TotalCharges` |
| Categorical features | `gender`, `Partner`, `Dependents`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod` |

The source CSV has 7,043 rows and 21 columns. All source columns are accounted for exactly once as ID, target, or configured features. `customerID` is explicitly listed under `drop_cols`, and `Churn` is the configured target rather than a feature. The available `clean_and_split_features` function separates `Churn` from features by default. No post-churn-only or target-derived column appears in the source/config inventory. This is a static review; the absent training pipeline prevents confirmation of the columns actually passed to a fitted estimator.

## Split manifest integrity

Source SHA-256: `88be4b93fbe0cc83421af1c503794c97c342eca914c1576db7c276e61d61358a`

Split manifest SHA-256: `6af410beaa2f3137e46087255cdac0f80595a0834b36a66643c41882ed246032`

| Split | Rows | Churn cases | Churn rate |
| --- | ---: | ---: | ---: |
| train | 4,930 | 1,308 | 26.531% |
| val | 1,056 | 280 | 26.515% |
| test | 1,057 | 281 | 26.585% |

The manifest has one row per customer, no missing IDs/split labels, no duplicate IDs, and its ID set exactly matches the source CSV. Therefore there is no customer overlap across train, validation, and test in this manifest. The split was created with stratified `train_test_split` calls using `random_state=42` in `notebooks/explore_and_split.py`.

## Preprocessing and fit leakage

The available upload cleaning code strips text whitespace, converts numeric fields and applies a deterministic `TotalCharges` rule: blank charges for `tenure == 0` become zero. This operation does not estimate parameters from a dataset and therefore has no cross-split fit leakage by itself. The code search found no `fit`, `fit_transform`, resampling, or SMOTE implementation in the repository's current source/notebook code. Since the training implementation is absent, it is not possible to verify that future imputers, encoders, scalers, feature selection, or resampling are fit on train only (or applied only after splitting).

## Findings and follow-up

- **Pass (static configuration):** the configured feature list matches the source columns and excludes both `customerID` and `Churn`.
- **Pass (manifest):** customer IDs are complete, unique, and partitioned exactly once; class proportions are balanced across splits.
- **Pass (available deterministic cleaning):** no data-fitted transformation is present in the reviewed cleaning path.
- **Unverified:** training-time preprocessing and resampling boundaries, because the training code is not implemented/present.
- **Evidence limitation:** checksums identify the exact files audited; they should be recomputed if either input file changes.

When training code is added, repeat the `fit`/`fit_transform`/resampling audit and verify the training entrypoint consumes the manifest rather than creating a new split or using the test set for tuning.
