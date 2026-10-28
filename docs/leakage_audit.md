# Leakage audit (Task 2.1)

Phạm vi: kiểm trên nhánh pipeline-leakage-review d1c8dd8.

## Scope and evidence

Audit date: 2026-10-28

Reviewed `configs/train_config.yaml`, `data/telco_churn.csv`, `artifacts/splits/dataset_splits.csv`, `notebooks/explore_and_split.py`, and the available dataset preprocessing/validation code under `src/churn_retention/modules/datasets/`. On `pipeline-leakage-review` at `d1c8dd8`, a repository search found no implemented model training pipeline or preprocessing transformer with `fit`/`fit_transform` calls; the `ml/training` and `ml/preprocessing` packages at that revision contain only package initializers. These statements do not describe main; runtime evidence for main `e26c373` is reported separately below.

## Feature and label inventory

| Role | Columns |
| --- | --- |
| Identifier (excluded by training config) | `customerID` |
| Target | `Churn` (positive label: `Yes`) |
| Numeric features | `SeniorCitizen`, `tenure`, `MonthlyCharges`, `TotalCharges` |
| Categorical features | `gender`, `Partner`, `Dependents`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod` |

The source CSV has 7,043 rows and 21 columns. All source columns are accounted for exactly once as ID, target, or configured features. `customerID` is explicitly listed under `drop_cols`, and `Churn` is the configured target rather than a feature. The available `clean_and_split_features` function separates `Churn` from features by default. The statement “không có cột hậu churn” is an assumption based on column names and the dataset description in `docs/dataset_profile.md`, not proof of when each feature is available. That description identifies a cross-sectional snapshot without event timestamps; availability at prediction time has not been established for every feature. No explicitly target-derived column is identified in the source/config inventory by this static review. On `pipeline-leakage-review` at `d1c8dd8`, the absent training pipeline prevents confirmation of the columns actually passed to a fitted estimator; the separate main probe below measures the Logistic inputs and outputs.

## Split manifest integrity

Both hashes are SHA-256 of the file bytes **normalized to LF**: replace CRLF
(`\r\n`) with LF (`\n`), then any remaining CR (`\r`) with LF. Preserve all
other bytes, including whitespace and the presence or absence of a final newline;
do not parse or reserialize the CSV. The previous hashes depended on CRLF line
endings; the LF values below were recomputed from the actual files.

Source SHA-256 (LF): `16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91`

Split manifest SHA-256 (LF): `bced80937bf69798aa314d03bd35d59d6c14134a593997d3e3dd8c7fe8b6512c`

Reproduce from the repository root (read-only; independent of checkout line endings):

```sh
python3 - <<'PY'
from hashlib import sha256
from pathlib import Path

for name in ("data/telco_churn.csv", "artifacts/splits/dataset_splits.csv"):
    content = Path(name).read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    print(f"{sha256(content).hexdigest()}  {name}")
PY
```

| Split | Rows | Churn cases | Churn rate |
| --- | ---: | ---: | ---: |
| train | 4,930 | 1,308 | 26.531% |
| val | 1,056 | 280 | 26.515% |
| test | 1,057 | 281 | 26.585% |

The manifest has one row per customer, no missing IDs/split labels, no duplicate IDs, and its ID set exactly matches the source CSV. Therefore there is no customer overlap across train, validation, and test in this manifest. The split was created with stratified `train_test_split` calls using `random_state=42` in `notebooks/explore_and_split.py`.

## Preprocessing and fit leakage

The available upload cleaning code strips text whitespace, converts numeric fields and applies a deterministic `TotalCharges` rule: blank charges for `tenure == 0` become zero. This operation does not estimate parameters from a dataset and therefore has no cross-split fit leakage by itself. The code search found no `fit`, `fit_transform`, resampling, or SMOTE implementation in the source/notebook code of `pipeline-leakage-review` at `d1c8dd8`. Since the training implementation is absent on that revision, it is not possible to verify that future imputers, encoders, scalers, feature selection, or resampling are fit on train only (or applied only after splitting).

## Kiểm tra trên main (e26c373)

These are **measured results**, run on 2026-10-28 against a temporary archive of
`e26c373898afb7b7e08dab74e8c26c219518723c`, separate from the branch audited above.
The probe executes the real Logistic trainer, workflow feature preparation and
`evaluate_cv` with the first unweighted Logistic candidate; it does not run the
whole CLI, CatBoost grid, feature selection, calibration or deployment.

| Executed check | Measured result |
| --- | --- |
| Prepare train/val from the source CSV and manifest | 4,930 train rows; 1,056 val rows; no shared customer IDs; blank `TotalCharges` filled with 0.0 for 7 train and 2 val rows |
| Observe `fit` on the two imputers, scaler, encoder and Logistic estimator | Every fit received 4,930 train rows; imputer input customer IDs exactly matched train |
| Predict probabilities on val | 0 additional fits; fitted preprocessing reused |
| Five-fold stratified CV, seed 42 | Every fold fit 3,944 rows; both imputers received exactly that fold's train IDs |
| CSV access inside `evaluate_cv` | 0 calls to `pd.read_csv`; the probe rejects any such call and source inspection finds no other CSV reader in this CV path |
| Logistic feature input/output | 19 configured input features → 45 transformed columns; neither `Churn` nor `customerID` appears in either set |

The runtime evidence covers these exact data, configuration and executed paths.
It does not prove that arbitrary train/val paths are disjoint or establish when
each feature becomes available. This is validation-boundary evidence, not a final
test-set performance claim.

Reproduce from the repository root in an environment with the main revision's
training dependencies installed. The shell archives main and removes only its
own temporary directory; the probe is inline and is not a tracked script:

```sh
bash <<'SH'
set -euo pipefail
audit_tmp=$(mktemp -d /tmp/leakage-audit-main.XXXXXX)
trap 'rm -rf "$audit_tmp"' EXIT
git archive e26c373898afb7b7e08dab74e8c26c219518723c | tar -x -C "$audit_tmp"
cd "$audit_tmp"
python3 scripts/make_splits.py --raw-churn-path data/telco_churn.csv \
  --splits-path artifacts/splits/dataset_splits.csv
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src MLFLOW_DISABLE_AGENT_HINT=1 python3 - <<'PY'
from collections import Counter
from contextlib import ExitStack
from functools import wraps
from unittest.mock import patch

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn_retention.ml.training.baseline import train_logistic
from churn_retention.ml.training.tune import _prepare, evaluate_cv, load_tune_config

config = load_tune_config("configs/training.yaml")
assert config.seed == 42 and config.n_splits == 5
train = pd.read_csv(config.train_path).set_index("customerID", drop=False)
val = pd.read_csv(config.val_path).set_index("customerID", drop=False)
assert len(train) == 4930 and len(val) == 1056
assert set(train.index).isdisjoint(val.index)
X, y = _prepare(train, config.training)
X_val, _ = _prepare(val, config.training, train["Churn"].unique().tolist())
events = []


def observe(cls):
    original = cls.fit

    @wraps(original)
    def fit(self, values, *args, **kwargs):
        ids = frozenset(values.index) if isinstance(values, pd.DataFrame) else None
        events.append((cls.__name__, len(values), ids))
        return original(self, values, *args, **kwargs)

    return fit


classes = (SimpleImputer, StandardScaler, OneHotEncoder, LogisticRegression)
with ExitStack() as stack:
    for cls in classes:
        stack.enter_context(patch.object(cls, "fit", observe(cls)))
    features = config.training["features"]
    model = train_logistic(X, y, {
        "numeric_features": features["numeric_cols"],
        "categorical_features": features["categorical_cols"],
        "logistic": config.training["models"]["logistic"],
    }, config.seed, False)
    expected_counts = {
        "SimpleImputer": 2, "StandardScaler": 1,
        "OneHotEncoder": 1, "LogisticRegression": 1,
    }
    assert Counter(name for name, _, _ in events) == Counter(expected_counts)
    assert all(rows == 4930 for _, rows, _ in events)
    assert all(ids == frozenset(train.index) for _, _, ids in events if ids is not None)
    before_predict = len(events)
    model.predict_proba(X_val)
    assert len(events) == before_predict
    print("Train fit rows: 4930 (2 imputers, scaler, encoder, logistic); val refits: 0")

    transformed = model.named_steps["preprocessor"].get_feature_names_out()
    assert X.shape[1] == 19 and len(transformed) == 45
    assert {"Churn", "customerID"}.isdisjoint(X.columns)
    assert not any("Churn" in name or "customerID" in name for name in transformed)
    print("Logistic features: 19 -> 45; Churn/customerID: absent")

    events.clear()
    with patch.object(pd, "read_csv", side_effect=AssertionError("CV read CSV")) as reader:
        evaluate_cv(X, y, config.candidates[0], config)
        assert reader.call_count == 0
    assert Counter(name for name, _, _ in events) == Counter({
        name: count * 5 for name, count in expected_counts.items()
    })
    assert all(rows == 3944 for _, rows, _ in events)
    expected_ids = [
        frozenset(X.iloc[fit_ids].index)
        for fit_ids, _ in StratifiedKFold(
            n_splits=5, shuffle=True, random_state=42
        ).split(X, y)
    ]
    imputer_ids = [ids for name, _, ids in events if name == "SimpleImputer"]
    assert Counter(imputer_ids) == Counter(expected_ids * 2)
    print("CV: 5 folds, 3944 fit rows/fold; exact fold IDs: match; CSV reads: 0")
PY
SH
```

Expected probe output (after split preparation):

```text
Train fit rows: 4930 (2 imputers, scaler, encoder, logistic); val refits: 0
Logistic features: 19 -> 45; Churn/customerID: absent
CV: 5 folds, 3944 fit rows/fold; exact fold IDs: match; CSV reads: 0
```

## Findings and follow-up

- **Pass (static configuration):** the configured feature list matches the source columns and excludes both `customerID` and `Churn`.
- **Pass (manifest):** customer IDs are complete, unique, and partitioned exactly once; class proportions are balanced across splits.
- **Pass (available deterministic cleaning):** no data-fitted transformation is present in the reviewed cleaning path.
- **Unverified on `pipeline-leakage-review` (`d1c8dd8`):** training-time preprocessing and resampling boundaries, because that revision does not implement training. The main probe separately confirms the measured Logistic/CV fit boundaries; unexecuted paths remain outside its scope.
- **Assumption, not a temporal guarantee:** feature names and descriptions do not establish the availability time of every feature.
- **Evidence limitation:** checksums identify the exact files audited; they should be recomputed if either input file changes.

When training code is added to `pipeline-leakage-review` or changes on main, repeat the `fit`/`fit_transform`/resampling audit and verify the training entrypoint consumes the manifest rather than creating a new split or using the test set for tuning.
