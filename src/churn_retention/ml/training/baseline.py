import pandas as pd

# scikit-learn ships neither stubs nor a py.typed marker for these imports.
from sklearn.compose import ColumnTransformer  # type: ignore[import-untyped]
from sklearn.impute import SimpleImputer  # type: ignore[import-untyped]
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # type: ignore[import-untyped]


def train_logistic(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cfg: dict,
    seed: int,
    class_weight: bool,
) -> Pipeline:
    """
    Train logistic regression model with preprocessing pipeline.

    Args:
        X_train: Training features.
        y_train: Training labels (0 or 1, where 1 = churn).
        cfg: Configuration dict with keys:
            - numeric_features: list of numeric column names
            - categorical_features: list of categorical column names
            - logistic: dict with C and max_iter
        seed: Random seed for reproducibility.
        class_weight: If True, use "balanced"; if False, use None.

    Returns:
        Fitted Pipeline with preprocessor and LogisticRegression.

    Raises:
        ValueError: If feature columns are missing from X_train.
    """
    numeric_features = cfg["numeric_features"]
    categorical_features = cfg["categorical_features"]

    # Validate columns exist
    missing_cols = set(numeric_features + categorical_features) - set(X_train.columns)
    if missing_cols:
        raise ValueError(f"Missing columns in X_train: {missing_cols}")

    # Build preprocessor
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # Build model
    model = LogisticRegression(
        C=cfg["logistic"]["C"],
        max_iter=cfg["logistic"]["max_iter"],
        class_weight="balanced" if class_weight else None,
        random_state=seed,
    )

    # Build and fit pipeline
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    pipeline.fit(X_train, y_train)

    return pipeline
