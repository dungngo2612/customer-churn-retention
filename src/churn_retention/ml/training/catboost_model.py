import pandas as pd
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


class CatBoostPreprocessor(BaseEstimator, TransformerMixin):
    """
    Preprocessor for CatBoost: handle missing values in categorical features.
    Numeric features are left as-is (CatBoost handles NaN natively).
    """

    def __init__(self, categorical_features: list[str]) -> None:
        self.categorical_features = categorical_features

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_copy = X.copy()

        # Fill missing values in categorical columns
        for col in self.categorical_features:
            if col in X_copy.columns:
                X_copy[col] = X_copy[col].astype(str).fillna("__MISSING__")

        return X_copy


def train_catboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cfg: dict,
    seed: int,
    class_weight: bool,
) -> Pipeline:
    """
    Train CatBoost classifier with preprocessing pipeline.

    Args:
        X_train: Training features.
        y_train: Training labels (0 or 1, where 1 = churn).
        cfg: Configuration dict with keys:
            - numeric_features: list of numeric column names
            - categorical_features: list of categorical column names
            - catboost: dict with iterations, depth, learning_rate
        seed: Random seed for reproducibility.
        class_weight: If True, use "Balanced"; if False, use None.

    Returns:
        Fitted Pipeline with preprocessor and CatBoostClassifier.

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
    preprocessor = CatBoostPreprocessor(categorical_features=categorical_features)

    # Build model
    cat_features_indices = [
        i
        for i, col in enumerate(numeric_features + categorical_features)
        if col in categorical_features
    ]

    model = CatBoostClassifier(
        iterations=cfg["catboost"]["iterations"],
        depth=cfg["catboost"]["depth"],
        learning_rate=cfg["catboost"]["learning_rate"],
        auto_class_weights="Balanced" if class_weight else None,
        cat_features=cat_features_indices,
        random_state=seed,
        verbose=False,
        allow_writing_files=False,
    )

    # Build and fit pipeline
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    pipeline.fit(X_train_preprocessed, y_train)

    return pipeline
