import pandas as pd

# CatBoost and scikit-learn ship neither stubs nor py.typed markers for these imports.
from catboost import CatBoostClassifier  # type: ignore[import-untyped]
from sklearn.base import BaseEstimator, TransformerMixin  # type: ignore[import-untyped]
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]


class CatBoostPreprocessor(BaseEstimator, TransformerMixin):
    """
    Preprocessor for CatBoost: handle missing values in categorical features.
    Numeric features are left as-is (CatBoost handles NaN natively).
    """

    def __init__(
        self, categorical_features: list[str], feature_cols: list[str]
    ) -> None:
        self.categorical_features = categorical_features
        self.feature_cols = feature_cols

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # R08: Select and order columns according to config
        X_selected = X[self.feature_cols].copy()

        # Fill missing values in categorical columns
        for col in self.categorical_features:
            if col in X_selected.columns:
                X_selected[col] = X_selected[col].astype(str).fillna("__MISSING__")

        return X_selected


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

    # R08: Select and order columns according to config
    feature_cols = numeric_features + categorical_features

    # Build preprocessor (now selects columns in transform)
    preprocessor = CatBoostPreprocessor(
        categorical_features=categorical_features, feature_cols=feature_cols
    )

    # R08: Use column names (not indices) for cat_features
    model = CatBoostClassifier(
        iterations=cfg["catboost"]["iterations"],
        depth=cfg["catboost"]["depth"],
        learning_rate=cfg["catboost"]["learning_rate"],
        auto_class_weights="Balanced" if class_weight else None,
        cat_features=categorical_features,
        random_state=seed,
        verbose=False,
        allow_writing_files=False,
    )

    # R15: Build and fit pipeline (fit once, not fit_transform then fit)
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    pipeline.fit(X_train, y_train)

    return pipeline
