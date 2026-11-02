"""Train-only RAG feature transformation, independent of any classifier/bundle."""

import io
import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd
import sklearn  # type: ignore[import-untyped]
from numpy.typing import NDArray
from sklearn.compose import ColumnTransformer  # type: ignore[import-untyped]
from sklearn.impute import SimpleImputer  # type: ignore[import-untyped]
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # type: ignore[import-untyped]

from churn_retention.contracts.reference import (
    FeatureColumn,
    Scalar,
    clean_features,
    json_bytes,
    sha256,
    validate_id,
    validate_schema,
)

NUMERIC_FEATURES = ("SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges")
CATEGORICAL_FEATURES = (
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
)
FEATURE_SCHEMA = tuple(
    FeatureColumn(
        name=name,
        kind="numeric",
        dtype="int64" if name in {"SeniorCitizen", "tenure"} else "float64",
    )
    for name in NUMERIC_FEATURES
) + tuple(
    FeatureColumn(name=name, kind="categorical", dtype="str")
    for name in CATEGORICAL_FEATURES
)


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite": str(value)}
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, type) and issubclass(value, np.generic):
        return np.dtype(value).name
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError("Unsupported preprocessing metadata value.")


def _train_ids(split_bytes: bytes) -> set[str]:
    splits = pd.read_csv(
        io.BytesIO(split_bytes), dtype={"customerID": "string", "split": "string"}
    )
    if list(splits.columns).count("customerID") != 1 or set(splits.columns) != {
        "customerID",
        "split",
    }:
        raise ValueError(
            "Split manifest requires exactly customerID and split columns."
        )
    ids = [validate_id(value) for value in splits.customerID]
    if (
        len(set(ids)) != len(ids)
        or not splits.split.isin(["train", "val", "test"]).all()
    ):
        raise ValueError("Split membership must be unique and valid.")
    train_ids = set(splits.loc[splits.split.eq("train"), "customerID"])
    if not train_ids:
        raise ValueError("Split manifest contains no train IDs.")
    return train_ids


def _describe(
    transformer: Any, categories: Mapping[str, tuple[str, ...]]
) -> dict[str, Any]:
    steps = {}
    for group in ("num", "cat"):
        for name, estimator in transformer.named_transformers_[group].steps:
            state = {
                attribute: getattr(estimator, attribute)
                for attribute in (
                    "statistics_",
                    "mean_",
                    "var_",
                    "scale_",
                    "n_samples_seen_",
                    "categories_",
                    "drop_idx_",
                    "n_features_in_",
                )
                if hasattr(estimator, attribute)
            }
            steps[f"{group}.{name}"] = {
                "class": type(estimator).__name__,
                "parameters": estimator.get_params(deep=False),
                "state": state,
            }
    return _json_value(
        {
            "schema": [column.model_dump() for column in FEATURE_SCHEMA],
            "input_feature_names": transformer.feature_names_in_,
            "output_feature_names": transformer.get_feature_names_out(),
            "categorical_vocabulary": categories,
            "parameters": {
                key: value
                for key, value in transformer.get_params(deep=False).items()
                if key != "transformers"
            },
            "transformer_order": [
                {"name": name, "columns": list(columns)}
                for name, _, columns in transformer.transformers_
            ],
            "sparse_output": transformer.sparse_output_,
            "steps": steps,
        }
    )


class RAGPreprocessor:
    """A fitted feature-only transformer. Fitting is explicit and train-guarded.

    This module has no runtime integration or persistence format. Its schema is
    RAG-owned; no training config, golden fixture, or model bundle supplies state.
    """

    def __init__(
        self,
        transformer: Any,
        categories: dict[str, tuple[str, ...]],
        fit_metadata: dict[str, Any],
    ) -> None:
        self._transformer = transformer
        self._categories = dict(categories)
        self._output_names = tuple(
            str(name) for name in transformer.get_feature_names_out()
        )
        self._metadata = {
            "fit": deepcopy(fit_metadata),
            "preprocessing": _describe(transformer, categories),
            "libraries": {
                "scikit-learn": sklearn.__version__,
                "numpy": np.__version__,
                "pandas": pd.__version__,
            },
        }

    @classmethod
    def fit_train(
        cls, train_path: str | Path, splits_path: str | Path
    ) -> "RAGPreprocessor":
        train_bytes = Path(train_path).read_bytes()
        split_bytes = Path(splits_path).read_bytes()
        expected_ids = _train_ids(split_bytes)
        validate_schema(list(FEATURE_SCHEMA))
        names = [column.name for column in FEATURE_SCHEMA]
        # Churn is never parsed or passed to fitting; IDs are only a split guard.
        frame = pd.read_csv(
            io.BytesIO(train_bytes),
            usecols=["customerID", *names],
            dtype={"customerID": "string"},
        )
        ids = [validate_id(value) for value in frame.customerID]
        if len(set(ids)) != len(ids) or set(ids) != expected_ids:
            raise ValueError(
                "Fit IDs must match exactly the train split; val/test IDs are forbidden."
            )
        categories: dict[str, tuple[str, ...]] = {}
        for name in CATEGORICAL_FEATURES:
            values = frame[name].tolist()
            if any(not isinstance(value, str) or not value.strip() for value in values):
                raise ValueError(
                    "Training categorical values must be non-missing strings."
                )
            categories[name] = tuple(sorted(set(values)))
        for row in frame[names].to_dict("records"):
            clean_features(
                {name: row[name] for name in names}, list(FEATURE_SCHEMA), categories
            )
        # The CSV is already processed. Validate it, but fit its original values
        # and row order, just as the original feature-only transformer did.
        # In particular, fitting imputation and query cleaning are distinct:
        # a missing TotalCharges is median-imputed at fit, zero-cleaned at query.
        features = frame[names].copy()
        transformer = ColumnTransformer(
            [
                (
                    "num",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    list(NUMERIC_FEATURES),
                ),
                (
                    "cat",
                    Pipeline(
                        [
                            (
                                "imputer",
                                SimpleImputer(
                                    strategy="constant", fill_value="__MISSING__"
                                ),
                            ),
                            ("onehot", OneHotEncoder(handle_unknown="ignore")),
                        ]
                    ),
                    list(CATEGORICAL_FEATURES),
                ),
            ]
        )
        transformer.fit(features)
        return cls(
            transformer,
            categories,
            {
                "row_count": len(ids),
                "train_sha256": sha256(train_bytes),
                "splits_sha256": sha256(split_bytes),
                "membership_sha256": sha256(json_bytes(sorted(ids))),
                "ordered_ids_sha256": sha256(json_bytes(ids)),
                "row_order": "original processed-train order",
                "target_used": False,
            },
        )

    @property
    def feature_schema(self) -> tuple[FeatureColumn, ...]:
        return FEATURE_SCHEMA

    @property
    def input_feature_names(self) -> tuple[str, ...]:
        return tuple(column.name for column in FEATURE_SCHEMA)

    @property
    def output_feature_names(self) -> tuple[str, ...]:
        return self._output_names

    @property
    def categorical_vocabulary(self) -> Mapping[str, tuple[str, ...]]:
        return MappingProxyType(dict(self._categories))

    @property
    def metadata(self) -> dict[str, Any]:
        return deepcopy(self._metadata)

    def clean(self, customer: Mapping[str, object]) -> dict[str, Scalar]:
        if not isinstance(customer, Mapping):
            raise TypeError("Customer input must be a mapping.")
        return clean_features(customer, list(FEATURE_SCHEMA), self._categories)

    def transform_one(self, customer: Mapping[str, object]) -> NDArray[np.float64]:
        return self.transform_many([customer])

    def transform_many(
        self, customers: Sequence[Mapping[str, object]]
    ) -> NDArray[np.float64]:
        if not isinstance(customers, Sequence) or isinstance(customers, (str, bytes)):
            raise TypeError("Customer batch must be a sequence of feature mappings.")
        if not customers:
            return np.empty((0, len(self._output_names)), dtype=np.float64)
        frame = pd.DataFrame(
            [self.clean(row) for row in customers], columns=self.input_feature_names
        )
        transformed = self._transformer.transform(frame)
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()
        result = np.asarray(transformed, dtype=np.float64, order="C")
        if (
            result.shape != (len(customers), len(self._output_names))
            or not np.isfinite(result).all()
        ):
            raise ValueError(
                "Expected a finite float64 matrix with stable feature dimensions."
            )
        return result
