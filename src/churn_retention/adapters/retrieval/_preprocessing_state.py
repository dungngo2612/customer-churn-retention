"""Data-only JSON codec for a bounded, fixed sklearn transformer allowlist.

No pickle, eval, module lookup, __setstate__, or artifact-supplied constructor is
used. JSON state restores only the five trusted local transformer classes.
"""

import math
from typing import Any

import numpy as np
import sklearn  # type: ignore[import-untyped]
from sklearn.compose import ColumnTransformer  # type: ignore[import-untyped]
from sklearn.impute import SimpleImputer  # type: ignore[import-untyped]
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # type: ignore[import-untyped]

from churn_retention.adapters.retrieval._reference_v2_format import (
    ReferenceV2Error,
    strict_json,
)
from churn_retention.adapters.retrieval.rag_preprocessor import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    RAGPreprocessor,
)
from churn_retention.contracts.reference import json_bytes

CLASSES = {
    cls.__name__: cls
    for cls in (
        ColumnTransformer,
        Pipeline,
        SimpleImputer,
        StandardScaler,
        OneHotEncoder,
    )
}
STATE_KEYS = {
    "ColumnTransformer": {
        "transformers",
        "remainder",
        "sparse_threshold",
        "n_jobs",
        "transformer_weights",
        "verbose",
        "verbose_feature_names_out",
        "feature_names_in_",
        "n_features_in_",
        "_columns",
        "_transformer_to_input_indices",
        "_remainder",
        "sparse_output_",
        "transformers_",
        "output_indices_",
        "_sklearn_version",
    },
    "Pipeline": {"steps", "transform_input", "memory", "verbose", "_sklearn_version"},
    "SimpleImputer": {
        "missing_values",
        "add_indicator",
        "keep_empty_features",
        "strategy",
        "fill_value",
        "copy",
        "feature_names_in_",
        "n_features_in_",
        "_fit_dtype",
        "_fill_dtype",
        "indicator_",
        "statistics_",
        "_sklearn_version",
    },
    "StandardScaler": {
        "with_mean",
        "with_std",
        "copy",
        "n_features_in_",
        "n_samples_seen_",
        "mean_",
        "var_",
        "scale_",
        "_sklearn_version",
    },
    "OneHotEncoder": {
        "categories",
        "sparse_output",
        "dtype",
        "handle_unknown",
        "drop",
        "min_frequency",
        "max_categories",
        "feature_name_combiner",
        "_infrequent_enabled",
        "n_features_in_",
        "categories_",
        "_drop_idx_after_grouping",
        "drop_idx_",
        "_n_features_outs",
        "_sklearn_version",
    },
}


def _dtype(value: str) -> np.dtype[Any]:
    result = np.dtype(value)
    if result.fields or result.kind not in "fiuObUS" or result.itemsize > 1024:
        raise ReferenceV2Error("Unsupported preprocessing array dtype.")
    return result


def _encode(value: Any) -> Any:
    if type(value).__name__ in CLASSES and type(value) is CLASSES[type(value).__name__]:
        state = value.__getstate__()
        if set(state) - STATE_KEYS[type(value).__name__]:
            raise ReferenceV2Error("Unsupported transformer state fields.")
        return {
            "kind": "estimator",
            "class": type(value).__name__,
            "state": _encode(state),
        }
    if isinstance(value, np.ndarray):
        return {
            "kind": "array",
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "items": _encode(value.tolist()),
        }
    if isinstance(value, np.dtype):
        return {"kind": "dtype", "value": value.str}
    if isinstance(value, np.generic):
        return {
            "kind": "scalar",
            "dtype": value.dtype.str,
            "value": _encode(value.item()),
        }
    if value is np.float64:
        return {"kind": "float64_type"}
    if isinstance(value, slice):
        return {"kind": "slice", "items": [value.start, value.stop, value.step]}
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {
            "kind": "dict",
            "items": {key: _encode(item) for key, item in value.items()},
        }
    if isinstance(value, (tuple, list)):
        return {
            "kind": "tuple" if isinstance(value, tuple) else "list",
            "items": [_encode(item) for item in value],
        }
    if isinstance(value, float) and math.isnan(value):
        return {"kind": "nan"}
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise ReferenceV2Error("Unsupported preprocessing state value.")


def _decode(value: Any, budget: list[int], depth: int = 0) -> Any:
    budget[0] -= 1
    if budget[0] < 0 or depth > 32:
        raise ReferenceV2Error("Preprocessing state exceeds decoding limits.")
    if value is None or type(value) in (str, int, float, bool):
        if isinstance(value, float) and not math.isfinite(value):
            raise ReferenceV2Error("Invalid preprocessing scalar.")
        return value
    if not isinstance(value, dict) or "kind" not in value:
        raise ReferenceV2Error("Invalid preprocessing state node.")
    kind = value["kind"]
    keys = {
        "estimator": {"kind", "class", "state"},
        "array": {"kind", "dtype", "shape", "items"},
        "scalar": {"kind", "dtype", "value"},
        "dtype": {"kind", "value"},
        "slice": {"kind", "items"},
        "dict": {"kind", "items"},
        "tuple": {"kind", "items"},
        "list": {"kind", "items"},
        "nan": {"kind"},
        "float64_type": {"kind"},
    }
    if kind not in keys or set(value) != keys[kind]:
        raise ReferenceV2Error("Unsupported preprocessing node kind or fields.")

    def child(item: Any) -> Any:
        return _decode(item, budget, depth + 1)

    if kind == "nan":
        return float("nan")
    if kind == "float64_type":
        return np.float64
    if kind == "dtype":
        return _dtype(value["value"])
    if kind == "scalar":
        return np.asarray(child(value["value"]), dtype=_dtype(value["dtype"]))[()]
    if kind in {"list", "tuple", "slice"}:
        if not isinstance(value["items"], list) or len(value["items"]) > 16_384:
            raise ReferenceV2Error("Invalid preprocessing sequence.")
        items = [child(item) for item in value["items"]]
        if kind == "slice":
            if len(items) != 3 or any(
                item is not None and type(item) is not int for item in items
            ):
                raise ReferenceV2Error("Invalid preprocessing slice.")
            return slice(*items)
        return tuple(items) if kind == "tuple" else items
    if kind == "dict":
        if not isinstance(value["items"], dict) or len(value["items"]) > 1024:
            raise ReferenceV2Error("Invalid preprocessing mapping.")
        return {key: child(item) for key, item in value["items"].items()}
    if kind == "array":
        shape = value["shape"]
        if (
            not isinstance(shape, list)
            or len(shape) > 2
            or any(type(n) is not int or n < 0 for n in shape)
            or math.prod(shape) > 16_384
        ):
            raise ReferenceV2Error("Invalid preprocessing array shape.")
        items = child(value["items"])
        array = np.asarray(items, dtype=_dtype(value["dtype"]))
        if list(array.shape) != shape or (
            array.dtype.kind == "O"
            and any(type(item) is not str for item in array.flat)
        ):
            raise ReferenceV2Error("Invalid preprocessing array values.")
        return array
    name = value["class"]
    if name not in CLASSES:
        raise ReferenceV2Error("Transformer class is not permitted.")
    state = child(value["state"])
    if (
        not isinstance(state, dict)
        or set(state) - STATE_KEYS[name]
        or state.pop("_sklearn_version", None) != sklearn.__version__
    ):
        raise ReferenceV2Error("Invalid or incompatible transformer state.")
    estimator = object.__new__(CLASSES[name])
    estimator.__dict__.update(state)
    return estimator


def dump_preprocessing(preprocessor: RAGPreprocessor) -> bytes:
    return json_bytes(
        {
            "encoding_version": 1,
            "transformer": _encode(preprocessor._transformer),
            "metadata": preprocessor.metadata,
        }
    )


def _validate_recipe(transformer: Any, fit_rows: int) -> None:
    expected_columns = [
        ("num", list(NUMERIC_FEATURES)),
        ("cat", list(CATEGORICAL_FEATURES)),
    ]
    if (
        [(name, list(cols)) for name, _, cols in transformer.transformers_]
        != expected_columns
        or [(name, list(cols)) for name, _, cols in transformer.transformers]
        != expected_columns
        or list(transformer.feature_names_in_)
        != list(NUMERIC_FEATURES + CATEGORICAL_FEATURES)
        or transformer.remainder != "drop"
        or transformer.sparse_threshold != 0.3
        or transformer.n_jobs is not None
        or transformer.transformer_weights is not None
        or transformer.verbose is not False
        or transformer.verbose_feature_names_out is not True
    ):
        raise ReferenceV2Error("Unsupported RAG transformer recipe.")
    num, cat = (transformer.named_transformers_[name] for name in ("num", "cat"))
    if (
        type(num) is not Pipeline
        or type(cat) is not Pipeline
        or list(num.named_steps) != ["imputer", "scaler"]
        or list(cat.named_steps) != ["imputer", "onehot"]
        or num.memory is not None
        or cat.memory is not None
    ):
        raise ReferenceV2Error("Unsupported RAG pipeline recipe.")
    ni, scaler, ci, encoder = (
        num.named_steps["imputer"],
        num.named_steps["scaler"],
        cat.named_steps["imputer"],
        cat.named_steps["onehot"],
    )
    if (
        type(ni) is not SimpleImputer
        or type(ci) is not SimpleImputer
        or type(scaler) is not StandardScaler
        or type(encoder) is not OneHotEncoder
        or ni.strategy != "median"
        or ci.strategy != "constant"
        or ci.fill_value != "__MISSING__"
        or ni.add_indicator
        or ci.add_indicator
        or ni.keep_empty_features
        or ci.keep_empty_features
        or not math.isnan(ni.missing_values)
        or not math.isnan(ci.missing_values)
        or not scaler.with_mean
        or not scaler.with_std
        or scaler.n_samples_seen_ != fit_rows
        or encoder.categories != "auto"
        or encoder.drop is not None
        or encoder.handle_unknown != "ignore"
        or encoder.dtype is not np.float64
        or not encoder.sparse_output
        or encoder.min_frequency is not None
        or encoder.max_categories is not None
        or encoder.feature_name_combiner != "concat"
    ):
        raise ReferenceV2Error("Unsupported fitted preprocessing parameters.")
    for value in (ni.statistics_, scaler.mean_, scaler.var_, scaler.scale_):
        if (
            not isinstance(value, np.ndarray)
            or value.shape != (4,)
            or value.dtype != np.float64
            or not np.isfinite(value).all()
        ):
            raise ReferenceV2Error("Invalid numerical fitted statistics.")
    if (
        np.any(scaler.var_ < 0)
        or np.any(scaler.scale_ <= 0)
        or len(encoder.categories_) != 15
    ):
        raise ReferenceV2Error("Invalid scaling or category statistics.")
    for values in encoder.categories_:
        if (
            not len(values)
            or any(type(item) is not str or not item for item in values)
            or list(values) != sorted(set(values))
        ):
            raise ReferenceV2Error("Invalid fitted category vocabulary.")


def load_preprocessing(data: bytes) -> RAGPreprocessor:
    payload = strict_json(data)
    if (
        not isinstance(payload, dict)
        or set(payload) != {"encoding_version", "transformer", "metadata"}
        or payload["encoding_version"] != 1
    ):
        raise ReferenceV2Error("Invalid preprocessing envelope.")
    transformer = _decode(payload["transformer"], [40_000])
    if type(transformer) is not ColumnTransformer:
        raise ReferenceV2Error("Expected a feature-only ColumnTransformer.")
    expected = payload["metadata"]
    _validate_recipe(transformer, expected["fit"]["row_count"])
    categories = {
        name: tuple(values)
        for name, values in expected["preprocessing"]["categorical_vocabulary"].items()
    }
    encoder = transformer.named_transformers_["cat"].named_steps["onehot"]
    actual_categories = {
        name: tuple(str(value) for value in values)
        for name, values in zip(CATEGORICAL_FEATURES, encoder.categories_, strict=True)
    }
    if categories != actual_categories:
        raise ReferenceV2Error(
            "Category validation vocabulary differs from the fitted encoder."
        )
    preprocessor = RAGPreprocessor(transformer, categories, expected["fit"])
    if preprocessor.metadata != expected:
        raise ReferenceV2Error("Restored preprocessing does not match its metadata.")
    return preprocessor
