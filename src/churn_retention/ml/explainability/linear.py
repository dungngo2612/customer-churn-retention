"""Interventional SHAP for the unweighted, uncalibrated logistic bundle."""

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from shap import LinearExplainer, maskers  # type: ignore[import-untyped]
from sklearn.compose import ColumnTransformer  # type: ignore[import-untyped]
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]
from sklearn.preprocessing import OneHotEncoder  # type: ignore[import-untyped]

from churn_retention.ml.training.bundle import LoadedBundle

BACKGROUND_SEED = 42
MAX_BACKGROUND_ROWS = 200
ID_COLUMNS = ("customer_id", "customerID")


def sample_background(background_df: pd.DataFrame) -> pd.DataFrame:
    """Sample train rows only; callers must establish their train provenance.

    A DataFrame cannot prove its origin. Reject explicit non-train provenance,
    and retain customer IDs in the result so callers can audit the sample.
    """
    if background_df.empty:
        raise ValueError("Background must contain train rows")
    if background_df.attrs.get("source_split", "train") != "train":
        raise ValueError("Background must come only from train.csv, never val/test")
    if "split" in background_df and not background_df["split"].eq("train").all():
        raise ValueError("Background contains rows outside the train split")
    return background_df.sample(
        n=min(len(background_df), MAX_BACKGROUND_ROWS), random_state=BACKGROUND_SEED
    ).copy()


def _logistic_parts(bundle: LoadedBundle) -> tuple[Any, Any]:
    if bundle.metadata.get("model_type") == "catboost":
        raise NotImplementedError("CatBoost: chưa hỗ trợ SHAP trong M2.3")
    pipeline = bundle._pipeline
    if not isinstance(pipeline, Pipeline) or list(pipeline.named_steps) != [
        "preprocessor",
        "model",
    ]:
        raise TypeError("SHAP requires an uncalibrated logistic pipeline")
    preprocessor, model = pipeline.steps[0][1], pipeline.steps[1][1]
    if not isinstance(model, LogisticRegression) or not isinstance(preprocessor, ColumnTransformer):
        raise TypeError("SHAP requires an uncalibrated logistic pipeline")
    if (
        model.class_weight is not None
        or bundle.metadata.get("class_weight", False)
        or bundle.metadata.get("calibration_applied", "original") != "original"
    ):
        raise ValueError("M2.3 supports only no-class-weight, uncalibrated logistic")
    if list(model.classes_) != [0, 1]:
        raise ValueError("SHAP requires binary classes [0, 1], with 1 = churn")
    return preprocessor, model


def _feature_mapping(preprocessor: Any) -> tuple[list[str], list[str]]:
    """Map fitted output columns without guessing from underscore-separated names."""
    names = list(preprocessor.get_feature_names_out())
    sources: list[str] = []
    for name, transformer, columns in preprocessor.transformers_:
        if isinstance(transformer, str) and transformer == "drop":
            continue
        if not isinstance(transformer, Pipeline):
            raise TypeError("Unsupported preprocessing mapping: expected fitted pipelines")
        last = transformer.steps[-1][1]
        if isinstance(last, OneHotEncoder):
            inputs = list(transformer[:-1].get_feature_names_out(columns))
            infrequent = getattr(last, "infrequent_categories_", None)
            counts = [len(categories) for categories in last.categories_]
            for i in range(len(inputs)):
                if infrequent is not None and infrequent[i] is not None:
                    counts[i] -= len(infrequent[i]) - 1
                if last.drop_idx_ is not None and last.drop_idx_[i] is not None:
                    counts[i] -= 1
            mapped = [col for col, count in zip(inputs, counts, strict=True) for _ in range(count)]
        else:
            mapped = list(transformer.get_feature_names_out(columns))
        if any(col not in columns for col in mapped):
            raise ValueError("Unsupported preprocessing: output does not map to input features")
        output_slice = preprocessor.output_indices_[name]
        if len(mapped) != output_slice.stop - output_slice.start:
            raise ValueError("Preprocessing feature mapping has inconsistent dimensions")
        sources.extend(mapped)
    if len(sources) != len(names):
        raise ValueError("Preprocessing feature mapping has inconsistent dimensions")
    return names, sources


def _dense(values: Any) -> NDArray[np.float64]:
    if hasattr(values, "toarray"):
        values = values.toarray()
    return np.asarray(values, dtype=np.float64)


def _features(bundle: LoadedBundle, frame: pd.DataFrame) -> pd.DataFrame:
    names = [column["name"] for column in bundle.feature_schema]
    missing = set(names) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing features for SHAP: {sorted(missing)}")
    return frame.loc[:, names]


def _calculate(
    bundle: LoadedBundle, background_df: pd.DataFrame, row_df: pd.DataFrame | None
) -> dict[str, Any]:
    preprocessor, model = _logistic_parts(bundle)
    background = sample_background(background_df)
    transformed_background = _dense(preprocessor.transform(_features(bundle, background)))
    frame = background if row_df is None else row_df
    transformed = _dense(preprocessor.transform(_features(bundle, frame)))
    names, sources = _feature_mapping(preprocessor)
    # Independent defaults to 100 rows; pass the explicit cap to retain our sample.
    masker = maskers.Independent(transformed_background, max_samples=MAX_BACKGROUND_ROWS)
    explainer = LinearExplainer(model, masker)
    values = np.asarray(explainer.shap_values(transformed), dtype=float)
    base_value = float(np.asarray(explainer.expected_value).item())
    grouped = {
        column["name"]: values[:, [source == column["name"] for source in sources]].sum(axis=1)
        for column in bundle.feature_schema
    }
    logits = np.asarray(model.decision_function(transformed), dtype=float)
    if not np.allclose(base_value + values.sum(axis=1), logits, atol=1e-6, rtol=0):
        raise ValueError("SHAP additivity failed for logistic log-odds")
    ids = next((background[c].astype(str).tolist() for c in ID_COLUMNS if c in background), [])
    return {
        "frame": frame,
        "transformed": transformed,
        "names": names,
        "sources": sources,
        "values": values,
        "grouped": grouped,
        "logits": logits,
        "base_value": base_value,
        "background": {
            "seed": BACKGROUND_SEED,
            "rows": len(background),
            "customer_ids": ids,
        },
    }


def _metadata(bundle: LoadedBundle, calculated: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_version": bundle.model_version,
        "output_type": "log_odds",
        "calibration_applied": "original",
        "feature_schema": [dict(column) for column in bundle.feature_schema],
        "base_value": calculated["base_value"],
        "background": calculated["background"],
    }


def explain_global(bundle: LoadedBundle, background_df: pd.DataFrame) -> dict[str, Any]:
    """Mean absolute SHAP on the sampled train background, grouped BEFORE abs."""
    calculated = _calculate(bundle, background_df, None)
    importance = [
        {"feature_name": name, "mean_abs_shap": float(np.abs(values).mean())}
        for name, values in calculated["grouped"].items()
    ]
    importance.sort(key=lambda item: (-item["mean_abs_shap"], item["feature_name"]))
    return {
        **_metadata(bundle, calculated),
        "feature_importance": importance,
        "detailed_importance": [
            {
                "feature_name": name,
                "source_feature": source,
                "mean_abs_shap": float(np.abs(calculated["values"][:, i]).mean()),
            }
            for i, (name, source) in enumerate(
                zip(calculated["names"], calculated["sources"], strict=True)
            )
        ],
    }


def explain_one(
    bundle: LoadedBundle, row_df: pd.DataFrame, background_df: pd.DataFrame
) -> dict[str, Any]:
    """Explain one row in log-odds. SHAP describes the model, not causality."""
    if len(row_df) != 1:
        raise ValueError("explain_one requires exactly one customer row")
    calculated = _calculate(bundle, background_df, row_df)
    customer_id = next(
        (str(row_df[c].iloc[0]) for c in ID_COLUMNS if c in row_df),
        str(row_df.index[0]),
    )
    contributions = []
    for name, values in calculated["grouped"].items():
        raw = row_df[name].iloc[0]
        feature_value = (
            "(missing; imputed)"
            if pd.isna(raw)
            else (float(raw) if isinstance(raw, (int, float, np.number)) else str(raw))
        )
        contributions.append(
            {
                "customer_id": customer_id,
                "model_version": bundle.model_version,
                "feature_name": name,
                "feature_value": feature_value,
                "contribution": float(values[0]),
                "base_value": calculated["base_value"],
                "explained_output": "log_odds",
                "calibration_applied": "original",
            }
        )
    return {
        **_metadata(bundle, calculated),
        "customer_id": customer_id,
        "contributions": contributions,
        "detailed_contributions": [
            {
                "feature_name": name,
                "source_feature": source,
                "transformed_value": float(calculated["transformed"][0, i]),
                "contribution": float(calculated["values"][0, i]),
            }
            for i, (name, source) in enumerate(
                zip(calculated["names"], calculated["sources"], strict=True)
            )
        ],
        "log_odds": float(calculated["logits"][0]),
        "probability": float(bundle.predict_proba(_features(bundle, row_df))[0]),
    }
