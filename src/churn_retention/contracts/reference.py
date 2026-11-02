"""Classifier-independent reference contracts, cleaning, and canonical utilities.

No bundle loading, fitted transformer access, model inference, or artifact I/O.
"""

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Annotated, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

Scalar = str | int | float
ChurnLabel = Literal["No", "Yes"]
Digest = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
NUMERIC_NAMES = frozenset({"SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"})
QUERY_METADATA = frozenset({"customerID", "customer_id", "Churn"})


class ReferenceArtifactError(ValueError):
    """The reference artifact is invalid, corrupt, or incompatible."""


class _StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid", strict=True, frozen=True, allow_inf_nan=False
    )


class FeatureColumn(_StrictModel):
    name: str
    kind: Literal["numeric", "categorical"]
    dtype: str


class StoredRecord(_StrictModel):
    customer_id: str
    features: dict[str, Scalar]
    churn: ChurnLabel


@dataclass(frozen=True)
class ReferenceRecord:
    features: Mapping[str, Scalar]
    churn: ChurnLabel
    customer_id: str = field(repr=False)

    def to_prompt_data(self) -> dict[str, object]:
        """Feature/label projection only; identifiers never enter this payload."""
        return {"features": dict(self.features), "Churn": self.churn}


@dataclass(frozen=True)
class PreparedQuery:
    features: Mapping[str, Scalar]
    vector: NDArray[np.float64] = field(repr=False)
    customer_id: str | None = field(default=None, repr=False)


@dataclass(frozen=True)
class SimilarCustomer(ReferenceRecord):
    distance: float


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def validate_schema(schema: list[FeatureColumn]) -> None:
    names = [c.name for c in schema]
    numeric = {c.name for c in schema if c.kind == "numeric"}
    if len(names) != 19 or len(set(names)) != 19 or set(names) & QUERY_METADATA:
        raise ReferenceArtifactError(
            "Feature schema must contain 19 unique features without IDs or Churn."
        )
    if numeric != NUMERIC_NAMES:
        raise ReferenceArtifactError("Unsupported numerical feature schema.")
    for column in schema:
        if column.kind == "numeric":
            if np.dtype(column.dtype).kind not in "iuf":
                raise ReferenceArtifactError(
                    "Invalid numerical dtype in feature schema."
                )
        elif column.dtype not in {"str", "object", "string"}:
            raise ReferenceArtifactError("Invalid categorical dtype in feature schema.")


def validate_id(value: object) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(
            "Customer ID must be a non-empty string without surrounding whitespace."
        )
    return value


def _number(value: object, column: FeatureColumn) -> int | float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (str, int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{column.name} must be a numerical scalar or numeric string.")
    try:
        number = float(value)  # type: ignore[arg-type]
    except (ValueError, TypeError, OverflowError):
        raise ValueError(f"{column.name} must be a finite number.") from None
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{column.name} must be finite and non-negative.")
    if np.dtype(column.dtype).kind in "iu":
        if not number.is_integer():
            raise ValueError(f"{column.name} must be an integer.")
        return int(number)
    return number


def clean_features(
    values: Mapping[str, object],
    schema: list[FeatureColumn],
    categories: Mapping[str, tuple[str, ...]],
) -> dict[str, Scalar]:
    names = {c.name for c in schema}
    if names - values.keys():
        raise ValueError("Customer input is missing required features.")
    if set(values) - names - QUERY_METADATA:
        raise ValueError("Customer input contains unsupported fields.")
    clean: dict[str, Scalar] = {}
    for column in schema:
        if column.kind == "numeric" and column.name != "TotalCharges":
            clean[column.name] = _number(values[column.name], column)
    if clean["SeniorCitizen"] not in (0, 1):
        raise ValueError("SeniorCitizen must be 0 or 1.")

    total = values["TotalCharges"]
    missing = (
        total is None
        or total is pd.NA
        or (isinstance(total, str) and not total.strip())
    )
    if isinstance(total, (float, np.floating)) and math.isnan(float(total)):
        missing = True
    if missing:
        if clean["tenure"] != 0:
            raise ValueError("Missing TotalCharges is only valid when tenure is zero.")
        total = 0.0
    for column in schema:
        if column.name == "TotalCharges":
            clean[column.name] = _number(total, column)
        elif column.kind == "categorical":
            value = values[column.name]
            if not isinstance(value, str) or value not in categories[column.name]:
                raise ValueError(f"Invalid categorical value for {column.name}.")
            clean[column.name] = value
    return {column.name: clean[column.name] for column in schema}
