"""LLM-only form data and validation, using the existing reference-store boundary."""

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

from churn_retention.adapters.retrieval._reference_v2_format import ReferenceV2Error
from churn_retention.adapters.retrieval.reference_store import ReferenceStore

DEFAULT_REFERENCE_PATH = (
    Path(__file__).resolve().parents[4] / "artifacts/references/telco-train-v2.zip"
)
# Approved local artifact from step 7.5; the pin is external to the ZIP manifest.
DEFAULT_REFERENCE_SHA256 = (
    "c85e61be7edfd0e91b248e0e353088677a345a378e64840b1613b51194b40bf8"
)


def reference_artifact_path() -> Path:
    configured = os.environ.get("CHURN_LLM_REFERENCE_ARTIFACT")
    if configured is not None and not configured.strip():
        raise ReferenceV2Error("Reference artifact path configuration is empty.")
    return Path(configured) if configured is not None else DEFAULT_REFERENCE_PATH


def reference_artifact_sha256() -> str:
    pin = os.environ.get("CHURN_LLM_REFERENCE_SHA256", DEFAULT_REFERENCE_SHA256).strip()
    if re.fullmatch(r"[a-f0-9]{64}", pin) is None:
        raise ReferenceV2Error("A valid external reference artifact hash is required.")
    return pin


VALIDATED_INPUT_KEY = "llm_input.validated_customer"
RESOURCES_KEY = "llm_input.resources"
ERROR_KEY = "llm_input.error"
FINGERPRINT_KEY = "llm_input.reference_fingerprint"
RESULT_KEY = "llm_input.prediction"
RESULT_INPUT_KEY = "llm_input.prediction_input"
SUCCESS_MESSAGE = "Dữ liệu hợp lệ. Sẵn sàng thực hiện dự đoán bằng LLM."

FEATURE_GROUPS = (
    (
        "Thông tin khách hàng",
        ("gender", "SeniorCitizen", "Partner", "Dependents", "tenure"),
    ),
    (
        "Dịch vụ sử dụng",
        (
            "PhoneService",
            "MultipleLines",
            "InternetService",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
        ),
    ),
    (
        "Hợp đồng và thanh toán",
        (
            "Contract",
            "PaperlessBilling",
            "PaymentMethod",
            "MonthlyCharges",
            "TotalCharges",
        ),
    ),
)

FEATURE_LABELS = {
    "gender": "Giới tính",
    "SeniorCitizen": "Khách hàng cao tuổi",
    "Partner": "Có bạn đời",
    "Dependents": "Có người phụ thuộc",
    "tenure": "Thời gian sử dụng",
    "PhoneService": "Dịch vụ điện thoại",
    "MultipleLines": "Nhiều đường dây điện thoại",
    "InternetService": "Dịch vụ Internet",
    "OnlineSecurity": "Bảo mật trực tuyến",
    "OnlineBackup": "Sao lưu trực tuyến",
    "DeviceProtection": "Bảo vệ thiết bị",
    "TechSupport": "Hỗ trợ kỹ thuật",
    "StreamingTV": "Truyền hình trực tuyến",
    "StreamingMovies": "Phim trực tuyến",
    "Contract": "Loại hợp đồng",
    "PaperlessBilling": "Hóa đơn điện tử",
    "PaymentMethod": "Phương thức thanh toán",
    "MonthlyCharges": "Chi phí hàng tháng",
    "TotalCharges": "Tổng chi phí",
}

_VALUE_LABELS = {
    "Female": "Nữ",
    "Male": "Nam",
    "No": "Không",
    "Yes": "Có",
    "No phone service": "Không sử dụng dịch vụ điện thoại",
    "No internet service": "Không sử dụng Internet",
    "DSL": "DSL",
    "Fiber optic": "Cáp quang",
    "Month-to-month": "Hàng tháng",
    "One year": "Một năm",
    "Two year": "Hai năm",
    "Bank transfer (automatic)": "Chuyển khoản ngân hàng (tự động)",
    "Credit card (automatic)": "Thẻ tín dụng (tự động)",
    "Electronic check": "Séc điện tử",
    "Mailed check": "Séc gửi qua bưu điện",
}


def value_label(value: str) -> str:
    return _VALUE_LABELS.get(value, value)


@dataclass(frozen=True)
class LLMInputResources:
    store: ReferenceStore = field(repr=False)
    feature_names: tuple[str, ...]
    choices: Mapping[str, tuple[str, ...]]


def form_resources(store: ReferenceStore) -> LLMInputResources:
    """Project category values from verified train records once, without IDs/labels.

    These are observed, store-validated values, not a new fitted vocabulary.
    The fitted preprocessor remains the authoritative validator in prepare_query.
    No model-bundle or training API is imported by this UI adapter.
    """
    schema = store.feature_schema
    names = tuple(column.name for column in schema)
    grouped_names = tuple(name for _, group in FEATURE_GROUPS for name in group)
    if len(names) != 19 or set(names) != set(grouped_names):
        raise ValueError("Reference schema does not match the 19-feature LLM form.")
    choices = {}
    for column in schema:
        if column.kind == "categorical":
            values = {record.features[column.name] for record in store.records}
            if not values or any(not isinstance(value, str) for value in values):
                raise ValueError("Reference contains invalid categorical form values.")
            choices[column.name] = tuple(sorted(str(value) for value in values))
    return LLMInputResources(store, names, MappingProxyType(choices))


def load_form_resources(path: str | Path) -> LLMInputResources:
    return form_resources(
        ReferenceStore.load(path, expected_artifact_sha256=reference_artifact_sha256())
    )


@dataclass(frozen=True)
class ReferenceFingerprint:
    path: str
    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int


def reference_fingerprint(path: str | Path) -> ReferenceFingerprint:
    """Cheap cache invalidation, not a replacement for Store integrity checks.

    Loading still verifies artifact hashes/version/schema. Stat metadata detects
    ordinary edits and atomic replacements without hashing the ZIP every rerun.
    """
    resolved = Path(path).resolve(strict=True)
    status = resolved.stat()
    if not resolved.is_file():
        raise ValueError("Reference artifact must be a regular file.")
    return ReferenceFingerprint(
        str(resolved),
        status.st_dev,
        status.st_ino,
        status.st_size,
        status.st_mtime_ns,
        status.st_ctime_ns,
    )


def _validation_message(error: ValueError | TypeError) -> str:
    # Translate only known validator messages; never display arbitrary values,
    # numerical examples, raw transformer exceptions, or customer information.
    message = str(error)
    if message == "Missing TotalCharges is only valid when tenure is zero.":
        return "Chỉ được để trống Tổng chi phí khi Thời gian sử dụng (tenure) bằng 0."
    if message == "SeniorCitizen must be 0 or 1.":
        return "Khách hàng cao tuổi chỉ nhận giá trị 0 hoặc 1."
    for name, label in FEATURE_LABELS.items():
        if message == f"Invalid categorical value for {name}.":
            return f"Vui lòng chọn giá trị hợp lệ cho {label}."
        if message == f"{name} must be an integer.":
            return f"{label} phải là số nguyên không âm."
        if message in {
            f"{name} must be a numerical scalar or numeric string.",
            f"{name} must be a finite number.",
            f"{name} must be finite and non-negative.",
        }:
            return f"{label} phải là số hợp lệ, hữu hạn và không âm."
    return "Không thể xác thực dữ liệu. Vui lòng kiểm tra đủ 19 đặc trưng và các giá trị đã nhập."


def validate_customer(
    resources: LLMInputResources, values: Mapping[str, object]
) -> dict[str, str | int | float]:
    if not isinstance(values, Mapping) or set(values) != set(resources.feature_names):
        raise ValueError(
            "Thông tin khách hàng phải chứa đúng 19 đặc trưng, không có ID hoặc nhãn Churn."
        )
    try:
        prepared = resources.store.prepare_query(values)
    except (ValueError, TypeError) as error:
        raise ValueError(_validation_message(error)) from None
    # Keep only raw cleaned features. The store's transformed vector is not UI input.
    return dict(prepared.features)
