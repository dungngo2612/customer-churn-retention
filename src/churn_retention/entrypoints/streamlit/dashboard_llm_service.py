"""Bounded, explicit Dashboard prediction bridge to the unchanged v2 service."""

import asyncio
import csv
import hashlib
import io
import json
import os
import re
from collections.abc import Callable, Mapping
from typing import Literal

import pandas as pd
from pydantic import ValidationError

from churn_retention.adapters.database.dashboard_runs import (
    DashboardRunStore,
    SavedRun,
    canonical,
)
from churn_retention.adapters.retrieval.reference_store import ReferenceStore
from churn_retention.contracts.application_security import (
    reject_credentials,
    safe_source_name,
)
from churn_retention.contracts.llm import LLMChurnPrediction
from churn_retention.contracts.llm_failure import DashboardPredictionError
from churn_retention.contracts.llm_prediction_record import FEATURE_NAMES
from churn_retention.entrypoints.streamlit.llm_input import (
    LLMInputResources,
    validate_customer,
)
from churn_retention.entrypoints.streamlit.llm_prediction_service import (
    LLMPredictionServiceError,
    predict_customer,
)
from churn_retention.modules.scoring.llm_predictor import LLMChurnPredictor
from churn_retention.modules.scoring.llm_router import LLMRouter

MAX_BATCH = 5
MAX_UPLOAD_ROWS = 1000
PredictOne = Callable[[ReferenceStore, Mapping[str, object]], LLMChurnPrediction]


def validate_frame(frame: pd.DataFrame, resources: LLMInputResources) -> pd.DataFrame:
    required = {*FEATURE_NAMES, "customerID", "Churn"}
    if set(frame.columns) != required or frame.columns.duplicated().any():
        raise ValueError(
            (
                "CSV cần đúng 19 feature, customerID và Churn đã quan sát; kh"
                "ông được thừa/thiếu/trùng cột."
            )
        )
    if frame.empty or len(frame) > MAX_UPLOAD_ROWS:
        raise ValueError("CSV cần từ 1 đến 1.000 dòng. Mỗi lượt chỉ chấm tối đa 5 dòng được chọn.")
    reject_credentials(frame.to_dict(orient="records"))
    rows = []
    seen: set[str] = set()
    for _, row in frame.iterrows():
        cid = row["customerID"]
        if (
            not isinstance(cid, str)
            or not cid.strip()
            or cid != cid.strip()
            or len(cid) > 100
            or any(ord(char) < 32 or ord(char) == 127 for char in cid)
            or cid in seen
        ):
            raise ValueError(
                "customerID phải là chuỗi không rỗng, duy nhất, không có khoảng trắng đầu/cuối."
            )
        seen.add(cid)
        observed = {
            "no": "No",
            "0": "No",
            "false": "No",
            "yes": "Yes",
            "1": "Yes",
            "true": "Yes",
        }.get(str(row["Churn"]).strip().lower())
        if observed is None:
            raise ValueError(
                "Churn phải là trạng thái đã quan sát Yes/No hoặc 0/1; không dùng nhãn LLM."
            )
        try:
            clean = validate_customer(resources, {name: row[name] for name in FEATURE_NAMES})
        except ValueError as error:
            raise ValueError(f"CSV có feature không hợp lệ: {error}") from None
        rows.append({"customerID": cid, "Churn": observed, **clean})
    return pd.DataFrame(rows)


def read_csv(content: bytes, resources: LLMInputResources) -> pd.DataFrame:
    if len(content) > 2_000_000:
        raise ValueError("CSV vượt giới hạn 2 MB.")
    try:
        text = content.decode("utf-8-sig")
        if "\x00" in text:
            raise ValueError("CSV chứa ký tự NUL không hợp lệ.")
        reader = csv.reader(io.StringIO(text), strict=True)
        header = next(reader)
        count = 0
        for row in reader:
            if not row:
                continue
            count += 1
            if len(row) != len(header):
                raise ValueError("CSV có dòng thừa hoặc thiếu trường.")
            if count > MAX_UPLOAD_ROWS:
                raise ValueError("CSV vượt giới hạn 1.000 dòng.")
        if len(header) != len(set(header)):
            raise ValueError("CSV có cột trùng tên.")
        frame = pd.read_csv(io.StringIO(text), dtype={"customerID": str}, keep_default_na=False)
    except (
        UnicodeError,
        csv.Error,
        pd.errors.ParserError,
        pd.errors.EmptyDataError,
        StopIteration,
    ):
        raise ValueError("Không thể đọc CSV UTF-8 hợp lệ.") from None
    return validate_frame(frame, resources)


class MockProvider:
    """Local demo response; Router/RAG/prompt/response validation remain real."""

    async def generate(self, prompt: str) -> str:
        payload = json.loads(
            prompt.split("\nBEGIN_CUSTOMER_DATA_JSON\n", 1)[1].split(
                "\nEND_CUSTOMER_DATA_JSON\n", 1
            )[0]
        )
        customer = payload["customer_to_predict"]
        assert set(customer) == set(FEATURE_NAMES)
        probability = 0.76 if float(customer["MonthlyCharges"]) >= 50 else 0.28
        return json.dumps(
            {
                "churn_prediction": int(probability >= 0.5),
                "churn_probability": probability,
                "reason": (
                    "MOCK provider: đầu ra mô phỏng, không gửi API và không phải dự đoán thực tế."
                ),
            }
        )


def mock_prediction(store: ReferenceStore, snapshot: Mapping[str, object]) -> LLMChurnPrediction:
    async def once() -> LLMChurnPrediction:
        async with LLMRouter(
            MockProvider(),
            MockProvider(),
            primary_provider="Groq",
            fallback_provider="Gemini",
        ) as router:
            return await LLMChurnPredictor(store, router, k=5).predict(snapshot)

    return asyncio.run(once())


class DashboardLLMAdapter:
    def __init__(
        self,
        resources: LLMInputResources,
        pin: str,
        store: DashboardRunStore,
        *,
        mode: Literal["mock", "live"] = "mock",
        predict_one: PredictOne | None = None,
    ):
        if mode not in ("mock", "live"):
            raise ValueError("Prediction mode must be mock or live.")
        if re.fullmatch(r"[a-f0-9]{64}", pin) is None:
            raise ValueError("An external reference SHA-256 pin is required.")
        if mode == "live" and (
            os.environ.get("CHURN_DASHBOARD_ALLOW_LIVE") != "1"
            or os.environ.get("CHURN_DASHBOARD_PREVIEW") == "1"
        ):
            raise ValueError("Live LLM disabled. Preview uses only mock providers.")
        self.resources, self.pin, self.store, self.mode = resources, pin, store, mode
        self.predict_one = predict_one or (mock_prediction if mode == "mock" else predict_customer)

    def prepare(self, frame: pd.DataFrame, source_name: str, request_id: str) -> SavedRun:
        if not 1 <= len(frame) <= MAX_BATCH:
            raise ValueError("Mỗi lượt dự đoán đơn/batch cần 1–5 khách hàng được chọn rõ ràng.")
        clean = validate_frame(frame, self.resources)
        source_name = safe_source_name(source_name)
        return self.store.reserve(
            clean,
            source_name,
            request_id,
            self.resources.store.reference_version,
            self.pin,
            self.mode,
        )

    def process(self, run_id: str) -> dict[str, object]:
        run = self.store.get(run_id)
        if (
            run.reference_sha256 != self.pin
            or run.execution_mode != self.mode
            or run.reference_version != self.resources.store.reference_version
        ):
            self.store.reject(run_id, "provenance")
            raise DashboardPredictionError("provenance")
        if run.status in {"PARTIAL", "FAILED"}:
            raise DashboardPredictionError(self.store.error_code(run_id), run.status)
        if run.status == "COMPLETE":
            return {"run_id": run_id, "status": run.status}
        if self.mode == "live" and (
            os.environ.get("CHURN_DASHBOARD_ALLOW_LIVE") != "1"
            or os.environ.get("CHURN_DASHBOARD_PREVIEW") == "1"
        ):
            self.store.reject(run_id, "configuration")
            raise DashboardPredictionError("configuration")
        if not self.store.start(run_id):
            raise ValueError("Run already in flight; no automatic retry")
        try:
            snapshot = json.loads(run.cohort_json)
            if (
                not isinstance(snapshot, list)
                or not 1 <= len(snapshot) <= MAX_BATCH
                or hashlib.sha256(canonical(snapshot).encode()).hexdigest() != run.dataset_digest
            ):
                raise ValueError("Invalid saved snapshot")
        except (ValueError, TypeError):
            self.store.reject(run_id, "snapshot")
            raise DashboardPredictionError("snapshot") from None
        try:
            frame = validate_frame(pd.DataFrame(snapshot), self.resources)
        except (ValueError, TypeError):
            self.store.reject(run_id, "validation")
            raise DashboardPredictionError("validation") from None
        for _, row in frame.iterrows():
            cid = row["customerID"]
            self.store.begin_attempt(run_id, cid)
            try:
                result = self.predict_one(
                    self.resources.store, {name: row[name] for name in FEATURE_NAMES}
                )
                if not isinstance(result, LLMChurnPrediction):
                    raise DashboardPredictionError("invalid_response")
                result = LLMChurnPrediction.model_validate(result.model_dump())
                reject_credentials(result.model_dump())
                if (
                    result.reference_version != run.reference_version
                    or result.reference_count != 5
                    or result.provider not in {"Groq", "Gemini"}
                ):
                    raise DashboardPredictionError("provenance")
            except Exception as error:  # noqa: BLE001 -- Persist only allowlisted failure codes.
                code = (
                    error.code
                    if isinstance(error, (DashboardPredictionError, LLMPredictionServiceError))
                    else "invalid_response"
                    if isinstance(error, ValidationError)
                    else "prediction_failed"
                )
                self.store.outcome(run_id, cid, None, error_code=code)
                finished = self.store.finish(run_id)
                raise DashboardPredictionError(code, finished.status) from None
            self.store.outcome(run_id, cid, result)
        finished = self.store.finish(run_id)
        return {
            "run_id": run_id,
            "status": finished.status,
            "total_stored_in_db": len(frame),
        }
