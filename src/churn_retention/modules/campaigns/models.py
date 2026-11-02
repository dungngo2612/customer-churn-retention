"""Domain models and data structures for Retention Campaign Management."""

from __future__ import annotations

import json
import math
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import Enum
from numbers import Real
from typing import Any


class CampaignStatus(str, Enum):
    """Trạng thái vòng đời của một chiến dịch giữ chân."""

    DRAFT = "DRAFT"
    SIMULATED = "SIMULATED"
    EXPORTED = "EXPORTED"


@dataclass(frozen=True)
class Campaign:
    """Bản ghi chiến dịch giữ chân bất biến (Immutable Campaign Entity).

    Thuộc tính:
    -----------
    campaign_id : str
        Định danh duy nhất của chiến dịch (UUID4).
    name : str
        Tên định danh do người dùng đặt cho chiến dịch.
    dataset_id : str
        Tham chiếu bất biến tới tập dữ liệu đầu vào.
    model_version : str
        Tham chiếu bất biến tới phiên bản mô hình ML đã chấm điểm.
    scoring_run_id : str
        Tham chiếu bất biến tới lượt chạy chấm điểm (scoring batch run).
    budget : float
        Hạn mức ngân sách tối đa được phân bổ cho chiến dịch (USD).
    assumptions : Dict[str, Any]
        Giả định kinh tế (margin, horizon_months, action_configs, v.v.).
    status : CampaignStatus
        Trạng thái hiện tại: DRAFT, SIMULATED, hoặc EXPORTED.
    results : Dict[str, Any]
        Kết quả phân bổ từ thuật toán ngân sách (tổng quan KPI và danh sách chi tiết).
    created_at : str
        Thời điểm tạo chiến dịch theo chuẩn ISO 8601 UTC.
    updated_at : str
        Thời điểm cập nhật cuối cùng theo chuẩn ISO 8601 UTC.
    """

    campaign_id: str
    name: str
    dataset_id: str
    model_version: str
    scoring_run_id: str
    budget: float
    assumptions: dict[str, Any] = field(default_factory=dict)
    status: CampaignStatus = CampaignStatus.SIMULATED
    results: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def __post_init__(self) -> None:
        if (
            isinstance(self.budget, bool)
            or not isinstance(self.budget, Real)
            or not math.isfinite(self.budget)
            or self.budget < 0
        ):
            raise ValueError("Campaign budget must be finite and non-negative")
        if any(
            not isinstance(value, str) or not value.strip()
            for value in (
                self.campaign_id,
                self.dataset_id,
                self.model_version,
                self.scoring_run_id,
            )
        ):
            raise ValueError("Campaign identity/provenance cannot be empty")
        if not isinstance(self.assumptions, dict) or not isinstance(self.results, dict):
            raise TypeError("Campaign snapshot must contain objects")
        json.dumps([self.assumptions, self.results], allow_nan=False)

    @classmethod
    def create_new(
        cls,
        name: str,
        dataset_id: str,
        model_version: str,
        scoring_run_id: str,
        budget: float,
        assumptions: dict[str, Any],
        results: dict[str, Any] | None = None,
        status: CampaignStatus = CampaignStatus.SIMULATED,
    ) -> Campaign:
        """Tạo một chiến dịch mới với UUID và timestamp hợp lệ."""
        now_iso = datetime.now(UTC).isoformat()
        return cls(
            campaign_id=str(uuid.uuid4()),
            name=name.strip() or f"Campaign_{now_iso[:10]}",
            dataset_id=dataset_id,
            model_version=model_version,
            scoring_run_id=scoring_run_id,
            budget=float(budget),
            assumptions=assumptions or {},
            status=status,
            results=results or {},
            created_at=now_iso,
            updated_at=now_iso,
        )

    def with_status(self, new_status: CampaignStatus) -> Campaign:
        """Tạo bản sao mới với trạng thái được cập nhật (bảo toàn tính bất biến)."""
        now_iso = datetime.now(UTC).isoformat()
        return Campaign(
            campaign_id=self.campaign_id,
            name=self.name,
            dataset_id=self.dataset_id,
            model_version=self.model_version,
            scoring_run_id=self.scoring_run_id,
            budget=self.budget,
            assumptions=self.assumptions,
            status=new_status,
            results=self.results,
            created_at=self.created_at,
            updated_at=now_iso,
        )

    def to_dict(self) -> dict[str, Any]:
        """Chuyển đổi thành từ điển tuần tự hóa được sang JSON."""
        d = asdict(self)
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Campaign:
        """Tái cấu trúc đối tượng Campaign từ từ điển."""
        status_val = data.get("status", CampaignStatus.SIMULATED.value)
        status = CampaignStatus(status_val) if isinstance(status_val, str) else status_val
        return cls(
            campaign_id=data["campaign_id"],
            name=data["name"],
            dataset_id=data["dataset_id"],
            model_version=data["model_version"],
            scoring_run_id=data["scoring_run_id"],
            budget=float(data["budget"]),
            assumptions=data.get("assumptions", {}),
            status=status,
            results=data.get("results", {}),
            created_at=data["created_at"],
            updated_at=data.get("updated_at", data["created_at"]),
        )
