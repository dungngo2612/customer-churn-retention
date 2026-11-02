"""Use cases for Retention Campaign Management."""

from __future__ import annotations

import io
import math
import uuid
from dataclasses import replace
from numbers import Real
from typing import TYPE_CHECKING, Any

import pandas as pd

from churn_retention.contracts.application_security import reject_credentials
from churn_retention.modules.campaigns.models import Campaign, CampaignStatus

if TYPE_CHECKING:
    from churn_retention.adapters.database.campaigns.repository import (
        CampaignRepository,
    )


class CampaignNotFoundError(KeyError):
    """Chiến dịch không tồn tại trong hệ thống."""


def get_default_repository() -> Any:
    """Khởi tạo kho lưu trữ chiến dịch mặc định."""
    from churn_retention.adapters.database.campaigns.repository import (
        SqliteCampaignRepository,
    )

    return SqliteCampaignRepository()


def create_campaign(
    name: str,
    dataset_id: str,
    model_version: str,
    scoring_run_id: str,
    budget: float,
    assumptions: dict[str, Any],
    results: dict[str, Any] | None = None,
    status: CampaignStatus = CampaignStatus.SIMULATED,
    repository: CampaignRepository | None = None,
    idempotency_key: str | None = None,
) -> Campaign:
    """Tạo mới một chiến dịch mô phỏng và lưu trữ bất biến vào cơ sở dữ liệu.

    Đặc tính bất biến: Mỗi chiến dịch gắn chặt với một bộ tham chiếu duy nhất
    (dataset_id, model_version, scoring_run_id). Khi thay đổi mô hình hoặc dữ liệu,
    hệ thống tạo chiến dịch mới và bảo toàn 100% dữ liệu của chiến dịch cũ.
    """
    if (
        isinstance(budget, bool)
        or not isinstance(budget, Real)
        or not math.isfinite(budget)
        or budget < 0
    ):
        raise ValueError(f"Ngân sách chiến dịch không được âm: {budget}")

    repo = repository or get_default_repository()

    campaign = Campaign.create_new(
        name=name,
        dataset_id=dataset_id,
        model_version=model_version,
        scoring_run_id=scoring_run_id,
        budget=budget,
        assumptions=assumptions or {},
        results=results or {},
        status=status,
    )

    if idempotency_key is not None:
        if not idempotency_key.strip():
            raise ValueError("Campaign idempotency key cannot be empty")
        campaign = replace(
            campaign,
            campaign_id=str(
                uuid.uuid5(uuid.NAMESPACE_URL, "churn-campaign:" + idempotency_key)
            ),
        )
        existing = repo.get(campaign.campaign_id)
        if existing is not None:
            campaign = replace(campaign, created_at=existing.created_at)
    try:
        repo.save(campaign)
    except ValueError:
        # A concurrent identical intent may have won after the initial read.
        existing = (
            repo.get(campaign.campaign_id) if idempotency_key is not None else None
        )
        if existing is None:
            raise
        repo.save(replace(campaign, created_at=existing.created_at))
    return repo.get(campaign.campaign_id) or campaign


def get_campaign(
    campaign_id: str,
    repository: CampaignRepository | None = None,
) -> Campaign:
    """Lấy thông tin chi tiết một chiến dịch theo ID."""
    repo = repository or get_default_repository()
    camp = repo.get(campaign_id)
    if camp is None:
        raise CampaignNotFoundError(f"Không tìm thấy chiến dịch với ID: {campaign_id}")
    return camp


def list_campaigns(
    repository: CampaignRepository | None = None,
) -> list[Campaign]:
    """Lấy danh sách tất cả các chiến dịch đã lưu trong hệ thống."""
    repo = repository or get_default_repository()
    return repo.list_all()


EXPORT_COLUMNS = [
    "customer_id",
    "action_code",
    "action_name",
    "cost",
    "benefit",
    "net_value",
]


def export_campaign_dataframe(
    campaign_id: str,
    repository: CampaignRepository | None = None,
) -> pd.DataFrame:
    """Trích xuất danh sách khách hàng được chọn can thiệp dưới dạng DataFrame."""
    campaign = get_campaign(campaign_id, repository)

    allocations = campaign.results.get("allocations", [])
    if not allocations:
        return pd.DataFrame(columns=EXPORT_COLUMNS)

    df = pd.DataFrame(allocations)

    # Chuẩn hóa tên cột customer_id nếu đầu vào là customerID
    if "customer_id" not in df.columns and "customerID" in df.columns:
        df["customer_id"] = df["customerID"]

    # Đảm bảo các cột bắt buộc tồn tại
    for col in EXPORT_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0 if col in ("cost", "benefit", "net_value") else ""

    # Lọc các khách hàng có hành động can thiệp (loại bỏ A0 - không can thiệp nếu cần,
    # hoặc giữ lại những khách có hành động được chọn)
    return df[EXPORT_COLUMNS].copy()


def safe_export_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Escape untrusted text cells for spreadsheet consumers; do not change stored IDs."""
    frame = frame.copy()
    reject_credentials(frame.to_dict(orient="records"))
    for column in frame.columns:
        frame[column] = frame[column].map(
            lambda value: (
                "'" + value
                if isinstance(value, str)
                and (
                    value.lstrip().startswith(("=", "+", "-", "@"))
                    or value.startswith(("\t", "\r", "\n"))
                )
                else value
            )
        )
    return frame


def export_campaign_csv(
    campaign_id: str,
    only_interventions: bool = True,
    repository: CampaignRepository | None = None,
) -> str:
    """Xuất danh sách khách hàng được chọn can thiệp ra định dạng CSV và chuyển trạng thái sang EXPORTED.

    Cột đầu ra chuẩn:
        customer_id: Mã định danh khách hàng
        action_code: Mã hành động
        action_name: Tên hành động can thiệp
        cost: Chi phí can thiệp kỳ vọng ($)
        benefit: Lợi ích kinh tế kỳ vọng ($)
        net_value: Giá trị ròng kỳ vọng ($)
    """
    repo = repository or get_default_repository()
    campaign = get_campaign(campaign_id, repo)

    df = export_campaign_dataframe(campaign_id, repo)
    if only_interventions and not df.empty and "action_code" in df.columns:
        df = df[df["action_code"] != "A0"]

    csv_buffer = io.StringIO()
    safe_export_frame(df).to_csv(csv_buffer, index=False, encoding="utf-8")
    csv_content = csv_buffer.getvalue()

    # Cập nhật trạng thái chiến dịch sang EXPORTED
    updated_campaign = campaign.with_status(CampaignStatus.EXPORTED)
    repo.save(updated_campaign)

    return csv_content
