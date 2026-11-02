"""Campaign database repository implementation with SQLite for local persistence."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Protocol

from churn_retention.adapters.database.local_sqlite import connect_local
from churn_retention.contracts.application_security import reject_credentials
from churn_retention.modules.campaigns.models import Campaign, CampaignStatus


class CampaignRepository(Protocol):
    """Interface định nghĩa hợp đồng lưu trữ và truy xuất chiến dịch."""

    def save(self, campaign: Campaign) -> None:
        """Lưu hoặc cập nhật một bản ghi chiến dịch."""
        ...

    def get(self, campaign_id: str) -> Campaign | None:
        """Truy xuất chiến dịch theo ID."""
        ...

    def list_all(self) -> list[Campaign]:
        """Liệt kê toàn bộ danh sách chiến dịch, sắp xếp mới nhất lên trước."""
        ...

    def delete(self, campaign_id: str) -> bool:
        """Xóa chiến dịch theo ID."""
        ...


class SqliteCampaignRepository:
    """Hiện thực CampaignRepository bằng SQLite, lưu trữ bền vững qua các lần khởi động lại."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        if db_path is None:
            repo_root = Path(__file__).resolve().parents[5]
            data_dir = repo_root / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            self._db_path = data_dir / "campaigns.db"
        else:
            self._db_path = Path(db_path)
            self._db_path.parent.mkdir(parents=True, exist_ok=True)

        self._init_db()

    @property
    def db_path(self) -> Path:
        return self._db_path

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager mở và đóng kết nối an toàn, tương thích tuyệt đối với Windows locking."""
        conn = connect_local(self._db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        """Khởi tạo cấu trúc bảng campaigns nếu chưa tồn tại."""
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS campaigns (
                    campaign_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    dataset_id TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    scoring_run_id TEXT NOT NULL,
                    budget REAL NOT NULL,
                    assumptions TEXT NOT NULL,
                    status TEXT NOT NULL,
                    results TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_campaigns_created_at
                ON campaigns(created_at DESC);
                """
            )
            conn.commit()

    def save(self, campaign: Campaign) -> None:
        """Lưu hoặc cập nhật một bản ghi chiến dịch vào SQLite."""
        reject_credentials(campaign.to_dict())
        assumptions_json = json.dumps(
            campaign.assumptions, ensure_ascii=False, sort_keys=True, allow_nan=False
        )
        results_json = json.dumps(
            campaign.results, ensure_ascii=False, sort_keys=True, allow_nan=False
        )

        with self._get_connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute(
                "SELECT * FROM campaigns WHERE campaign_id=?", (campaign.campaign_id,)
            ).fetchone()
            if existing is not None:
                snapshot = (
                    campaign.name,
                    campaign.dataset_id,
                    campaign.model_version,
                    campaign.scoring_run_id,
                    float(campaign.budget),
                    assumptions_json,
                    results_json,
                    campaign.created_at,
                )
                previous = (
                    existing["name"],
                    existing["dataset_id"],
                    existing["model_version"],
                    existing["scoring_run_id"],
                    existing["budget"],
                    existing["assumptions"],
                    existing["results"],
                    existing["created_at"],
                )
                if snapshot != previous:
                    raise ValueError(
                        "Campaign snapshot is immutable; create a new campaign."
                    )
            if existing is not None:
                order = {"DRAFT": 0, "SIMULATED": 1, "EXPORTED": 2}
                if order[campaign.status.value] <= order[existing["status"]]:
                    return
            conn.execute(
                """
                INSERT INTO campaigns (
                    campaign_id, name, dataset_id, model_version, scoring_run_id,
                    budget, assumptions, status, results, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(campaign_id) DO UPDATE SET
                    status = excluded.status,
                    updated_at = excluded.updated_at;
                """,
                (
                    campaign.campaign_id,
                    campaign.name,
                    campaign.dataset_id,
                    campaign.model_version,
                    campaign.scoring_run_id,
                    float(campaign.budget),
                    assumptions_json,
                    campaign.status.value,
                    results_json,
                    campaign.created_at,
                    campaign.updated_at,
                ),
            )
            conn.commit()

    def get(self, campaign_id: str) -> Campaign | None:
        """Truy xuất chiến dịch theo ID."""
        with self._get_connection() as conn:
            cur = conn.execute(
                "SELECT * FROM campaigns WHERE campaign_id = ?",
                (campaign_id,),
            )
            row = cur.fetchone()
            if row is None:
                return None
            return self._row_to_campaign(row)

    def list_all(self) -> list[Campaign]:
        """Liệt kê toàn bộ chiến dịch đã lưu, xếp theo created_at giảm dần."""
        with self._get_connection() as conn:
            cur = conn.execute("SELECT * FROM campaigns ORDER BY created_at DESC")
            return [self._row_to_campaign(row) for row in cur.fetchall()]

    def delete(self, campaign_id: str) -> bool:
        """Xóa chiến dịch theo ID."""
        with self._get_connection() as conn:
            cur = conn.execute(
                "DELETE FROM campaigns WHERE campaign_id = ?",
                (campaign_id,),
            )
            conn.commit()
            return cur.rowcount > 0

    def _row_to_campaign(self, row: sqlite3.Row) -> Campaign:
        """Chuyển đổi một dòng dữ liệu SQLite thành thực thể Campaign."""
        reject_credentials(dict(row))
        assumptions = json.loads(row["assumptions"]) if row["assumptions"] else {}
        results = json.loads(row["results"]) if row["results"] else {}

        return Campaign(
            campaign_id=row["campaign_id"],
            name=row["name"],
            dataset_id=row["dataset_id"],
            model_version=row["model_version"],
            scoring_run_id=row["scoring_run_id"],
            budget=float(row["budget"]),
            assumptions=assumptions,
            status=CampaignStatus(row["status"]),
            results=results,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
