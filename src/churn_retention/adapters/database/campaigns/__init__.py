"""Campaigns database adapter package."""

from churn_retention.adapters.database.campaigns.repository import (
    CampaignRepository,
    SqliteCampaignRepository,
)

__all__ = ["CampaignRepository", "SqliteCampaignRepository"]
