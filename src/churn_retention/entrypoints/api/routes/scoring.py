from collections.abc import Callable
from datetime import datetime

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from churn_retention.contracts.http.scoring import PredictionBatch
from churn_retention.modules.scoring.ports import Predictor
from churn_retention.modules.scoring.repository import ScoringRunRepository
from churn_retention.modules.scoring.use_cases import DatasetNotFoundError, score_dataset


class CreateScoringRunRequest(BaseModel):
    dataset_id: str


def create_scoring_router(
    loader: Callable[[str], pd.DataFrame],
    predictor: Predictor,
    repository: ScoringRunRepository,
    run_id_factory: Callable[[], str],
    clock: Callable[[], datetime],
) -> APIRouter:
    router = APIRouter(prefix="/scoring", tags=["scoring"])

    @router.post("/runs", response_model=PredictionBatch, status_code=201)
    def create_run(request: CreateScoringRunRequest) -> PredictionBatch:
        try:
            batch = score_dataset(
                request.dataset_id, loader, predictor, run_id_factory, clock
            )
        except DatasetNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        repository.save(batch)
        return batch

    @router.get("/runs/{run_id}", response_model=PredictionBatch)
    def get_run(run_id: str) -> PredictionBatch:
        batch = repository.get(run_id)
        if batch is None:
            raise HTTPException(
                status_code=404, detail=f"Scoring run '{run_id}' not found"
            )
        return batch

    return router
