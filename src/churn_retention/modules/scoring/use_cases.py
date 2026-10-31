from collections.abc import Callable
from datetime import datetime

import pandas as pd

from churn_retention.contracts.http.scoring import Prediction, PredictionBatch
from churn_retention.modules.scoring.ports import Predictor


class DatasetNotFoundError(LookupError):
    """The injected loader could not find the requested dataset."""


class InvalidDatasetError(ValueError):
    """R09: The dataset has invalid structure (null/duplicate customer_id, etc.)."""


def score_dataset(
    dataset_id: str,
    loader: Callable[[str], pd.DataFrame],
    predictor: Predictor,
    run_id_factory: Callable[[], str],
    clock: Callable[[], datetime],
) -> PredictionBatch:
    try:
        dataset = loader(dataset_id)
    except FileNotFoundError as error:
        raise DatasetNotFoundError(f"Dataset '{dataset_id}' not found") from error

    # R09: Raise InvalidDatasetError for dataset validation failures
    if "customer_id" not in dataset.columns:
        raise InvalidDatasetError("Dataset must contain customer_id column")
    customers = dataset["customer_id"]
    if not all(isinstance(customer_id, str) for customer_id in customers):
        raise InvalidDatasetError("customer_id values must be non-null strings")
    if customers.duplicated().any():
        raise InvalidDatasetError("customer_id values must be unique")

    customer_ids = customers.tolist()
    features = dataset.set_index("customer_id")
    model_version = predictor.model_version
    probabilities = predictor.predict_batch(features, customer_ids)
    if len(probabilities) != len(customer_ids):
        raise ValueError(
            f"Predictor returned {len(probabilities)} probabilities "
            f"for {len(customer_ids)} customers"
        )

    # The Predictor contract orders its list by customer_ids; key it at this boundary.
    probabilities_by_id = dict(zip(customer_ids, probabilities, strict=True))
    run_id = run_id_factory()
    scored_at = clock()
    return PredictionBatch(
        scoring_run_id=run_id,
        model_version=model_version,
        predictions=[
            Prediction(
                customer_id=customer_id,
                dataset_id=dataset_id,
                scoring_run_id=run_id,
                churn_probability=probabilities_by_id[customer_id],
                model_version=model_version,
                scored_at=scored_at,
            )
            for customer_id in customer_ids
        ],
    )
