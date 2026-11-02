"""Exact Euclidean top-K retrieval over a cached reference store."""

from collections.abc import Mapping

import numpy as np
from sklearn.neighbors import NearestNeighbors  # type: ignore[import-untyped]

from churn_retention.contracts.reference import ReferenceRecord, SimilarCustomer
from churn_retention.contracts.reference_source import ReferenceSource

__all__ = ["ReferenceRecord", "SimilarCustomer", "SimilarCustomerRetriever"]


def _validate_k(k: int) -> int:
    if type(k) is not int or k <= 0:
        raise ValueError("K must be a positive integer.")
    return k


class SimilarCustomerRetriever:
    def __init__(self, store: ReferenceSource, *, k: int = 5) -> None:
        self._store = store
        self._k = _validate_k(k)
        # This fit only indexes already-transformed vectors; it fits no scaler,
        # encoder, classifier, or label-dependent model.
        self._neighbors = NearestNeighbors(metric="euclidean", algorithm="brute", n_jobs=1)
        self._neighbors.fit(store.matrix)

    def retrieve(
        self,
        query: Mapping[str, object],
        *,
        k: int | None = None,
        customer_id: str | None = None,
    ) -> list[SimilarCustomer]:
        requested = self._k if k is None else _validate_k(k)
        prepared = self._store.prepare_query(query, customer_id=customer_id)
        records = self._store.records
        # Ask for all distances so ties at the K boundary are broken globally
        # by customer ID, rather than backend-dependent partial-sort ordering.
        distances, indices = self._neighbors.kneighbors(prepared.vector, n_neighbors=len(records))
        ids = np.asarray([records[int(index)].customer_id for index in indices[0]])
        order = np.lexsort((ids, distances[0]))
        results = []
        for position in order:
            record = records[int(indices[0, position])]
            if record.customer_id == prepared.customer_id:
                continue
            results.append(
                SimilarCustomer(
                    record.features,
                    record.churn,
                    record.customer_id,
                    float(distances[0, position]),
                )
            )
            if len(results) == requested:
                break
        return results
