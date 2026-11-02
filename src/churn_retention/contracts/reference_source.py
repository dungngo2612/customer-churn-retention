"""Structural retrieval input shared by explicit legacy and v2 stores."""

from collections.abc import Mapping
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from churn_retention.contracts.reference import PreparedQuery, ReferenceRecord


class ReferenceSource(Protocol):
    @property
    def records(self) -> tuple[ReferenceRecord, ...]: ...

    @property
    def matrix(self) -> NDArray[np.float64]: ...

    def prepare_query(
        self, query: Mapping[str, object], *, customer_id: str | None = None
    ) -> PreparedQuery: ...
