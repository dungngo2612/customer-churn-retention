"""Public LLM/RAG Store: v2 only, with no legacy import or fallback.

The implementation is the JSON loader validated in step 7.5. v1 callers must
explicitly import legacy_reference_store; old bundle filters belong there.
"""

from churn_retention.adapters.retrieval.reference_v2 import VerificationReferenceV2
from churn_retention.contracts.reference import (
    ChurnLabel,
    FeatureColumn,
    PreparedQuery,
    ReferenceArtifactError,
    ReferenceRecord,
    Scalar,
    StoredRecord,
    clean_features,
)

ReferenceStore = VerificationReferenceV2

__all__ = [
    "ChurnLabel",
    "FeatureColumn",
    "PreparedQuery",
    "ReferenceArtifactError",
    "ReferenceRecord",
    "ReferenceStore",
    "Scalar",
    "StoredRecord",
    "clean_features",
]
