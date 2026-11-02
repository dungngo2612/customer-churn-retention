"""Public LLM/RAG Store: v2 only, with no legacy import or fallback.

The JSON loader restores only RAG preprocessing state. Historical classifier-
backed v1 loading is preserved on the archive branch, outside main.
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
