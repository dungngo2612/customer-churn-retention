"""The remaining entrypoints must run with classifier packages unavailable."""

import os
import subprocess
import sys
from pathlib import Path


def test_entrypoints_and_rag_build_without_classifier_dependencies(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    script = r"""
import importlib.abc
import os
from pathlib import Path
import socket
import sys

root = Path(sys.argv[1])
sys.path[:0] = [str(root), str(root / "src"), str(root / "tests")]
blocked = (
    "catboost", "shap", "mlflow",
    "churn_retention.ml.training", "churn_retention.ml.explainability",
    "churn_retention.adapters.model_registry", "churn_retention.adapters.predictor",
    "churn_retention.adapters.retrieval.legacy_reference_store",
    "churn_retention.adapters.retrieval._artifact",
)

class ClassifierBarrier(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == prefix or fullname.startswith(prefix + ".") for prefix in blocked):
            raise AssertionError("Classifier dependency reached by an LLM entrypoint")

def no_bundle(event, args):
    if event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
        name = os.fsdecode(args[0]).replace("\\", "/")
        if name.endswith("model.pkl") or "/artifacts/bundles/" in name:
            raise AssertionError("Classifier bundle access is forbidden")

def no_network(*args, **kwargs):
    raise AssertionError("No API calls during isolation checks")

sys.meta_path.insert(0, ClassifierBarrier())
sys.addaudithook(no_bundle)
socket.socket.connect = no_network
socket.socket.connect_ex = no_network
socket.create_connection = no_network

import main
from churn_retention.dashboard import demo
from churn_retention.entrypoints.streamlit import dashboard_llm_ui, llm_prediction_service
from churn_retention.entrypoints.worker import worker
from churn_retention.adapters.retrieval.reference_store import ReferenceStore
from churn_retention.entrypoints.streamlit.dashboard_llm_service import mock_prediction
from llm_v2_fixtures import make_source, query

source = make_source(Path(sys.argv[2]) / "synthetic-rag")
source.build()  # RAG preprocessing fitting remains allowed and necessary.
store = ReferenceStore.load(source.output, trusted=True)
features = {
    name: value for name, value in query(source).items()
    if name not in {"customerID", "Churn"}
}
result = mock_prediction(store, features)
assert result.reference_count == 5 and result.reference_version == store.reference_version
assert not result.probability_calibrated
assert not hasattr(worker, "IdempotentPredictionStore")
assert not hasattr(worker.JobWorker, "_handle_scoring")
assert main.health_check() == {"status": "ok"}
assert not any(name == p or name.startswith(p + ".") for name in sys.modules for p in blocked)
print("LLM entrypoints and RAG round trip passed without classifier dependencies")
"""
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", script, str(root), str(tmp_path)],
        cwd=tmp_path,
        env={**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"},
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    assert result.stdout.strip().endswith("passed without classifier dependencies")
