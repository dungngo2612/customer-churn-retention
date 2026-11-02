"""Clean process for the primary Cường UI, without bundles/legacy modules/APIs."""

import importlib.abc
import json
import os
import socket
import sys
import types
from pathlib import Path
from typing import Any
from unittest.mock import patch


def main() -> None:
    root, artifact, pin = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(root / "src"))
    blocked = (
        "churn_retention.ml",
        "churn_retention.workflows",
        "churn_retention.adapters.predictor",
        "churn_retention.adapters.retrieval.legacy_reference_store",
        "churn_retention.adapters.retrieval._artifact",
        "churn_retention.adapters.llm.groq_provider",
        "churn_retention.adapters.llm.gemini_provider",
        "churn_retention.modules.scoring.jobs",
        "churn_retention.modules.scoring.use_cases",
        "catboost",
        "shap",
        "mlflow",
        "groq",
        "google.genai",
    )

    class Barrier(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname: str, path: Any = None, target: Any = None):
            if any(fullname == p or fullname.startswith(p + ".") for p in blocked):
                raise AssertionError(
                    "Legacy classifier/provider implementation unavailable"
                )
            if fullname == "sklearn.linear_model._logistic":
                raise AssertionError("Logistic implementation unavailable")

    class NoClassifier:
        def __new__(cls, *args: Any, **kwargs: Any):
            raise AssertionError("No classifier construction or inference")

    poison = types.ModuleType("sklearn.linear_model._logistic")
    poison.LogisticRegression = poison.LogisticRegressionCV = NoClassifier  # type: ignore[attr-defined]
    sys.modules[poison.__name__] = poison
    sys.meta_path.insert(0, Barrier())

    def audit(event: str, args: tuple[Any, ...]) -> None:
        if event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
            path = os.fsdecode(args[0]).replace("\\", "/")
            if path.endswith((".pkl", ".pickle", ".joblib", "telco-train-v1.zip")):
                raise AssertionError("No model/bundle/v1 file reads")
        if event == "pickle.find_class" and str(args[0]).startswith(
            ("sklearn", "churn_retention.ml")
        ):
            raise AssertionError("No model unpickling")

    sys.addaudithook(audit)

    def forbidden(*args: Any, **kwargs: Any):
        raise AssertionError("No network, old store or fitting")

    patch.multiple(socket.socket, connect=forbidden, connect_ex=forbidden).start()
    socket.create_connection = forbidden
    for key in ("GROQ_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY"):
        os.environ.pop(key, None)
    os.environ.update(
        {
            "CHURN_DASHBOARD_PREVIEW": "1",
            "CHURN_DASHBOARD_STATE": str(Path.cwd() / "state.sqlite3"),
            "CHURN_LLM_REFERENCE_ARTIFACT": str(artifact),
            "CHURN_LLM_REFERENCE_SHA256": pin,
        }
    )
    from sklearn.compose import ColumnTransformer  # type: ignore[import-untyped]
    from sklearn.impute import SimpleImputer  # type: ignore[import-untyped]
    from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]
    from sklearn.preprocessing import OneHotEncoder, StandardScaler  # type: ignore[import-untyped]
    from streamlit.testing.v1 import AppTest

    from churn_retention.adapters.database.dashboard_runs import DashboardRunStore
    from churn_retention.adapters.retrieval.rag_preprocessor import RAGPreprocessor
    from churn_retention.entrypoints.streamlit import dashboard_llm_ui as ui
    from churn_retention.entrypoints.streamlit import demo_security
    from churn_retention.entrypoints.streamlit.dashboard_llm_service import (
        DashboardLLMAdapter,
    )
    from churn_retention.entrypoints.worker.worker import IdempotentPredictionStore

    # This import-isolation probe uses AppTest with no HTTP server.
    patch.object(demo_security, "require_safe_runtime", lambda: None).start()
    patch.object(RAGPreprocessor, "fit_train", forbidden).start()
    patch.object(IdempotentPredictionStore, "__init__", forbidden).start()
    for cls in (
        ColumnTransformer,
        SimpleImputer,
        Pipeline,
        OneHotEncoder,
        StandardScaler,
    ):
        patch.multiple(cls, fit=forbidden, fit_transform=forbidden).start()
    adapter = DashboardLLMAdapter(
        ui.resources(), pin, DashboardRunStore(ui.runtime_path())
    )
    run = adapter.prepare(ui.synthetic_frame(), "synthetic-demo.csv", "seed")
    job = ui.execute(adapter, run)
    assert job is not None and job.status.value == "SUCCESS"
    os.environ["CHURN_DASHBOARD_DEMO_RUN"] = run.run_id
    patch.object(DashboardLLMAdapter, "process", forbidden).start()
    at = AppTest.from_file(
        str(root / "src/churn_retention/dashboard/app.py"), default_timeout=30
    ).run()
    at.session_state["active_scoring_context"] = {"protected": True}
    assert not at.exception and not at.error and len(at.tabs) == 4
    at.slider(key="e2e_budget_slider").set_value(1000).run()
    at.button(key="btn_save_e2e_camp").click().run()
    assert not at.exception and not at.error
    assert ui.list_campaigns()[0].scoring_run_id == run.run_id
    assert at.session_state["active_scoring_context"] == {"protected": True}
    assert not any(
        n == p or n.startswith(p + ".") for n in sys.modules for p in blocked
    )
    assert not (root / "src/churn_retention/ml").exists()
    assert not (root / "artifacts/bundles").exists()
    assert not (root / "data").exists()
    for name, module in tuple(sys.modules.items()):
        filename = getattr(module, "__file__", None)
        if name.startswith("churn_retention") and isinstance(filename, str):
            assert Path(filename).resolve().is_relative_to(root / "src")
    print(
        json.dumps(
            {
                "primary_dashboard_tabs": 4,
                "legacy_imports": 0,
                "real_api_calls": 0,
                "accepted_mock_predictions": 4,
                "campaign_saved": True,
            }
        )
    )


if __name__ == "__main__":
    main()
