"""Fresh-process standalone UI probe; only controlled synthetic queries are read."""

import hashlib
import importlib.abc
import json
import os
import socket
import sys
import types
import zipfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch


def main() -> None:
    root, query_file, pin, scenario = (
        Path(sys.argv[1]),
        Path(sys.argv[2]),
        sys.argv[3],
        sys.argv[4],
    )
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(root / "src"))
    blocked = (
        "churn_retention.ml",
        "churn_retention.workflows",
        "churn_retention.adapters.predictor",
        "churn_retention.adapters.database",
        "churn_retention.adapters.task_queue",
        "churn_retention.entrypoints.worker",
        "churn_retention.dashboard",
        "churn_retention.modules.campaigns",
        "churn_retention.adapters.retrieval._artifact",
        "churn_retention.adapters.retrieval.legacy_reference_store",
        "churn_retention.entrypoints.streamlit.explanation_views",
        "churn_retention.modules.scoring.jobs",
        "churn_retention.modules.scoring.batch_ports",
        "churn_retention.modules.scoring.use_cases",
        "churn_retention.modules.scoring.repository",
        "churn_retention.modules.scoring.ports",
        "churn_retention.adapters.llm.groq_provider",
        "churn_retention.adapters.llm.gemini_provider",
        "shap",
        "catboost",
        "mlflow",
    )

    class Barrier(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname: str, path: Any = None, target: Any = None):
            if any(fullname == p or fullname.startswith(p + ".") for p in blocked):
                raise AssertionError("Legacy/scoring/provider imports unavailable")
            if fullname == "sklearn.linear_model._logistic":
                raise AssertionError("Classifier implementation unavailable")

    class NoClassifier:
        def __new__(cls, *args: Any, **kwargs: Any):
            raise AssertionError("No classifier construction or inference")

    # compose imports the linear_model namespace for its unrelated target regressor.
    # Poison only logistic exports: its real implementation cannot be imported.
    poison = types.ModuleType("sklearn.linear_model._logistic")
    poison.LogisticRegression = poison.LogisticRegressionCV = NoClassifier  # type: ignore[attr-defined]
    sys.modules[poison.__name__] = poison
    sys.meta_path.insert(0, Barrier())

    def audit(event: str, args: tuple[Any, ...]) -> None:
        if event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
            path = os.fsdecode(args[0]).replace("\\", "/")
            if (
                path.endswith(
                    (
                        ".pkl",
                        ".pickle",
                        ".joblib",
                        "scoring_page.py",
                        "1_Explain.py",
                        "telco-train-v1.zip",
                    )
                )
                or "/artifacts/bundles/" in path
            ):
                raise AssertionError("Legacy files/models unavailable")
        if event == "pickle.find_class" and str(args[0]).startswith(
            ("sklearn", "churn_retention.ml")
        ):
            raise AssertionError("No classifier/state unpickling")

    sys.addaudithook(audit)

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("No network or preprocessing fitting")

    patch.multiple(socket.socket, connect=forbidden, connect_ex=forbidden).start()
    socket.create_connection = forbidden
    os.environ["GROQ_API_KEY"] = os.environ["GEMINI_API_KEY"] = "fake-no-network"
    os.environ.pop("CHURN_LLM_REFERENCE_ARTIFACT", None)
    os.environ.pop("CHURN_LLM_REFERENCE_SHA256", None)
    if scenario != "project":
        os.environ["CHURN_LLM_REFERENCE_SHA256"] = pin
    if scenario == "missing_keys":
        os.environ.pop("GROQ_API_KEY")
        os.environ.pop("GEMINI_API_KEY")
    if scenario == "untrusted_artifact":
        os.environ["CHURN_LLM_REFERENCE_SHA256"] = "0" * 64
    if scenario == "missing_artifact":
        os.environ["CHURN_LLM_REFERENCE_ARTIFACT"] = str(Path.cwd() / "missing.zip")
    if scenario == "v1_rejected":
        old = Path.cwd() / "synthetic-v1.zip"
        with zipfile.ZipFile(old, "w") as archive:
            archive.writestr("manifest.json", '{"format_version":1}')
            archive.writestr("bundle/model.pkl", b"never-unpickle")
        os.environ["CHURN_LLM_REFERENCE_ARTIFACT"] = str(old)
        os.environ["CHURN_LLM_REFERENCE_SHA256"] = hashlib.sha256(
            old.read_bytes()
        ).hexdigest()

    import streamlit as st
    from sklearn.compose import ColumnTransformer  # type: ignore[import-untyped]
    from sklearn.impute import SimpleImputer  # type: ignore[import-untyped]
    from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]
    from sklearn.preprocessing import OneHotEncoder, StandardScaler  # type: ignore[import-untyped]
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    from streamlit.testing.v1 import AppTest

    from churn_retention.adapters.retrieval.rag_preprocessor import RAGPreprocessor
    from churn_retention.entrypoints.streamlit import llm_input
    from churn_retention.entrypoints.streamlit import llm_prediction_service as service
    from churn_retention.modules.scoring.llm_router import LLMRouter

    patch.object(RAGPreprocessor, "fit_train", forbidden).start()
    for cls in (
        ColumnTransformer,
        SimpleImputer,
        Pipeline,
        StandardScaler,
        OneHotEncoder,
    ):
        patch.multiple(cls, fit=forbidden, fit_transform=forbidden).start()
    prompts: list[str] = []

    class Provider:
        async def generate(self, prompt: str) -> str:
            prompts.append(prompt)
            if scenario == "invalid_response":
                return "SYNTHETIC_INVALID_RESPONSE"
            return json.dumps(
                {
                    "churn_prediction": 1,
                    "churn_probability": 0.6,
                    "reason": "Bằng chứng tham khảo còn hạn chế; ước lượng chưa chắc chắn.",
                },
                ensure_ascii=False,
            )

    @asynccontextmanager
    async def router() -> AsyncIterator[LLMRouter]:
        async with LLMRouter(
            Provider(), Provider(), primary_provider="Groq", fallback_provider="Gemini"
        ) as instance:
            yield instance

    patch.object(service, "groq_gemini_router", router).start()
    loads: list[Path] = []
    original_load = llm_input.load_form_resources

    def load(path: str | Path) -> llm_input.LLMInputResources:
        loads.append(Path(path))
        return original_load(path)

    llm_input.load_form_resources = load
    original_navigation = st.navigation
    registries: list[tuple[Path, ...]] = []

    def navigation(pages: Any, **kwargs: Any) -> Any:
        assert len(pages) == 2 and kwargs["position"] == "sidebar"
        selected = original_navigation(pages, **kwargs)
        context = get_script_run_ctx()
        assert context is not None
        registries.append(
            tuple(
                Path(page["script_path"]).resolve()
                for page in context.pages_manager.get_pages().values()
            )
        )
        return selected

    patch.object(st, "navigation", navigation).start()
    values = json.loads(query_file.read_text())
    if scenario == "retention":
        values["tenure"] = 12
        values["TotalCharges"] = 840.0
    at = AppTest.from_file(str(root / "apps/streamlit_llm/app.py"), default_timeout=25)
    at.session_state["active_scoring_context"] = {"protected": True}
    at.run()
    assert not at.exception
    if scenario in {"untrusted_artifact", "missing_artifact", "v1_rejected"}:
        assert at.error and not prompts
        assert llm_input.RESOURCES_KEY not in at.session_state
        assert "Không thể nạp dữ liệu tham chiếu hợp lệ" in at.error[0].value
        assert str(root) not in at.error[0].value
    else:
        assert not at.error and len(at.selectbox) + len(at.number_input) == 19
        store = at.session_state[llm_input.RESOURCES_KEY].store
        assert (
            store.manifest.format_version == 2
            and store.reference_version.startswith("rag-v2-")
        )
        at.run()
        assert len(loads) == 1 and not prompts
        if scenario == "lifecycle":
            at.button[0].click().run()
            assert at.error and not prompts
            assert llm_input.VALIDATED_INPUT_KEY not in at.session_state
        for field in at.selectbox:
            assert field.key is not None
            field.select(values[field.key.removeprefix("llm_input.feature.")])
        for number in at.number_input:
            assert number.key is not None
            number.set_value(values[number.key.removeprefix("llm_input.feature.")])
        at.button[0].click().run()
        assert not at.exception
        if scenario in {"invalid_response", "missing_keys"}:
            assert at.error and llm_input.RESULT_KEY not in at.session_state
            assert len(prompts) == (2 if scenario == "invalid_response" else 0)
            assert "SYNTHETIC_INVALID_RESPONSE" not in at.error[0].value
        else:
            assert not at.error and at.metric[0].value == "60%" and len(prompts) == 1
            result = at.session_state[llm_input.RESULT_KEY]
            assert (
                result.reference_count == 5
                and result.reference_version == store.reference_version
            )
            assert result.provider == "Groq" and result.probability_calibrated is False
            assert any(result.reference_version in item.value for item in at.text)
            snapshot = dict(at.session_state[llm_input.VALIDATED_INPUT_KEY])
            assert set(snapshot) == set(values)
            assert snapshot["TotalCharges"] == (values["TotalCharges"] or 0.0)
            payload = json.loads(
                prompts[0]
                .split("\nBEGIN_CUSTOMER_DATA_JSON\n", 1)[1]
                .split("\nEND_CUSTOMER_DATA_JSON\n", 1)[0]
            )
            assert set(payload["customer_to_predict"]) == set(values)
            assert len(payload["reference_customers"]) == 5
            assert not any(record.customer_id in prompts[0] for record in store.records)
            at.run()
            assert len(prompts) == 1 and len(loads) == 1
            if scenario == "retention":
                from churn_retention.entrypoints.streamlit.llm_prediction_record import (
                    RECORD_KEY,
                )

                accepted = at.session_state[RECORD_KEY]
                at.switch_page("retention_page.py").run()
                assert not at.exception and not at.error
                assert any("DỮ LIỆU GIẢ" in item.value for item in at.info)
                assert len(prompts) == 1
                at.radio[0].set_value("Kết quả LLM vừa gửi").run()
                assert not at.exception and not at.error and not at.metric
                assert at.selectbox[0].value is None  # Never infer status from label=1.
                at.selectbox[0].select("Đang hoạt động").run()
                assert (
                    not at.exception
                    and not at.error
                    and len(at.dataframe[0].value) == 1
                )
                at.number_input(key="llm_retention.budget").set_value(0.0).run()
                assert not at.exception and not at.error
                assert set(at.dataframe[0].value["action_code"]) == {"A0"}
                at.run()
                assert len(prompts) == 1 and at.session_state[RECORD_KEY] is accepted
                at.switch_page("../streamlit/pages/2_Predict.py").run()
                assert not at.exception and len(prompts) == 1
                for field in at.selectbox:
                    assert field.key is not None
                    field.select(values[field.key.removeprefix("llm_input.feature.")])
                for number in at.number_input:
                    assert number.key is not None
                    number.set_value(
                        values[number.key.removeprefix("llm_input.feature.")]
                    )
                at.button[0].click().run()
                assert not at.exception and not at.error and len(prompts) == 2
                assert at.session_state[RECORD_KEY].record_id != accepted.record_id
                assert (
                    prompts[0] == prompts[1]
                )  # No business metadata entered the prompt.
                at.switch_page("retention_page.py").run()
                at.radio[0].set_value("Kết quả LLM vừa gửi").run()
                assert not at.error and at.selectbox[0].value is None
                os.environ["CHURN_LLM_REFERENCE_SHA256"] = "0" * 64
                at.radio[0].set_value("Kết quả LLM vừa gửi").run()
                assert at.error and not at.metric and len(prompts) == 2
            if scenario == "lifecycle":
                # Form drafts only reach the server at submit; simulate a draft
                # already observed by the server, as the existing UI tests do.
                at.session_state["llm_input.feature.tenure"] = 1
                at.run()
                assert at.info and not at.metric and len(prompts) == 1
                assert dict(at.session_state[llm_input.VALIDATED_INPUT_KEY]) == snapshot
                at.button[0].click().run()
                assert at.error and len(prompts) == 1
                assert llm_input.RESULT_KEY not in at.session_state
                at.number_input(key="llm_input.feature.TotalCharges").set_value(22.0)
                at.button[0].click().run()
                assert not at.error and len(prompts) == 2
                os.environ["CHURN_LLM_REFERENCE_SHA256"] = "0" * 64
                at.run()
                assert at.error and llm_input.RESOURCES_KEY not in at.session_state
                assert (
                    llm_input.RESULT_KEY not in at.session_state and len(prompts) == 2
                )
                os.environ["CHURN_LLM_REFERENCE_SHA256"] = pin
                at.run()
                assert not at.error and len(loads) == 3
                previous = at.session_state[llm_input.RESOURCES_KEY].store
                artifact = llm_input.reference_artifact_path()
                replacement = artifact.with_suffix(".replacement")
                replacement.write_bytes(artifact.read_bytes())
                replacement.replace(artifact)
                at.run()
                assert not at.error and len(loads) == 4
                assert at.session_state[llm_input.RESOURCES_KEY].store is not previous
                assert len(prompts) == 2
    expected_page = (root / "apps/streamlit/pages/2_Predict.py").resolve()
    expected_retention = (root / "apps/streamlit_llm/retention_page.py").resolve()
    assert registries and all(
        pages == (expected_page, expected_retention) for pages in registries
    )
    if scenario != "project":
        for path in ("pages/1_Explain.py", "pages/scoring_page.py"):
            try:
                at.switch_page(path)
            except ValueError:
                pass
            else:
                raise AssertionError("Unregistered legacy page became navigable")
    assert at.session_state["active_scoring_context"] == {"protected": True}
    assert not any(
        name == p or name.startswith(p + ".") for name in sys.modules for p in blocked
    )
    assert sys.modules[poison.__name__] is poison and not hasattr(poison, "__file__")
    for name, module in tuple(sys.modules.items()):
        module_file = getattr(module, "__file__", None)
        if name.startswith("churn_retention") and isinstance(module_file, str):
            assert Path(module_file).resolve().is_relative_to(root / "src")
    print(
        json.dumps(
            {
                "scenario": scenario,
                "registered_pages": 2,
                "legacy_imports": 0,
                "mock_llm_calls": len(prompts),
            }
        )
    )


if __name__ == "__main__":
    main()
