"""Public runtime v2, fail-closed trust/cache behavior and isolated Predict UI."""

import io
import json
import os
import pickle
import subprocess
import sys
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, cast
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from churn_retention.adapters.retrieval import reference_v2
from churn_retention.adapters.retrieval._reference_v2_format import (
    ReferenceV2Error,
    reference_version,
)
from churn_retention.adapters.retrieval.nearest_neighbors import (
    SimilarCustomerRetriever,
)
from churn_retention.adapters.retrieval.reference_store import ReferenceStore
from churn_retention.contracts.reference import ReferenceRecord, json_bytes, sha256
from churn_retention.entrypoints.streamlit import llm_input, llm_prediction_service
from churn_retention.modules.scoring.llm_prompt import ChurnPromptBuilder
from churn_retention.modules.scoring.llm_router import LLMRouter

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "apps/streamlit/pages/2_Predict.py"
BASELINE = json.loads((ROOT / "tests/fixtures/llm_rag_v1_golden.json").read_text())
CASES = BASELINE["queries"]


def digest(value: object) -> str:
    return sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    )


@pytest.fixture(scope="module")
def store(approved_rag_reference: Path) -> ReferenceStore:
    return ReferenceStore.load(
        approved_rag_reference,
        expected_artifact_sha256=llm_input.DEFAULT_REFERENCE_SHA256,
    )


@pytest.fixture(scope="module")
def queries(store: ReferenceStore) -> dict[str, dict[str, object]]:
    frame = pd.read_csv(
        ROOT / "data/processed/val.csv",
        usecols=[column.name for column in store.feature_schema],
    )
    return {
        case["name"]: {
            column.name: frame.iloc[case["row"]][column.name] for column in store.feature_schema
        }
        for case in CASES
    }


def fill(at: AppTest, values: dict[str, object]) -> None:
    for control in at.selectbox:
        assert control.key is not None
        control.select(values[control.key.removeprefix("llm_input.feature.")])
    for number in at.number_input:
        assert number.key is not None
        number.set_value(
            cast(
                int | float | None,
                values[number.key.removeprefix("llm_input.feature.")],
            )
        )


@pytest.fixture
def app(monkeypatch: pytest.MonkeyPatch, queries: dict[str, dict[str, object]]):
    monkeypatch.delenv("CHURN_LLM_REFERENCE_ARTIFACT", raising=False)
    monkeypatch.delenv("CHURN_LLM_REFERENCE_SHA256", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "fake-no-network")
    monkeypatch.setenv("GEMINI_API_KEY", "fake-no-network")
    prompts: list[str] = []

    class Provider:
        async def generate(self, prompt: str) -> str:
            prompts.append(prompt)
            return json.dumps(
                {
                    "churn_prediction": 1,
                    "churn_probability": 0.6,
                    "reason": "Bằng chứng tham khảo còn hạn chế; ước lượng chưa chắc chắn.",
                },
                ensure_ascii=False,
            )

    @asynccontextmanager
    async def router():
        async with LLMRouter(
            Provider(), Provider(), primary_provider="Groq", fallback_provider="Gemini"
        ) as instance:
            yield instance

    monkeypatch.setattr(llm_prediction_service, "groq_gemini_router", router)
    loader = Mock(wraps=llm_input.load_form_resources)
    monkeypatch.setattr(llm_input, "load_form_resources", loader)
    at = AppTest.from_file(str(PAGE), default_timeout=25)
    at.session_state["active_scoring_context"] = {"protected": "existing context"}
    at.run()
    assert not at.exception and not at.error
    return at, loader, prompts, queries["first"]


def test_public_store_is_existing_v2_implementation_and_requires_trust(
    store: ReferenceStore,
) -> None:
    assert ReferenceStore is reference_v2.VerificationReferenceV2
    assert store.manifest.format_version == 2 and store.reference_version.startswith("rag-v2-")
    assert store.clean_features(store.records[0].features) == dict(store.records[0].features)
    with pytest.raises(ReferenceV2Error, match="trusted"):
        ReferenceStore.load(llm_input.DEFAULT_REFERENCE_PATH)
    for keyword in ("expected_reference_version", "expected_dataset_sha256"):
        with pytest.raises(ReferenceV2Error, match="requested"):
            ReferenceStore.load(
                llm_input.DEFAULT_REFERENCE_PATH, trusted=True, **{keyword: "wrong"}
            )


def test_public_store_never_falls_back_to_v1(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("The runtime must not unpickle a v1 bundle")

    monkeypatch.setattr(pickle, "loads", forbidden)
    with pytest.raises(ReferenceV2Error):
        ReferenceStore.load(ROOT / "artifacts/references/telco-train-v1.zip", trusted=True)


def test_external_pin_authenticates_the_exact_archive_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, approved_rag_reference: Path
) -> None:
    path = tmp_path / "race.zip"
    path.write_bytes(approved_rag_reference.read_bytes())
    read = reference_v2._read_payloads

    def swapped(data: bytes):
        assert sha256(data) == llm_input.DEFAULT_REFERENCE_SHA256
        path.write_bytes(b"untrusted replacement after hash")
        return read(data)

    monkeypatch.setattr(reference_v2, "_read_payloads", swapped)
    loaded = ReferenceStore.load(path, expected_artifact_sha256=llm_input.DEFAULT_REFERENCE_SHA256)
    assert loaded.manifest.format_version == 2
    assert path.read_bytes() == b"untrusted replacement after hash"


def test_runtime_matrix_matches_frozen_v1(store: ReferenceStore) -> None:
    assert store.matrix.shape == (4930, 45) and store.matrix.dtype == np.float64
    assert (
        sha256(np.asarray(store.matrix, dtype="<f8", order="C").tobytes())
        == BASELINE["matrix"]["values_sha256"]
    )
    assert (
        list(store.preprocessor.output_feature_names)
        == BASELINE["preprocessing"]["output_feature_names"]
    )
    assert store.preprocessor.metadata["preprocessing"] == BASELINE["preprocessing"]


@pytest.mark.parametrize("case", CASES, ids=[case["name"] for case in CASES])
def test_runtime_vectors_top_five_examples_and_prompts_match_golden(
    store: ReferenceStore, queries: dict[str, dict[str, object]], case: dict[str, Any]
) -> None:
    query = queries[case["name"]]
    prepared = store.prepare_query(query)
    assert prepared.vector.shape == (1, 45) and prepared.vector.dtype == np.float64
    assert (
        sha256(np.asarray(prepared.vector, dtype="<f8", order="C").tobytes())
        == case["vector_sha256"]
    )
    neighbors = SimilarCustomerRetriever(store).retrieve(query)
    assert digest([row.customer_id for row in neighbors]) == case["ordered_neighbor_ids_sha256"]
    np.testing.assert_allclose(
        [row.distance for row in neighbors], case["distances"], atol=1e-12, rtol=1e-12
    )
    built = ChurnPromptBuilder(store.feature_schema).build_with_metadata(
        prepared.features, neighbors
    )
    assert built.reference_count == 5 and sha256(built.text.encode()) == case["prompt"]["sha256"]
    if any(row.customer_id in built.text for row in store.records):
        pytest.fail("A private reference ID entered a prompt")


@pytest.mark.parametrize("exclude", [False, True])
def test_runtime_tie_breaking_and_self_exclusion(
    store: ReferenceStore, queries: dict[str, dict[str, object]], exclude: bool
) -> None:
    features = store.clean_features(queries["first"])
    records = tuple(
        ReferenceRecord(features, "No", "synthetic-" + suffix)
        for suffix in ("f", "b", "e", "a", "d", "c")
    )
    tied = ReferenceStore(
        store.manifest,
        store.preprocessor,
        records,
        store.preprocessor.transform_many([features] * 6),
    )
    rows = SimilarCustomerRetriever(tied).retrieve(
        features, customer_id="synthetic-c" if exclude else None
    )
    suffixes = ("a", "b", "d", "e", "f") if exclude else ("a", "b", "c", "d", "e")
    assert [row.customer_id for row in rows] == ["synthetic-" + suffix for suffix in suffixes]
    assert len({row.distance for row in rows}) == 1


def test_predict_page_uses_v2_caches_and_preserves_snapshot_and_scoring_context(
    app,
) -> None:
    at, loader, prompts, values = app
    before = pickle.dumps(at.session_state[llm_input.RESOURCES_KEY].store.preprocessor._transformer)
    assert len(at.selectbox) + len(at.number_input) == 19
    at.run()
    assert loader.call_count == 1 and not prompts
    fill(at, values)
    at.button[0].click().run()
    assert not at.exception and not at.error and at.metric[0].value == "60%"
    result = at.session_state[llm_input.RESULT_KEY]
    assert result.reference_version == "rag-v2-dcbf9bbad91d309d0b84"
    assert result.reference_count == 5 and result.probability_calibrated is False
    assert len(at.session_state[llm_input.VALIDATED_INPUT_KEY]) == 19
    assert at.session_state["active_scoring_context"] == {"protected": "existing context"}
    assert any(result.reference_version in item.value for item in at.text)
    at.run()
    assert len(prompts) == 1 and loader.call_count == 1
    assert (
        pickle.dumps(at.session_state[llm_input.RESOURCES_KEY].store.preprocessor._transformer)
        == before
    )


def test_changed_pin_clears_cached_store_result_and_never_uses_v1(
    app, monkeypatch: pytest.MonkeyPatch
) -> None:
    at, loader, prompts, values = app
    fill(at, values)
    at.button[0].click().run()
    assert llm_input.RESULT_KEY in at.session_state
    monkeypatch.setenv("CHURN_LLM_REFERENCE_SHA256", "0" * 64)
    at.run()
    assert at.error and not at.exception
    assert (
        llm_input.RESULT_KEY not in at.session_state
        and llm_input.RESOURCES_KEY not in at.session_state
    )
    assert len(prompts) == 1 and loader.call_count == 2
    assert at.session_state["active_scoring_context"] == {"protected": "existing context"}
    monkeypatch.delenv("CHURN_LLM_REFERENCE_SHA256")
    at.run()
    assert not at.error and llm_input.RESULT_KEY not in at.session_state and loader.call_count == 3


def test_changed_artifact_path_invalidates_cache_without_modifying_real_artifact(
    app, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    at, loader, _, _ = app
    previous = at.session_state[llm_input.RESOURCES_KEY].store
    copied = tmp_path / "approved-copy.zip"
    copied.write_bytes(llm_input.DEFAULT_REFERENCE_PATH.read_bytes())
    monkeypatch.setenv("CHURN_LLM_REFERENCE_ARTIFACT", str(copied))
    at.run()
    assert not at.exception and not at.error and loader.call_count == 2
    assert at.session_state[llm_input.RESOURCES_KEY].store is not previous
    assert at.session_state["active_scoring_context"] == {"protected": "existing context"}


def invalid_copy(directory: Path, kind: str) -> Path:
    target = directory / "invalid.zip"
    if kind == "missing":
        return target
    if kind == "v1":
        target.write_bytes((ROOT / "artifacts/references/telco-train-v1.zip").read_bytes())
        return target
    with zipfile.ZipFile(llm_input.DEFAULT_REFERENCE_PATH) as archive:
        payloads = {name: archive.read(name) for name in archive.namelist()}
    if kind == "checksum":
        payloads["records.jsonl"] += b"changed"
    if kind in {"schema", "shape"}:
        if kind == "schema":
            schema = json.loads(payloads["schema.json"])
            schema[0]["dtype"] = "float64"
            payloads["schema.json"] = json_bytes(schema)
        else:
            matrix = np.load(io.BytesIO(payloads["matrix.npy"]), allow_pickle=False)[:, :-1]
            buffer = io.BytesIO()
            np.save(buffer, matrix, allow_pickle=False)
            payloads["matrix.npy"] = buffer.getvalue()
        manifest = json.loads(payloads["manifest.json"])
        manifest["files"] = {
            name: sha256(value) for name, value in payloads.items() if name != "manifest.json"
        }
        manifest["schema_sha256"] = manifest["files"]["schema.json"]
        manifest["reference_version"] = reference_version(manifest["files"], manifest["libraries"])
        payloads["manifest.json"] = json_bytes(manifest)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, value in payloads.items():
            archive.writestr(name, value)
    target.chmod(0o600)
    return target


@pytest.mark.parametrize(
    "kind",
    ["missing", "checksum", "schema", "shape", "untrusted", "malformed_pin", "v1"],
)
def test_bad_v2_artifacts_fail_safely_before_router_and_preserve_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str, request: pytest.FixtureRequest
) -> None:
    if kind == "v1":
        request.getfixturevalue("frozen_rag_oracle")
    elif kind != "missing":
        request.getfixturevalue("approved_rag_reference")
    path = invalid_copy(tmp_path, kind)
    monkeypatch.setenv("CHURN_LLM_REFERENCE_ARTIFACT", str(path))
    pin = (
        "0" * 64
        if kind == "untrusted"
        else "invalid"
        if kind == "malformed_pin"
        else sha256(path.read_bytes())
        if path.exists()
        else llm_input.DEFAULT_REFERENCE_SHA256
    )
    monkeypatch.setenv("CHURN_LLM_REFERENCE_SHA256", pin)
    router = Mock(side_effect=AssertionError("No prediction before valid artifact loading"))
    monkeypatch.setattr(llm_prediction_service, "groq_gemini_router", router)
    at = AppTest.from_file(str(PAGE), default_timeout=25)
    at.session_state["active_scoring_context"] = {"protected": True}
    at.run()
    assert not at.exception and at.error
    assert "Không thể nạp dữ liệu tham chiếu hợp lệ" in at.error[0].value
    assert str(tmp_path) not in at.error[0].value
    assert (
        llm_input.RESOURCES_KEY not in at.session_state
        and llm_input.RESULT_KEY not in at.session_state
    )
    assert at.session_state["active_scoring_context"] == {"protected": True}
    router.assert_not_called()


@pytest.mark.parametrize("valid_response", [True, False])
def test_full_predict_ui_in_classifier_and_legacy_free_process(
    tmp_path: Path, valid_response: bool, approved_rag_reference: Path
) -> None:
    script = (
        "\n"
        "import importlib.abc, json, os, socket, sys, types\n"
        "from contextlib import asynccontextmanager\n"
        "sys.dont_write_bytecode = True\n"
        "sys.path.insert(0, sys.argv[1])\n"
        'blocked = ("sklearn.linear_model._logistic", "churn_retentio'
        'n.ml", "churn_retention.workflows", "churn_retention.adapter'
        's.predictor", "churn_retention.adapters.retrieval._artifact"'
        ', "churn_retention.adapters.retrieval.legacy_reference_store'
        '", "churn_retention.entrypoints.streamlit.explanation_views"'
        ")\n"
        "class Barrier(importlib.abc.MetaPathFinder):\n"
        " def find_spec(self,fullname,path=None,target=None):\n"
        '  if any(fullname == p or fullname.startswith(p+".") for p i'
        'n blocked): raise AssertionError("Legacy/classifier imports '
        'unavailable")\n'
        "class NoClassifier:\n"
        ' def __new__(cls,*args,**kwargs): raise AssertionError("Logi'
        'stic Regression unavailable")\n'
        'poison = types.ModuleType("sklearn.linear_model._logistic")\n'
        "poison.LogisticRegression = poison.LogisticRegressionCV = No"
        "Classifier\n"
        "sys.modules[poison.__name__] = poison\n"
        "sys.meta_path.insert(0,Barrier())\n"
        "def audit(event,args):\n"
        ' if event == "open" and isinstance(args[0],(str,bytes,os.Pat'
        "hLike)):\n"
        '  p = os.fsdecode(args[0]).replace("\\\\","/")\n'
        '  if p.endswith(".pkl") or p.endswith("telco-train-v1.zip") '
        'or "/artifacts/bundles/" in p or p.endswith("scoring_page.py'
        '"):\n'
        '   raise AssertionError("Old bundle/v1/scoring entrypoint un'
        'available")\n'
        ' if event == "pickle.find_class" and (str(args[0]).startswit'
        'h("sklearn") or str(args[0]).startswith("churn_retention.ml"'
        ")):\n"
        '  raise AssertionError("No unpickling ML state")\n'
        "sys.addaudithook(audit)\n"
        'def forbidden(*args,**kwargs): raise AssertionError("No netw'
        'ork, fitting or provider SDK")\n'
        "socket.socket.connect = socket.socket.connect_ex = socket.cr"
        "eate_connection = forbidden\n"
        'os.environ["GROQ_API_KEY"] = os.environ["GEMINI_API_KEY"] = '
        '"fake-no-network"\n'
        'os.environ.pop("CHURN_LLM_REFERENCE_ARTIFACT",None)\n'
        'os.environ.pop("CHURN_LLM_REFERENCE_SHA256",None)\n'
        "from churn_retention.entrypoints.streamlit import llm_input,"
        " llm_prediction_service as service\n"
        "from churn_retention.modules.scoring.llm_router import LLMRo"
        "uter\n"
        "from churn_retention.adapters.retrieval.rag_preprocessor imp"
        "ort RAGPreprocessor\n"
        "from sklearn.compose import ColumnTransformer\n"
        "from sklearn.impute import SimpleImputer\n"
        "from sklearn.preprocessing import StandardScaler,OneHotEncod"
        "er\n"
        "from sklearn.pipeline import Pipeline\n"
        "RAGPreprocessor.fit_train = forbidden\n"
        "for cls in (ColumnTransformer,SimpleImputer,StandardScaler,O"
        "neHotEncoder,Pipeline): cls.fit = cls.fit_transform = forbid"
        "den\n"
        "from streamlit.testing.v1 import AppTest\n"
        "import pandas as pd\n"
        "prompts = []\n"
        "class Provider:\n"
        " async def generate(self,prompt):\n"
        "  prompts.append(prompt)\n"
        '  return json.dumps({"churn_prediction":1,"churn_probability'
        '":0.6,"reason":"Bằng chứng tham khảo còn hạn chế; ước lượng '
        'chưa chắc chắn."},ensure_ascii=False) if sys.argv[4] == "val'
        'id" else "INVALID_RESPONSE"\n'
        "@asynccontextmanager\n"
        "async def router():\n"
        " async with LLMRouter(Provider(),Provider(),primary_provider"
        '="Groq",fallback_provider="Gemini") as instance: yield insta'
        "nce\n"
        "service.groq_gemini_router = router\n"
        "loads = []\n"
        "real_load = llm_input.load_form_resources\n"
        "def counted(path):\n"
        " loads.append(True)\n"
        " return real_load(path)\n"
        "llm_input.load_form_resources = counted\n"
        "at = AppTest.from_file(sys.argv[2],default_timeout=25)\n"
        'at.session_state["active_scoring_context"] = {"protected":Tr'
        "ue}\n"
        "at.run()\n"
        "assert not at.exception and not at.error\n"
        "store = at.session_state[llm_input.RESOURCES_KEY].store\n"
        "assert store.manifest.format_version == 2 and store.referenc"
        'e_version.startswith("rag-v2-")\n'
        "at.run()\n"
        "assert len(loads) == 1 and not prompts\n"
        "names = [c.name for c in store.feature_schema]\n"
        "frame = pd.read_csv(sys.argv[3],usecols=names)\n"
        "values = {name:frame.iloc[0][name] for name in names}\n"
        "assert len(at.selectbox)+len(at.number_input) == 19\n"
        "for control in at.selectbox: control.select(values[control.k"
        'ey.removeprefix("llm_input.feature.")])\n'
        "for control in at.number_input: control.set_value(values[con"
        'trol.key.removeprefix("llm_input.feature.")])\n'
        "at.button[0].click().run()\n"
        "assert not at.exception\n"
        'if sys.argv[4] == "valid":\n'
        " assert at.success and not at.error and at.metric[0].value ="
        '= "60%"\n'
        " result = at.session_state[llm_input.RESULT_KEY]\n"
        " assert result.reference_version == store.reference_version "
        "and result.reference_count == 5\n"
        " assert any(result.reference_version in item.value for item "
        "in at.text)\n"
        " assert len(at.session_state[llm_input.VALIDATED_INPUT_KEY])"
        " == 19\n"
        " assert len(prompts) == 1\n"
        "else:\n"
        " assert at.error and llm_input.RESULT_KEY not in at.session_"
        "state and len(prompts) == 2\n"
        'assert at.session_state["active_scoring_context"] == {"prote'
        'cted":True}\n'
        'payload = json.loads(prompts[0].split("\\nBEGIN_CUSTOMER_DATA'
        '_JSON\\n",1)[1].split("\\nEND_CUSTOMER_DATA_JSON\\n",1)[0])\n'
        'assert set(payload["customer_to_predict"]) == set(names) and'
        ' "Churn" not in payload["customer_to_predict"]\n'
        'assert len(payload["reference_customers"]) == 5\n'
        "assert not any(row.customer_id in prompts[0] for row in stor"
        "e.records)\n"
        "count = len(prompts)\n"
        "at.run()\n"
        "assert len(prompts) == count and len(loads) == 1\n"
        "assert sys.modules[poison.__name__] is poison and not hasatt"
        'r(poison,"__file__")\n'
        'assert not any(name == p or name.startswith(p+".") for name '
        "in sys.modules for p in blocked[1:])\n"
        'assert "churn_retention.adapters.llm.groq_provider" not in s'
        'ys.modules and "churn_retention.adapters.llm.gemini_provider'
        '" not in sys.modules\n'
        'print("independent full Predict validation/retrieval/router/'
        'response/UI passed")\n'
    )
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            script,
            str(ROOT / "src"),
            str(PAGE),
            str(ROOT / "data/processed/val.csv"),
            "valid" if valid_response else "invalid",
        ],
        cwd=tmp_path,
        env={**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"},
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    assert (
        result.stdout.strip()
        == "independent full Predict validation/retrieval/router/response/UI passed"
    )
