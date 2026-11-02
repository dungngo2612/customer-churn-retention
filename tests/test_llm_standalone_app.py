"""Launch the independent app with synthetic v2 data and no legacy deployment."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from llm_v2_fixtures import make_source

ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "tests/llm_standalone_probe.py"
RUNTIME_FILES = (
    "adapters/llm/base.py",
    "adapters/retrieval/reference_store.py",
    "adapters/retrieval/reference_v2.py",
    "adapters/retrieval/_reference_v2_format.py",
    "adapters/retrieval/_preprocessing_state.py",
    "adapters/retrieval/rag_preprocessor.py",
    "adapters/retrieval/nearest_neighbors.py",
    "contracts/reference.py",
    "contracts/reference_source.py",
    "contracts/llm.py",
    "contracts/llm_failure.py",
    "contracts/application_security.py",
    "contracts/llm_prediction_record.py",
    "entrypoints/streamlit/llm_input.py",
    "entrypoints/streamlit/llm_prediction_service.py",
    "entrypoints/streamlit/llm_prediction_record.py",
    "entrypoints/streamlit/llm_retention_presenter.py",
    "modules/retention/__init__.py",
    "modules/retention/policy.py",
    "modules/retention/use_cases.py",
    "modules/retention/budget.py",
    "modules/retention/llm_adapter.py",
    "modules/retention/demo.py",
    "modules/scoring/llm_predictor.py",
    "modules/scoring/llm_prompt.py",
    "modules/scoring/llm_response.py",
    "modules/scoring/llm_router.py",
)


@pytest.fixture(scope="module")
def deployment(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, str]:
    workspace = tmp_path_factory.mktemp("standalone-llm")
    source = make_source(workspace / "synthetic-source", n=8)
    source.build()
    root = workspace / "app"
    package = Path("src/churn_retention")
    files = {package / relative for relative in RUNTIME_FILES}
    for path in tuple(files):
        for parent in path.parents:
            if parent == Path("src"):
                break
            files.add(parent / "__init__.py")
    files.update(
        {
            Path("apps/streamlit_llm/app.py"),
            Path("apps/streamlit_llm/retention_page.py"),
            Path("apps/streamlit/pages/2_Predict.py"),
        }
    )
    for path in files:
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / path, target)
    artifact = root / "artifacts/references/telco-train-v2.zip"
    artifact.parent.mkdir(parents=True)
    shutil.copy2(source.output, artifact)
    pin = hashlib.sha256(artifact.read_bytes()).hexdigest()
    values = source.frame.iloc[0].drop(["customerID", "Churn"]).to_dict()
    values["TotalCharges"] = None
    query = workspace / "synthetic-query.json"
    query.write_text(json.dumps(values))
    # Existing but unregistered pages must not become navigable or executable.
    pages = root / "apps/streamlit_llm/pages"
    pages.mkdir()
    for name in ("1_Explain.py", "scoring_page.py"):
        (pages / name).write_text('raise AssertionError("Legacy page executed")\n')
    assert not (root / "src/churn_retention/ml").exists()
    assert not (root / "src/churn_retention/workflows").exists()
    assert not (root / "artifacts/bundles").exists()
    assert not (root / "artifacts/references/telco-train-v1.zip").exists()
    assert not (root / "apps/streamlit/scoring_page.py").exists()
    return root, query, pin


def run_probe(root: Path, query: Path, pin: str, scenario: str, cwd: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-I", str(PROBE), str(root), str(query), pin, scenario],
        cwd=cwd,
        env={**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"},
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)
    assert report["scenario"] == scenario
    assert report["legacy_imports"] == 0 and report["registered_pages"] == 2


@pytest.mark.parametrize(
    "scenario",
    [
        "lifecycle",
        "retention",
        "invalid_response",
        "missing_keys",
        "untrusted_artifact",
        "missing_artifact",
        "v1_rejected",
    ],
)
def test_standalone_in_clean_process_without_legacy_or_shared_navigation(
    deployment: tuple[Path, Path, str], tmp_path: Path, scenario: str
) -> None:
    root, query, pin = deployment
    run_probe(root, query, pin, scenario, tmp_path)


def test_project_v2_default_with_mock_llm_in_clean_process(
    deployment: tuple[Path, Path, str], tmp_path: Path
) -> None:
    if not (ROOT / "artifacts/references/telco-train-v2.zip").is_file():
        if os.environ.get("CHURN_GOLDEN_REQUIRE_ASSETS") == "1":
            pytest.fail("Approved local v2 artifact required in strict mode")
        pytest.skip("Local v2 artifact absent; synthetic standalone cases still run")
    run_probe(ROOT, deployment[1], "", "project", tmp_path)
