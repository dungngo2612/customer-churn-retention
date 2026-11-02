# LLM/RAG runtime migration — step 7.6

The Predict page now uses the approved reference artifact v2 by default. It restores
fitted feature preprocessing from JSON, transforms the same 19 inputs, retrieves
five examples and invokes the existing prompt/predictor/router/response-validation
flow. It never fits preprocessing, loads a classifier or falls back to artifact v1.
This document supersedes the runtime-boundary statements in the historical step
7.5 report; the artifact format and builder from that step are unchanged.

## Runtime and compatibility

```mermaid
flowchart TD
    UI[Streamlit Predict: 19 inputs] --> Validation[Validation and submitted snapshot]
    Config[Configured v2 path and external SHA-256 pin] --> Store[Public ReferenceStore]
    Store --> Loader[Existing reference_v2 JSON loader]
    Loader --> JSON[Fitted RAG preprocessing JSON]
    Loader --> Data[Validated records and reference matrix]
    Validation --> Store
    Store --> Retrieval[Unchanged Euclidean retrieval: K=5]
    Retrieval --> Prompt[Unchanged Prompt Builder]
    Prompt --> Predictor[Existing LLMChurnPredictor and Router]
    Predictor --> Response[Response Validator]
    Response --> Result[Existing Streamlit result rendering]
```

`reference_store.py` exports the existing `VerificationReferenceV2` implementation
as `ReferenceStore`, without another Store class or an import of a v1 loader.
`load`, `prepare_query`, `clean_features`, `records`, `matrix`, `feature_schema`
and `reference_version` remain available. Optional reference-version and dataset
hash filters are supported. Historical model-bundle filters are available only on the archive branch.
V2 errors inherit the neutral `ReferenceArtifactError` for existing callers.

Classifier-backed v1 loading/building and golden capture were removed from main.
Their historical implementation is preserved on [codex/archive-self-training-2026-11-02](https://github.com/dungngo2612/customer-churn-retention/tree/codex/archive-self-training-2026-11-02).
Immutable v1 JSON/NPY oracles remain for comparison; current tests load v2 and do
not import or deserialize a classifier. Retrieval uses the neutral ReferenceSource
protocol. Providers, models, router and timeouts remain unchanged.

## Artifact configuration and cache

Default artifact: `artifacts/references/telco-train-v2.zip`.

Default approved archive SHA-256:
`c85e61be7edfd0e91b248e0e353088677a345a378e64840b1613b51194b40bf8`.

Reference version: `rag-v2-dcbf9bbad91d309d0b84`.

Administrators can configure `CHURN_LLM_REFERENCE_ARTIFACT` and
`CHURN_LLM_REFERENCE_SHA256`. A replacement artifact needs its own externally
approved hash. Do not derive that approval from an unknown file or its manifest.
Blank paths, malformed pins and mismatches are rejected. The Streamlit runtime
always passes the external pin, never `trusted=True`.

The v2 loader reads one bounded byte snapshot and authenticates those exact bytes
before ZIP parsing. It verifies manifest, payload checksums, exact library
versions, feature schema, fitted preprocessing, train-only provenance, matrix
shape/dtype/value hash and record alignment. No pickle or classifier is required.
The records are retransformed for alignment validation, without fitting.

Session caching uses resolved path, device/inode, size, modification/change times
and the configured pin. The identity is checked again after loading. A changed
file/path/pin invalidates cached LLM resources, snapshots and results. This cheap
stat fingerprint is cache invalidation, not artifact authentication. Re-loading
always verifies the pin. Failure renders a sanitized error and stops before the
Router; it never tries an old bundle. `active_scoring_context` is untouched.

The existing form groups, 19 features, validation, loading spinner, submitted
snapshot and result rendering remain. Reference version displays the v2 value.
No internal neighbor IDs enter prompts or public reports.

## Scope of independence

The Predict page and its LLM/RAG call chain no longer import the legacy Store,
`_artifact.py`, training pipeline or bundle predictor. They require neither
`model.pkl`, a model bundle, `feature_schema.json` from that bundle nor v1 ZIP.
No LogisticRegression object is instantiated or called.

The unrelated shared Streamlit root `apps/streamlit/scoring_page.py` still imports
`BundlePredictor`, and the project keeps its combined dependency/package layout.
Starting that shared root can still import legacy training code. This task does
not make the entire Streamlit application/distribution independent: separating
its entrypoint is explicitly out of scope. Scoring, Explain, LR, training, batch
and SHAP implementations and `pyproject.toml` are unchanged.

Scikit-learn remains necessary for feature transformers. Its `sklearn.compose`
import currently reaches the `sklearn.linear_model` namespace through an unrelated
LinearRegression import. Isolated tests make the actual logistic implementation
unavailable and replace its namespace exports with non-instantiable sentinels.
This demonstrates absence of a classifier dependency without claiming that no
library namespace mentions logistic regression.

## Regression coverage

The new runtime tests cover:

- Public Store reuse, explicit trust, version/source filters and no v1 fallback.
- External pin authentication of the exact byte snapshot, even if the source path
  changes after hashing.
- Full matrix and all eight fixed vectors, ordered top-five IDs, distances,
  ordered examples and complete prompt hashes against the unchanged v1 baseline.
- Deterministic tie-breaking and self-exclusion with controlled synthetic IDs.
- Default v2 form, cache reuse/invalidation, snapshot, result/reference version
  rendering and preservation of `active_scoring_context`.
- Missing artifact, payload checksum tampering, wrong schema, wrong matrix shape,
  unapproved source/hash, malformed pin and explicit v1 rejection before Router.
- Fresh-process full UI integration for both valid and invalid LLM responses:
  validation, v2 load, retrieval, prompt, real Router with mock providers, response
  validation and rendering. Import/open barriers prohibit training, v1, old
  bundles, classifier implementation and classifier deserialization. Fit methods
  and network sockets are disabled; no provider SDK is loaded or API called.

The LLM unit fixtures now build small independent v2 artifacts from synthetic
train records; preprocessing receives only features. V1 tests keep the explicit
legacy loader as an oracle.
Original datasets, splits, artifacts and golden expected values are untouched.

## Current implementation

- `contracts/reference.py` / `reference_source.py`: neutral data contracts and protocol.
- `adapters/retrieval/rag_preprocessor.py`: classifier-free fit/transform for reference data.
- `_preprocessing_state.py`, `_reference_v2_format.py`, `reference_v2.py`: JSON codec and bounded loader.
- `workflows/build_reference_v2.py`: offline builder; no classifier source or bundle input.
- `entrypoints/streamlit/llm_input.py`: pinned resources and feature validation.
- `tests/test_reference_contracts.py`, `test_reference_v2.py`, `test_reference_retrieval.py`,
  `test_rag_preprocessor.py`, `test_llm_rag_golden.py`: current RAG and frozen-oracle regressions.

The step 7.x implementation inventory is preserved on the classifier archive branch.

## Equivalence and operational risks

The immutable baseline requires a 4,930 x 45 float64 reference matrix, eight
1 x 45 query vectors, five neighbors per query and eight complete prompt hashes.
Matrix/vector bytes and neighbor/example order must remain exact. Euclidean
distances use the existing absolute/relative tolerance of 1e-12; library/BLAS
rounding can differ around 2e-15 without changing neighbors or prompts. Tie order
and self-exclusion retain the original retrieval algorithm.

Artifact v2 and private baseline are intentionally Git-ignored and contain
internal records. Deployments must provision the approved v2 file separately;
missing files fail closed. A future rebuild changes the pin/version and requires
explicit approval plus golden verification. Exact sklearn/NumPy/pandas versions
are part of the format; upgrading them can require a separately reviewed artifact
and codec update. These restrictions are preserved rather than bypassed here.

Only runtime migration is implemented. No preprocessing v3, artifact rebuild,
entrypoint split, deployment, commit or push is performed.

## Verification on 2026-11-01

| Check | Result |
| --- | --- |
| New runtime v2 tests | 26 passed |
| Required runtime, v2 Store, preprocessing, retrieval, prompt, predictor, Streamlit and golden group | 486 passed |
| Full regression suite | 952 passed, 15 warnings, no skipped tests (197.64 seconds) |
| Full 4,930 x 45 float64 matrix and eight vectors | Exact hashes |
| Eight top-five neighbor ID/order comparisons | Exact hashes; IDs remain private |
| Euclidean distances | Existing atol/rtol 1e-12 satisfied |
| Eight complete prompts | Exact hashes, five examples each |
| Ties and self-exclusion | Passed |
| Fresh-process UI success/error integration | Both passed without old bundles/v1/classifier/training, fitting or network |
| Ruff check and format check, 21 Python files in this migration | Passed |
| Mypy, all 21 Python files in this migration | Passed |

Global `ruff check .` still reports the pre-existing unsorted import in
`notebooks/explore_and_split.py`. Global `ruff format --check .` still reports
that notebook and code fences in `docs/llm_predictor.md` and
`docs/reference_retrieval.md`. These three files retain their pre-task hashes;
unrelated pending work was not reformatted. The full test warnings come from
existing bundle-name/dirty-tree checks and third-party deprecations.

Tests used a socket-blocking guard and fake provider credentials/responses; no
real API was called. The independent subprocesses also install their own network
and legacy import/file barriers. Golden asset checks were required, not skipped.

Git branch remains `feat/ml-batch-scoring`; HEAD remains
`84109d9178acf5d11d3d49770eb22568c9f958d4`. Of 177 pre-existing files snapshotted
before this task, only the 17 files listed above changed. Protected data, split
manifest, both artifacts, model bundles, golden fixtures, LR/training/batch/SHAP,
Router/Providers, predictor/service, shared scoring entrypoint, Explain and
`pyproject.toml` retain their hashes. The v1 Store copy is byte-for-byte identical
to the pre-task Store. Existing dirty paths remain present. No Git write or
network operation was performed.
