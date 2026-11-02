# LLM/RAG v1 golden baseline

This freezes the existing behavior before changing preprocessing or artifact
architecture. It does not train a classifier, refit preprocessing, rebuild the
reference artifact, or call an LLM. The source HEAD at capture was
`84109d9178acf5d11d3d49770eb22568c9f958d4`.

## Files and privacy

- `tests/fixtures/llm_rag_v1_golden.json`: versionable baseline with feature schema,
  vocabulary, aggregate fitted state, configuration, distances, reference labels,
  structural observations, and SHA-256 digests. It contains no raw customer IDs,
  query rows, query vectors, full prompts, API keys, or validation labels.
- `artifacts/golden/llm-rag-v1-private.json`: Git-ignored internal observations,
  mode `0600`, with eight actual transformed vectors and their ordered neighbor
  IDs. Do not commit, publish, or paste this file into reports.
- `scripts/capture_llm_rag_golden.py`: offline one-time capture using the existing
  Store, Retriever, and Prompt Builder. Both output files use exclusive creation;
  an existing snapshot is never overwritten.
- `tests/test_llm_rag_golden.py`: regression checks independent of the existing
  synthetic training/bundle fixtures.

The private file is a local audit companion. Public hashes independently check
all query vectors and ordered neighbor IDs even when that file is unavailable.
Hashes are integrity checks, not anonymization of a privately supplied dataset;
the fixture should still be distributed only with the project's authorization.

## Frozen behavior

The source is `artifacts/references/telco-train-v1.zip`, reference version
`ref-d56bcf40bc5cc5b3452d`. Artifact and validation CSV bytes are fingerprinted.
Neither source file is modified or copied into the public fixture.

The baseline records all 19 ordered input descriptors and 45 output names,
categorical vocabulary, transformer order, sparse-output behavior, imputer
parameters/statistics, scaler parameters/mean/variance/scale/sample count, and
OneHotEncoder parameters/categories/drop state. Public fitted statistics are
aggregate values, not individual customer rows. Query vectors and the entire
4,930-by-45 reference matrix are checked as float64 values in canonical
little-endian C order. Reference row order is fingerprinted independently.

Eight fixed validation positions cover the first row, zero tenure, no phone,
no internet, senior customers, one-year and two-year contracts, and maximum
TotalCharges. Positions are frozen in the JSON. Selection used feature values
only; the query loader reads only the 19 feature columns, excluding both
customerID and Churn. No validation labels are used to choose neighbors or build
prompts. Synthetic poisoned metadata additionally verifies label independence.

For each query the baseline stores the cleaned-feature hash, vector hash,
shape/dtype, ordered neighbor-ID-list hash, five Euclidean distances, complete
UTF-8 prompt hash, prompt length, actual few-shot count, reference-label order,
ordered-example hash, template hash, data-block structure, and output schema.
The current default uses five examples and a 16,000-character budget. The full
prompts remain transient in memory and contain no reference IDs.

Sixteen probes freeze accepted missing TotalCharges representations and errors
for nonzero-tenure missing charges, missing numeric/category values, unknown
categories, booleans, fractional integers, negative/nonfinite numbers, missing or
extra fields, and reordered input keys. Missing TotalCharges becomes zero only
for zero tenure. Other missing numeric values and missing/unknown categories are
rejected before imputation. Learned imputer state is still recorded separately.

A separate controlled tie case uses six identical vectors and shuffled synthetic
IDs, testing ascending ID order across the K=5 boundary and query-ID exclusion.
It uses the saved preprocessor and never fits a new one. Numeric vector and
matrix hashes, IDs, order, labels, and prompt bytes require exact equality.
Only real-query distances use `rtol=1e-12, atol=1e-12` for numeric-kernel noise.

## Running the checks

Run from the repository root with the original local assets available:

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CHURN_GOLDEN_REQUIRE_ASSETS=1 python3 -m pytest tests/test_llm_rag_golden.py -q
MYPYPATH=src python3 -m mypy --explicit-package-bases scripts/capture_llm_rag_golden.py tests/test_llm_rag_golden.py
python3 -m ruff check scripts/capture_llm_rag_golden.py tests/test_llm_rag_golden.py
python3 -m ruff format --check scripts/capture_llm_rag_golden.py tests/test_llm_rag_golden.py
```

The golden test fixture blocks network connections, preprocessing fit methods,
and Pipeline classifier inference throughout Store loading and test execution.
It also verifies that fitted preprocessing is not mutated. Neighbor-index fitting
is allowed; it indexes existing vectors without training preprocessing.

Without the real artifact or validation CSV, tests skip explicitly, matching the
project's existing local-asset integration policy. Strict mode above turns missing
assets, including the private companion, into a failure. Provision the existing
assets in CI and require strict mode for a release/refactor gate; a skipped run
does not certify equivalence. The private companion may be transferred only
through an approved internal channel with its restrictive permissions preserved.

Do not refresh expected results to make a regression pass. To intentionally
capture a separately reviewed baseline, use new public and Git-ignored private
output paths via `--public` and `--private`; default paths refuse overwriting.
The capture command uses the original v1 artifact and no fit methods or network.

## Applying this baseline to the later refactor

Keep source-provenance checks for the immutable v1 oracle. Separately feed the
candidate Store/preprocessor into the same behavioral comparisons; do not expect
a v2 archive's byte hash or legacy estimator class names to equal v1. No v2 path
is implemented in this step. Internal preprocessing structure may change only
with explicit review; output names, values, neighbor IDs/order, reference labels,
and prompt hashes must remain frozen unless a behavior change is approved.

Changes to fit-row order, cleaning before fitting, category ordering, dtype,
library versions, sparse/dense conversion, or float summation can change vectors
and nearest-neighbor boundaries. Keep the fit population train-only and separate
from the validation queries. Prompt byte hashes also detect whitespace, escaping,
instruction, schema, and example-order changes; do not weaken them casually.

This baseline checks deterministic behavior, not predictive quality, calibration,
or whether the original saved preprocessing was fitted exclusively on train IDs.
It does not repair the existing dependency on the Logistic Regression bundle.

## Verification on 2026-10-31

- Golden module: 48 cases, all executed in strict asset mode.
- Golden, retrieval, prompt, predictor, and Streamlit LLM group: 323 passed.
- Full test suite: 757 passed, 15 warnings, no skipped tests.
- Ruff check and formatting for `src apps tests scripts`: passed (109 files).
- Mypy for both new Python files with `MYPYPATH=src` and
  `--explicit-package-bases`: passed.
- Whole-repository Ruff remains blocked by pre-existing import ordering in
  `notebooks/explore_and_split.py`; whole-repository formatting also reports that
  notebook and `docs/llm_predictor.md`, `docs/reference_retrieval.md`. Those files
  were deliberately preserved.
- Capture and test runs used an external socket-blocking `sitecustomize` in
  addition to the golden module's network/fitting guard. No real API was called.

The pre-existing dirty files and all protected data/artifact files were compared
against their pre-task SHA-256 fingerprints. Branch and HEAD were also checked
before and after. No existing source file or asset was changed by this task.
