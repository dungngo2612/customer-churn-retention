# Independent Reference Artifact v2 — step 7.5

The local artifact is `artifacts/references/telco-train-v2.zip`, reference version
`rag-v2-dcbf9bbad91d309d0b84`. It is Git-ignored, 389,593 bytes and mode 0600.
It contains internal customer data; do not commit or publicly distribute it.
No model bundle, classifier, pickle, joblib or API credential is embedded.

## Layout and manifest

```text
telco-train-v2.zip
  manifest.json
  schema.json
  preprocessing/state.json
  preprocessing/fit_ids.json
  records.jsonl
  matrix.npy
  splits.csv
```

The strict manifest contains format version 2, reference version, schema version
and SHA-256, preprocessing encoding version and SHA-256, source train/split hashes,
record count, 19 input features, vector dimension, ordered output names and their
hash, float64 dtype and matrix-value hash, exact library versions, each payload's
SHA-256, and train-only fit provenance. Provenance records fit count, original
train order hash, train membership hash, target_used=false, and source hashes.
The fit-ID payload permits verifying both membership and original fit order.

Reference versions are deterministic `rag-v2-` hashes of the payload checksums and
library versions. They cannot equal the v1 reference version. No logistic
model_type, model_version or bundle_sha256 is required or stored.

## Builder and fitted-state encoding

`workflows/build_reference_v2.py` reads processed train and split sources, fits
the independent RAGPreprocessor on exactly train IDs, and checks the source bytes
did not change during fitting. Labels are validated No/Yes and retained only in
reference records, never passed to preprocessing. Records are canonical and sorted
by internal ID; the fitted statistics still use original processed-train order.

The builder stores all records, the freshly transformed matrix, schema, split
manifest, fit order and fitted preprocessing. `_preprocessing_state.py` encodes
the feature-only ColumnTransformer/Pipeline/Imputer/Scaler/OneHotEncoder state as
JSON. Arrays have bounded dtype/shape tags. Only five installed local classes and
fixed state-field allowlists can be restored. NumPy dtype/scalar/slice tags use
fixed local decoding rules. No artifact-controlled module lookup, eval, pickle,
__setstate__ or arbitrary constructor is used. State restoration does not fit.

The codec is intentionally tied to the recorded exact sklearn/NumPy/pandas
versions. Unsupported versions, state fields, topology, recipe parameters,
statistics or vocabulary are rejected rather than approximated. JSON numeric
round-tripping retains the existing float64 values exactly on the checked data.

ZIP entries use fixed timestamps, permissions and sorted names. Repeated builds
from the same bytes/environment produce identical archive bytes. Publication is
atomic and exclusive using a same-filesystem hard link; existing outputs are
never overwritten, including an atomic publication race. The generated temporary
archive is fully loaded/validated before publication.

## Loader security and trust boundary

`VerificationReferenceV2` is a separate offline verification loader. It does not
replace ReferenceStore.load or select a new runtime path. Callers must explicitly
set `trusted=True` for a known local source or provide an externally trusted
`expected_artifact_sha256`; loading without either is rejected.

Checksums establish integrity, not authenticity. An expected hash computed from
an unknown file itself is not an external trust decision. An attacker can rewrite
both data and manifest hashes; only use approved sources or a hash delivered by a
trusted channel. Even with data-only JSON, provenance claims require source trust.

The loader:

1. Bounds compressed file size and the expanded archive before reading payloads.
2. Requires exactly seven known paths, rejecting extra/missing/duplicate entries,
   path traversal, symlinks, encryption, unsupported compression and excessive
   compression ratios. It never extracts archive files.
3. Checks format, exact library compatibility and every payload checksum before
   restoring preprocessing or decoding NPY data.
4. Validates schema/order, train-only split membership, fit population/order
   hashes, sources, fitted recipe, finite statistics and encoder vocabulary.
5. Reads the bounded NPY header before allocating/loading the matrix. Shape,
   float64 dtype, C order, exact byte size, finite values and value hash must agree.
   NumPy loading always uses allow_pickle=False.
6. Validates record count/order, IDs, labels and exactly 19 canonical features.
   It retransforms all records once and requires exact matrix alignment.
7. Returns read-only matrix views and ID-free prompt projections. Parser or
   transformer errors are sanitized to exclude customer data and local paths.

Limits are 64 MiB total expanded data, at most 65 MiB compressed archive, seven
entries, 200:1 compression ratio, 100,000 rows, 1,024 output columns and 2,000,000
matrix elements. Entry limits include 64 KiB manifest, 16 KiB schema, 1 MiB fitted
JSON, 8 MiB fit IDs/splits each, and 32 MiB records/matrix each. State decoding
also caps nesting at 32, nodes at 40,000 and individual arrays at 16,384 elements.
These limits are restrictive by design; oversized legitimate inputs also fail.

## v1/v2 equivalence

The comparison reads only v1 JSON records/manifest and its NPY oracle, never its
model.pkl. Actual vectors come from fitted state restored from v2 without refit.

| Check | Result |
| --- | --- |
| Records, features, labels, internal IDs and order | Exact, 4,930 records |
| Train/split source hashes | Exact |
| Restored fitted parameters/statistics/vocabulary/output names | Exact |
| Matrix | 4,930 x 45, float64, byte-for-byte exact |
| Matrix maximum absolute error | 0.0 |
| Matrix SHA-256 | f4087f1b797b512c63ed79cbaba9942891b67e244c4f24453d6832df4a6cdf47 |
| Fixed query vectors | 8/8 exact |
| Top-5 neighbor IDs/order | 8/8 exact; IDs stay internal |
| Ordered examples and complete prompt hashes | 8/8 exact, five examples each |
| Tie-breaking and self-exclusion | Controlled synthetic boundary tests pass |

The recorded comparison observed a 1.9984014443252818e-15 Euclidean-distance
difference for zero_tenure; other query distances in that run were exact.
Matrix/query bytes and retrieval method bodies are identical, so this is numeric
rounding in distance evaluation, not a changed transformation or neighbor order.
It is below the existing 1e-12 golden tolerance. Baseline and algorithm were not
modified to suppress it. Prompt bytes remain exactly equal in all eight cases.

Archive SHA-256:
`c85e61be7edfd0e91b248e0e353088677a345a378e64840b1613b51194b40bf8`.
The source records and neighbor identities are deliberately absent from this report.

## Independence, tests and runtime boundary

Fresh-process tests block old ML/training, v1 Store/artifact loader/builder,
predictor adapters, model files and artifact v1. They build v2, then disable all
preprocessing fit methods before loading, query transformation, retrieval and
prompt generation. Full matrix, eight query, top-5 and prompt hashes still match.
The actual sklearn logistic implementation is unavailable via an import barrier
and non-instantiable poison exports, as in step 7.4. sklearn.compose's unrelated
LinearRegression namespace import remains a library-internal side effect.

The existing Retriever had a runtime import of ReferenceStore solely for its
type annotation. That import is now TYPE_CHECKING-only, with a forward annotation;
all retrieval method bodies remain AST-identical. This enables the existing
algorithm to verify v2 without importing the old Store. No old pending change was
discarded. No v1 Store, v1 builder, Predictor or Streamlit file was edited.

Tests cover deterministic generation, manifest/schema/recipe validation, every
checksum/missing payload, corrupted archives, invalid matrix types/shapes/values,
NPY allocation bounds, unsafe ZIP entries, JSON/class decoding attacks, train-only
provenance, trust/pin enforcement, no refit/unpickle, golden behavior and isolated
build/load/retrieval. Existing golden fixtures and capture script stay immutable.

```bash
PYTHONPATH=src python3 -m churn_retention.workflows.build_reference_v2 --out artifacts/references/telco-train-v2.zip
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m scripts.verify_reference_v2
```

The first command refuses the existing artifact; select a new path for any future
build. No automatic baseline refresh or runtime migration is provided. The
existing runtime still selects v1 and retains its model-bundle dependencies.
Runtime migration is the next separately authorized step, not part of this change.

## Verification on 2026-11-01

- 80 new v2 cases passed, including isolated build/load/query/retrieval/prompt.
- Required v2, preprocessing, golden, reference/retrieval, prompt, predictor and
  Streamlit LLM group: 460 passed. This includes 48 unchanged golden tests and
  all 57 unchanged RAGPreprocessor tests.
- Full regression suite: 926 passed, 15 warnings, no skipped tests.
- Ruff check/format and Mypy: all seven new/changed Python files passed.
- All test/verification runs used socket-blocking guards; no real API was called.
- Of 169 files present before this task, only nearest_neighbors.py changed, with
  its method bodies proven AST-identical. All protected code/data, v1 artifacts,
  golden files, capture script and other pending changes retain their hashes.
- Branch and HEAD remain feat/ml-batch-scoring and
  84109d9178acf5d11d3d49770eb22568c9f958d4. There was no commit, push, fetch,
  reset, stash, branch switch, merge or rebase.
- Seven new code/documentation files are untracked. The additional v2 ZIP remains
  ignored and private; .gitignore and pyproject.toml were not edited.
