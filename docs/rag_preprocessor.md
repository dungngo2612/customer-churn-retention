# Independent RAG preprocessing — step 7.4

This adds a RAG-owned, feature-only preprocessor and offline verification. It does
not integrate it into ReferenceStore, Predictor, Streamlit, or artifact building.
There is no new artifact/persistence format or reference version in this step.

## Observed v1 preprocessing before implementation

The saved configuration/state was read from the frozen golden fixture and checked
against the original training feature-selection and transformer definitions.
The implementation never imports that training code or reads its bundle.

Input order is the four numeric features below followed by the 15 categorical
features in this order:

```text
gender, Partner, Dependents, PhoneService, MultipleLines, InternetService,
OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV,
StreamingMovies, Contract, PaperlessBilling, PaymentMethod
```

| Numeric feature | Median | Scaler mean | Population variance | Scale |
| --- | --- | --- | --- | --- |
| SeniorCitizen | 0.0 | 0.16105476673427993 | 0.13511612884644664 | 0.3675814587903566 |
| tenure | 29.0 | 32.47119675456389 | 607.3193529288333 | 24.643850205047777 |
| MonthlyCharges | 70.57499999999999 | 64.95008113590264 | 914.4942677418957 | 30.240606272723696 |
| TotalCharges | 1387.4 | 2301.797565922921 | 5254794.809592453 | 2292.3339219215977 |

Numeric preprocessing is SimpleImputer(strategy="median") then StandardScaler
with centering and unit scaling. Categorical preprocessing is SimpleImputer with
constant `__MISSING__`, then OneHotEncoder(categories="auto", drop=None,
handle_unknown="ignore", dtype=float64, sparse_output=True), without frequency or
category-count restrictions. Vocabulary is learned on train only and sorted by
the encoder within each feature. Exact vocabulary and all 45 output names are in
the unchanged golden fixture and the new preprocessor metadata.

ColumnTransformer uses numeric then categorical branches, remainder="drop",
sparse_threshold=0.3, no weights, n_jobs=None, verbose=False, and prefixed output
names. Its combined output on this dataset is dense: four scaled numeric columns
followed by 41 one-hot columns. Public transform methods return C-contiguous
float64 matrices. The original scaler saw 4,930 samples.

The processed train CSV has no missing values. Original training selects its
feature columns without sorting rows or adding further cleaning. The new fitter
validates RAG-compatible rows but fits the original processed values and order.
It does not substitute query-cleaned values when computing fitted statistics.

Fit imputation and query cleaning are distinct. A missing TotalCharges in a
zero-tenure training row is median-imputed while fitting the numeric transformer;
at query/reference transform time, shared validation replaces that value with
zero. A dedicated synthetic test protects this distinction. At query time other
missing numeric values, invalid numeric values, and missing/unknown categories
are rejected. The encoder's ignore-unknown setting does not bypass validation.

## New API and fitting provenance

`adapters/retrieval/rag_preprocessor.py` exposes:

- `RAGPreprocessor.fit_train(train_path, splits_path)` for explicit fitting.
- `clean(customer)`, `transform_one(customer)`, `transform_many(customers)`.
- `feature_schema`, `input_feature_names`, `output_feature_names`,
  `categorical_vocabulary`, and defensive-copy `metadata`.

Schema/order are owned by the new RAG module, not read from training config,
model.pkl, a reference ZIP, or golden values. Constants describe the unchanged
19-feature interface. No saved statistic or fitted estimator is copied in.

Fitting reads the source bytes, validates split columns/labels and unique IDs,
and requires dataset IDs to equal exactly the manifest's train population.
Duplicate, unknown, held-out, incomplete, and mixed/full-dataset populations fail
before fitting. IDs are removed before passing features to ColumnTransformer.
Churn is excluded by CSV usecols and is never supplied as y. Target values may
change or the target column may be absent without changing fitted preprocessing.

The fitter preserves processed-train row order, validates the 19 features, learns
category vocabulary, and fits only the numeric/categorical transformer branches.
Metadata records the observed fitted state, library versions, train/split byte
hashes, train population hash and ordered-ID hash; it includes no raw IDs.
Transforms never refit. An empty batch returns shape (0, output_dimension).

## Equivalence evidence

Expected values come from immutable v1 golden JSON, private vector observations,
and matrix.npy/records.json in the v1 ZIP. Actual values come from a newly fitted
RAGPreprocessor. No fitted preprocessor is extracted from model.pkl.

On the existing data:

| Check | Result |
| --- | --- |
| Fitted parameters, medians, mean, variance, scale, categories | Exact |
| Input/output names and order | Exact |
| Reference matrix | 4,930 x 45, float64, byte-for-byte exact |
| Maximum matrix absolute error | 0.0 |
| Matrix SHA-256 | f4087f1b797b512c63ed79cbaba9942891b67e244c4f24453d6832df4a6cdf47 |
| Fixed validation vectors | 8/8 exact; maximum absolute error 0.0 |
| Top-5 neighbor IDs and order | 8/8 exact; IDs remain internal |
| Euclidean distances | 8/8 exact; maximum absolute error 0.0 |
| Ordered reference examples and prompt hashes | 8/8 exact, five examples each |
| K-boundary tie-breaking and self-exclusion | Preserved, synthetic controlled tests |

`scripts/verify_rag_preprocessor.py` computes these comparisons and prints a
non-identifying JSON report. Its CandidateReferenceView is an in-memory test/tool
harness for the unchanged retriever, not an alternate runtime Store or new
artifact. The view uses only new query vectors and the new reference matrix.
Reference labels are used only for few-shot examples, never preprocessing fit.
Validation queries load only feature columns, never validation IDs/labels.

## Classifier independence and sklearn import nuance

An isolated fresh process fits from train/splits and reproduces the full matrix
hash and eight query hashes while access to both model bundles and reference ZIPs
is denied. Imports of repository ML/training, old artifact loader/Store,
retriever, and predictor adapters are blocked in that process.

In the installed sklearn, compose.__init__ additionally exports
TransformedTargetRegressor, importing LinearRegression from linear_model. Blocking
the entire linear_model namespace would therefore also block an unrelated
standard utility. The isolated test instead installs poison LogisticRegression
and LogisticRegressionCV exports before importing sklearn, blocks the actual
`sklearn.linear_model._logistic` implementation and all legacy modules, and checks
that no real logistic module file was loaded. The poison classes cannot be
instantiated. Fitting and all matrix/query hash checks still pass.

Thus new preprocessing requires no Logistic Regression implementation, instance,
fitted state or bundle. This does not claim sklearn never imports the broader
linear_model namespace. The comparison tool also imports the existing retrieval
adapter, whose current v1 import graph remains legacy-dependent; it does not
deserialize or invoke a classifier. The isolated test covers the new module
separately from that comparison harness.

## Running verification

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m scripts.verify_rag_preprocessor
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CHURN_GOLDEN_REQUIRE_ASSETS=1 python3 -m pytest tests/test_rag_preprocessor.py tests/test_llm_rag_golden.py -q
MYPYPATH=src python3 -m mypy --explicit-package-bases src/churn_retention/adapters/retrieval/rag_preprocessor.py scripts/verify_rag_preprocessor.py tests/test_rag_preprocessor.py
```

No baseline refresh is provided. Keep the current library versions, raw processed
fit values and row order when certifying exact compatibility. Future data,
vocabulary, imputation policies or library changes require a separately reviewed
comparison. Matching this technical baseline does not establish LLM prediction
quality or calibration.

The next artifact step may use this independent fitter, but artifact v2 and runtime
migration require separate authorization. The existing v1 runtime is unchanged.

## Verification on 2026-11-01

- New tests: 57 cases, including actual classifier-unavailable fitting, train-only
  guards, fitting-vs-query imputation, exact matrix/query checks and retrieval.
- New, golden, reference/retrieval, prompt, predictor and Streamlit LLM group:
  380 passed, including all 48 unchanged golden cases.
- Full regression suite: 846 passed, 15 warnings, no skipped tests.
- Ruff check/format and Mypy for all three new Python files: passed.
- The standalone comparison tool exits successfully with equivalent=true;
  matrix, query vectors and distances have maximum absolute error 0.0, and all
  eight complete prompt hashes equal v1.
- Verification used socket-blocking guards. No real API was called.
- All 165 files present before this task retain their SHA-256 fingerprints,
  including every existing dirty file, dataset, split manifest, model bundle,
  reference artifact, golden capture script and public/private golden fixture.
- Branch/HEAD remain `feat/ml-batch-scoring` and
  `84109d9178acf5d11d3d49770eb22568c9f958d4`. Only four new untracked files were
  added: this document, the new module, verification tool and new tests.
- No source file already present was edited; no commit, push or integration Git
  operation was performed. No artifact v2 or runtime migration was implemented.
