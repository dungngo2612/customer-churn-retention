import type { PredictionBatch } from "./types";

// DEMO: all IDs, probabilities, and timestamps below are fabricated.
export const demoScoringRun: PredictionBatch = {
  scoring_run_id: "demo-scoring-run",
  model_version: "demo-model-not-trained",
  predictions: [
    {
      customer_id: "demo-C001",
      dataset_id: "demo-dataset-fake",
      scoring_run_id: "demo-scoring-run",
      churn_probability: 0.82,
      model_version: "demo-model-not-trained",
      scored_at: "2026-10-17T03:00:00+00:00",
    },
    {
      customer_id: "demo-C002",
      dataset_id: "demo-dataset-fake",
      scoring_run_id: "demo-scoring-run",
      churn_probability: 0.35,
      model_version: "demo-model-not-trained",
      scored_at: "2026-10-17T03:00:00+00:00",
    },
    {
      customer_id: "demo-C003",
      dataset_id: "demo-dataset-fake",
      scoring_run_id: "demo-scoring-run",
      churn_probability: 0.64,
      model_version: "demo-model-not-trained",
      scored_at: "2026-10-17T03:00:00+00:00",
    },
    {
      customer_id: "demo-C004",
      dataset_id: "demo-dataset-fake",
      scoring_run_id: "demo-scoring-run",
      churn_probability: 0.12,
      model_version: "demo-model-not-trained",
      scored_at: "2026-10-17T03:00:00+00:00",
    },
  ],
};
