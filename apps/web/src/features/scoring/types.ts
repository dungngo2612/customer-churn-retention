export interface Prediction {
  customer_id: string;
  dataset_id: string;
  scoring_run_id: string;
  churn_probability: number;
  model_version: string;
  scored_at: string;
}

export interface PredictionBatch {
  scoring_run_id: string;
  model_version: string;
  predictions: Prediction[];
}
