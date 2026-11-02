"""Small fictional cohort; probabilities are hand-authored, never LLM outputs."""

from churn_retention.contracts.llm import LLMChurnPrediction
from churn_retention.contracts.llm_prediction_record import PredictionRecord
from churn_retention.modules.retention.llm_adapter import CustomerContext


def demo_inputs() -> tuple[tuple[PredictionRecord, CustomerContext], ...]:
    base: dict[str, object] = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.0,
        "TotalCharges": 840.0,
    }
    result = []
    for index, (probability, active, tenure, charge) in enumerate(
        [
            (0.8, True, 12, 70.0),
            (0.4, True, 24, 40.0),
            (0.9, False, 12, 100.0),
            (0.7, True, 0, 80.0),
        ],
        start=1,
    ):
        record = PredictionRecord.capture(
            {
                **base,
                "tenure": tenure,
                "MonthlyCharges": charge,
                "TotalCharges": tenure * charge,
            },
            LLMChurnPrediction(
                churn_prediction=int(probability >= 0.5),
                churn_probability=probability,
                reason="DỮ LIỆU GIẢ: xác suất viết tay để minh họa, không phải kết quả LLM.",
                provider=None,
                reference_count=0,
                reference_version="synthetic-demo-v1",
            ),
            origin="synthetic",
        )
        result.append(
            (
                record,
                CustomerContext(customer_ref=f"DEMO-{index:03}", observed_active=active),
            )
        )
    return tuple(result)
