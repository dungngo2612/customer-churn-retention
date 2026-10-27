from typing import Any, Dict

import numpy as np
import pandas as pd

from churn_retention.ml.evaluation.metrics import (
    _validate_prediction_arrays,
    align_and_validate_data,
    compute_calibration_metrics,
    compute_ranking_metrics,
    compute_threshold_metrics,
)


def evaluate_predictions(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    """Tổng hợp metric từ các mảng nhãn 0/1 và xác suất churn cùng thứ tự.

    Bên gọi phải truyền threshold; proba >= threshold được phân loại là churn.
    Kết quả có cùng cấu trúc với generate_evaluation_report.
    """
    y_true, y_proba = _validate_prediction_arrays(y_true, y_proba)
    return {
        "total_samples": int(len(y_true)),
        "ranking_metrics": compute_ranking_metrics(y_true, y_proba),
        "threshold_metrics": compute_threshold_metrics(y_true, y_proba, threshold=threshold),
        "calibration_metrics": compute_calibration_metrics(y_true, y_proba),
    }


def generate_evaluation_report(
    df_true: pd.DataFrame,
    df_pred: pd.DataFrame,
    threshold: float = 0.5,
    id_col: str = "customerID",
    label_col: str = "Churn",
    prob_col: str = "churn_prob",
) -> Dict[str, Any]:
    """Thực hiện kiểm định dữ liệu, ghép cặp theo customerID và tổng hợp

    toàn bộ báo cáo metric đánh giá mô hình.

    Parameters:
        df_true: DataFrame chứa nhãn thực tế và customerID.
        df_pred: DataFrame chứa xác suất dự đoán và customerID.
        threshold: Ngưỡng xác suất để phân loại nhị phân (mặc định 0.5).
            0.5 chỉ để test; ngưỡng thật do bên gọi truyền vào (chọn trên validation,
            khoá ở bước sau), không dùng test set để chọn.
        id_col: Tên cột định danh khách hàng.
        label_col: Tên cột nhãn thực tế.
        prob_col: Tên cột xác suất dự đoán.

    Returns:
        Dict chứa đầy đủ các nhóm metric (Ranking, Threshold, Calibration)
        cùng metadata tổng số dòng đã đánh giá.
    """
    # 1. Kiểm tra và ghép nối dữ liệu an toàn theo customerID
    y_true, y_prob = align_and_validate_data(
        df_true=df_true,
        df_pred=df_pred,
        id_col=id_col,
        label_col=label_col,
        prob_col=prob_col,
    )

    return evaluate_predictions(y_true, y_prob, threshold=threshold)
