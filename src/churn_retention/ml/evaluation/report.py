from typing import Any, Dict

import pandas as pd

from churn_retention.ml.evaluation.metrics import (
    align_and_validate_data,
    compute_calibration_metrics,
    compute_ranking_metrics,
    compute_threshold_metrics,
)


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

    # 2. Tính toán các nhóm metric
    ranking_res = compute_ranking_metrics(y_true, y_prob)
    threshold_res = compute_threshold_metrics(y_true, y_prob, threshold=threshold)
    calibration_res = compute_calibration_metrics(y_true, y_prob)

    # 3. Đóng gói thành báo cáo tổng hợp hoàn chỉnh
    report = {
        "total_samples": int(len(y_true)),
        "ranking_metrics": ranking_res,  # Average Precision (chính), ROC-AUC
        "threshold_metrics": threshold_res,  # Precision, Recall, F1, Confusion Matrix
        "calibration_metrics": calibration_res,  # Brier score, Log loss, Calibration curve
    }

    return report
