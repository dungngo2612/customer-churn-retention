from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _validate_probabilities(y_prob: np.ndarray) -> None:
    if not np.all(np.isfinite(y_prob)):
        raise ValueError("Xác suất dự đoán chứa giá trị NaN hoặc vô cực (inf).")
    if np.any((y_prob < 0.0) | (y_prob > 1.0)):
        raise ValueError("Xác suất dự đoán nằm ngoài khoảng hợp lệ [0.0, 1.0].")


def _validate_prediction_arrays(
    y_true: np.ndarray, y_prob: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true)
    try:
        y_prob = np.asarray(y_prob, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Xác suất dự đoán phải là các giá trị số.") from exc

    if y_true.ndim != 1 or y_prob.ndim != 1:
        raise ValueError("y_true và y_prob phải là các mảng một chiều.")
    if len(y_true) != len(y_prob):
        raise ValueError("y_true và y_prob phải có cùng độ dài.")
    if len(y_true) == 0:
        raise ValueError("Không thể đánh giá dataset rỗng.")
    if not np.all(np.isin(y_true, [0, 1])):
        raise ValueError("Nhãn dạng số chỉ được chứa 0 và 1, không được có giá trị thiếu.")

    _validate_probabilities(y_prob)
    if len(np.unique(y_true)) < 2:
        raise ValueError("Cần có cả lớp churn và không churn để tính ROC-AUC.")
    return y_true.astype(int), y_prob


def align_and_validate_data(
    df_true: pd.DataFrame,
    df_pred: pd.DataFrame,
    id_col: str = "customerID",
    label_col: str = "Churn",
    prob_col: str = "churn_prob",
) -> Tuple[np.ndarray, np.ndarray]:
    """Kiểm tra tính toàn vẹn của customerID, xác thực nhãn churn 2 lớp

    và ghép cặp dữ liệu theo ID mà không dựa vào thứ tự dòng ngầm định.

    Parameters:
        df_true: DataFrame chứa nhãn thực tế và customerID.
        df_pred: DataFrame chứa xác suất dự đoán và customerID.
        id_col: Tên cột định danh khách hàng (mặc định 'customerID').
        label_col: Tên cột nhãn thực tế (mặc định 'Churn').
        prob_col: Tên cột xác suất dự đoán churn (mặc định 'churn_prob').

    Returns:
        Tuple (y_true, y_prob): Hai mảng 1D NumPy đã khớp thứ tự theo từng khách hàng.

    Raises:
        KeyError: Khi thiếu cột bắt buộc.
        ValueError: Khi ID bị trùng, thiếu, lệch hoặc nhãn không thuộc {0, 1}.
    """
    # 1. Kiểm tra sự tồn tại của các cột bắt buộc
    required_checks = [
        (df_true, id_col, "df_true"),
        (df_true, label_col, "df_true"),
        (df_pred, id_col, "df_pred"),
        (df_pred, prob_col, "df_pred"),
    ]
    for df, col, name in required_checks:
        if col not in df.columns:
            raise KeyError(f"Cột bắt buộc '{col}' không tồn tại trong {name}.")

    # 2. Kiểm tra trùng lặp ID trong nội bộ từng bảng
    if df_true[id_col].duplicated().any():
        dups_true = df_true[id_col][df_true[id_col].duplicated()].unique().tolist()
        raise ValueError(
            f"df_true chứa {len(dups_true)} customerID bị trùng lặp. Ví dụ: {dups_true[:5]}"
        )

    if df_pred[id_col].duplicated().any():
        dups_pred = df_pred[id_col][df_pred[id_col].duplicated()].unique().tolist()
        raise ValueError(
            f"df_pred chứa {len(dups_pred)} customerID bị trùng lặp. Ví dụ: {dups_pred[:5]}"
        )

    for df, name in ((df_true, "df_true"), (df_pred, "df_pred")):
        ids = df[id_col]
        missing = ids.isna() | ids.astype("string").str.strip().eq("")
        if missing.any():
            raise ValueError(f"{name} chứa {int(missing.sum())} customerID bị thiếu hoặc rỗng.")

    # 3. Kiểm tra thiếu hoặc lệch ID giữa hai tập (Set Comparison)
    set_true = set(df_true[id_col])
    set_pred = set(df_pred[id_col])

    missing_in_pred = set_true - set_pred
    extra_in_pred = set_pred - set_true

    if missing_in_pred:
        raise ValueError(
            f"Kết quả dự đoán thiếu {len(missing_in_pred)} customerID so với nhãn gốc. "
            f"Ví dụ: {list(missing_in_pred)[:5]}"
        )
    if extra_in_pred:
        raise ValueError(
            f"Kết quả dự đoán thừa {len(extra_in_pred)} customerID không tồn tại trong nhãn gốc. "
            f"Ví dụ: {list(extra_in_pred)[:5]}"
        )

    # 4. Chuẩn hóa nhãn Churn về hai lớp số: không churn = 0, churn = 1
    y_raw = df_true[label_col].copy()

    if y_raw.empty:
        raise ValueError("Không thể đánh giá dataset rỗng.")

    if pd.api.types.is_bool_dtype(y_raw):
        y_int = y_raw.astype(int).to_numpy()

    elif pd.api.types.is_numeric_dtype(y_raw):
        if y_raw.isna().any() or not set(y_raw.unique()).issubset({0, 1}):
            raise ValueError("Nhãn dạng số chỉ được chứa 0 và 1, không được có giá trị thiếu.")
        y_int = y_raw.astype(int).to_numpy()

    else:
        normalized = y_raw.astype("string").str.strip().str.lower()
        mapping = {
            "yes": 1,
            "no": 0,
            "1": 1,
            "0": 0,
            "true": 1,
            "false": 0,
        }

        invalid_labels = normalized.isna() | ~normalized.isin(mapping)
        if invalid_labels.any():
            invalid_values = normalized[invalid_labels].unique().tolist()
            raise ValueError(f"Nhãn Churn không hợp lệ: {invalid_values}")

        y_int = normalized.map(mapping).to_numpy(dtype=int)

    # 5. Ghép nối dữ liệu theo customerID (tuyệt đối không dùng thứ tự dòng)
    df_true_clean = pd.DataFrame({id_col: df_true[id_col], "y_true": y_int})
    df_pred_clean = pd.DataFrame({id_col: df_pred[id_col], "y_prob": df_pred[prob_col]})

    merged = pd.merge(df_true_clean, df_pred_clean, on=id_col, how="inner")

    # 6. Kiểm tra tính hợp lệ của xác suất dự đoán
    y_prob = merged["y_prob"].to_numpy(dtype=float)
    _validate_probabilities(y_prob)

    y_true = merged["y_true"].to_numpy(dtype=int)
    return y_true, y_prob


def compute_ranking_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> Dict[str, float]:
    """Tính các metric đánh giá khả năng xếp hạng xác suất của mô hình.

    - average_precision: Metric xếp hạng chính (PR-AUC).
    - roc_auc: Metric xếp hạng bổ sung.
    """
    if len(np.unique(y_true)) < 2:
        raise ValueError("Cần có cả lớp churn và không churn để tính ROC-AUC.")
    avg_precision = float(average_precision_score(y_true, y_prob))
    roc_auc = float(roc_auc_score(y_true, y_prob))

    return {
        "average_precision": round(avg_precision, 4),
        "roc_auc": round(roc_auc, 4),
    }


def compute_threshold_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Tính các metric phân loại nhị phân tại một ngưỡng (threshold) chốt.

    Bao gồm Precision, Recall, F1 và Confusion Matrix chi tiết.
    """
    y_true, y_prob = _validate_prediction_arrays(y_true, y_prob)
    if not (0.0 <= threshold <= 1.0):
        raise ValueError(f"Ngưỡng threshold phải nằm trong [0.0, 1.0], nhận được: {threshold}")

    y_pred_bin = (y_prob >= threshold).astype(int)

    prec = float(precision_score(y_true, y_pred_bin, zero_division=0))
    rec = float(recall_score(y_true, y_pred_bin, zero_division=0))
    f1 = float(f1_score(y_true, y_pred_bin, zero_division=0))

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_bin, labels=[0, 1]).ravel()

    return {
        "threshold": threshold,
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
    }


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> Dict[str, Any]:
    """Tính các metric đánh giá độ chuẩn xác của xác suất dự đoán (Calibration).

    - brier_score: Sai số bình phương trung bình giữa xác suất dự đoán và nhãn thực tế [0, 1].
    - log_loss: Cross-entropy loss đánh giá độ phạt của sai số xác suất.
    - calibration_curve: Tọa độ các điểm (prob_true, prob_pred) để biểu diễn đường chuẩn hóa.
    """
    brier = float(brier_score_loss(y_true, y_prob))
    loss = float(log_loss(y_true, y_prob))

    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy=strategy)

    return {
        "brier_score": round(brier, 4),
        "log_loss": round(loss, 4),
        "calibration_curve": {
            "prob_true": [round(float(val), 4) for val in prob_true],
            "prob_pred": [round(float(val), 4) for val in prob_pred],
            "n_bins": n_bins,
            "strategy": strategy,
        },
    }
