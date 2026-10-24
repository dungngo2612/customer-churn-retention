"""SHAP Explainability module for CatBoost model bundle.

Cung cấp các hàm trích xuất SHAP Global (Feature Importance) và Local SHAP (đóng góp từng thuộc tính của khách hàng).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from catboost import Pool
import numpy as np
import pandas as pd


def get_global_feature_importances(predictor: Any) -> pd.DataFrame:
    """Trích xuất độ quan trọng của các đặc trưng (Global Feature Importances)."""
    try:
        bundle = getattr(predictor, "_bundle", None)
        if bundle is None:
            return pd.DataFrame(columns=["feature", "importance"])

        pipeline = getattr(bundle, "_pipeline", None)
        if pipeline is None:
            return pd.DataFrame(columns=["feature", "importance"])

        clf = pipeline.named_steps.get("model")
        if clf is None:
            return pd.DataFrame(columns=["feature", "importance"])

        feature_names = clf.feature_names_
        importances = clf.feature_importances_

        df_imp = pd.DataFrame({
            "feature": feature_names,
            "importance": importances,
        }).sort_values(by="importance", ascending=False).reset_index(drop=True)

        return df_imp
    except Exception:
        return pd.DataFrame(columns=["feature", "importance"])


def explain_single_customer_shap(
    predictor: Any,
    customer_row: pd.Series | pd.DataFrame,
) -> Optional[pd.DataFrame]:
    """Tính toán giá trị TreeSHAP cục bộ cho một khách hàng đơn lẻ."""
    try:
        bundle = getattr(predictor, "_bundle", None)
        if bundle is None:
            return None

        pipeline = getattr(bundle, "_pipeline", None)
        if pipeline is None:
            return None

        prep = pipeline.named_steps.get("preprocessor")
        clf = pipeline.named_steps.get("model")
        if prep is None or clf is None:
            return None

        # Đảm bảo đầu vào là DataFrame 1 dòng
        if isinstance(customer_row, pd.Series):
            df_in = customer_row.to_frame().T
        else:
            df_in = customer_row.head(1).copy()

        # Tiền xử lý
        X_trans = prep.transform(df_in)
        cat_indices = clf.get_cat_feature_indices()
        pool = Pool(X_trans, cat_features=cat_indices)

        # Trích xuất Shap Values
        shap_raw = clf.get_feature_importance(pool, type="ShapValues")
        # shap_raw có shape (1, n_features + 1), cột cuối là base value (bias)
        shap_vals = shap_raw[0, :-1]
        base_value = shap_raw[0, -1]
        feature_names = clf.feature_names_

        records = []
        for feat_name, s_val in zip(feature_names, shap_vals):
            feat_val = df_in[feat_name].iloc[0] if feat_name in df_in.columns else ""
            records.append({
                "feature": feat_name,
                "value": str(feat_val),
                "shap_value": float(s_val),
                "impact": "Tăng rủi ro Churn (+)" if s_val > 0 else "Giảm rủi ro Churn (-)",
            })

        df_res = pd.DataFrame(records)
        df_res["abs_shap"] = df_res["shap_value"].abs()
        df_res = df_res.sort_values(by="abs_shap", ascending=False).drop(columns=["abs_shap"]).reset_index(drop=True)
        return df_res
    except Exception:
        return None
