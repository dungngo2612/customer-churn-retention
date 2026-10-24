"""Budget-constrained retention policy optimization module.

Mô đun tối ưu hóa phân bổ hành động can thiệp giữ chân khách hàng dưới ràng buộc ngân sách.

BẢN CHẤT TOÁN HỌC & TÍNH CHẤT XẤP XỈ:
------------------------------------
Bài toán chọn hành động giữ chân dưới ngân sách B với mỗi khách hàng nhận tối đa một
hành động can thiệp thuộc lớp bài toán Ba-lô đa lựa chọn (Multiple-Choice Knapsack Problem - MCKP).
Đây là bài toán thuộc lớp NP-hard.

Phương pháp giải:
1. Thuật toán Tham lam (Greedy Knapsack Heuristic):
   - ĐÂY LÀ PHƯƠNG ÁN XẤP XỈ (heuristic approximation), đảm bảo thời gian chạy O(N log N)
     và khả thi trên tập dữ liệu lớn hàng chục nghìn khách hàng.
   - Sắp xếp các cặp (khách hàng, hành động) theo thứ tự ưu tiên:
     + Cặp có chi phí C = 0 và lợi ích ΔEV > 0 được chọn ngay lập tức (không chia cho 0).
     + Cặp có C > 0 được sắp theo tỷ số lợi ích trên chi phí (efficiency_ratio = ΔEV / C) giảm dần.
   - Duyệt tuần tự: nếu khách chưa được gán hành động và tổng chi phí tích lũy + C <= B,
     gán hành động đó cho khách.
   - Khách hàng không được can thiệp (hoặc không có hành động nào có ΔEV > 0) được gán về
     phương án cơ sở A0 (Không can thiệp, C = 0, ΔEV = 0).

2. Phương pháp Duyệt vét cạn (Brute-Force Optimal Solver):
   - Dùng cho tập mẫu nhỏ (5 đến 10 khách hàng) để tìm nghiệm tối ưu toàn cục thực sự.
   - Đo lường Khoảng cách tối ưu (Optimality Gap):
       Optimality Gap = (ΔEV_opt - ΔEV_greedy) / ΔEV_opt  (nếu ΔEV_opt > 0)
     để đối chiếu và kiểm chứng chất lượng nghiệm của thuật toán tham lam.
"""

from __future__ import annotations

import itertools
from typing import Dict, List, Optional, Sequence, Tuple
import pandas as pd

from churn_retention.modules.retention.policy import (
    RetentionPolicy,
    compute_proxy_clv,
)
from churn_retention.modules.retention.use_cases import (
    ActionConfig,
    ValidationError,
    default_actions,
    evaluate_actions,
)

BUDGET_COLUMNS = [
    "rank",
    "customerID",
    "action_code",
    "action_name",
    "churn_prob",
    "proxy_value",
    "cost",
    "benefit",
    "net_value",
    "breakeven_prob",
    "churn_prob_after",
    "efficiency_ratio",
]


def select_actions_under_budget(
    customers: pd.DataFrame,
    predictions: pd.DataFrame,
    budget: float,
    actions: Optional[Sequence[ActionConfig]] = None,
    margin: float = 0.50,
    horizon_months: int = 12,
) -> pd.DataFrame:
    """Chọn hành động can thiệp cho từng khách hàng theo thuật toán Tham lam dưới ngân sách B.

    Tham số:
    ---------
    customers : pd.DataFrame
        Dữ liệu thuộc tính khách hàng.
    predictions : pd.DataFrame
        Bảng xác suất rời bỏ (customer_id, churn_prob).
    budget : float
        Ngân sách tối đa B (USD). Phải >= 0.
    actions : Sequence[ActionConfig] | None
        Danh sách cấu hình các hành động hợp lệ. Mặc định dùng default_actions().
    margin : float
        Biên lợi nhuận đóng góp (0 < margin <= 1). Mặc định 0.50.
    horizon_months : int
        Kỳ hạn phân tích (tháng). Mặc định 12.

    Trả về:
    -------
    pd.DataFrame: Bảng kết quả phân bổ với các cột trong BUDGET_COLUMNS.
    """
    if budget < 0:
        raise ValidationError(f"Ngân sách (budget) không được âm, nhận {budget}")

    options = evaluate_actions(customers, predictions, actions, margin, horizon_months)
    if options.empty:
        return pd.DataFrame(columns=BUDGET_COLUMNS)

    # 1. Lọc các cặp (khách, hành động) có lợi ích ròng dương (ΔEV > 0) và không phải A0
    cand = options[(~options["is_no_action"]) & (options["net_value"] > 0)].copy()

    # 2. Tính tỷ số lợi ích trên chi phí (efficiency_ratio)
    def _compute_ratio(row: pd.Series) -> float:
        cost = float(row["cost"])
        nv = float(row["net_value"])
        if cost <= 0:
            return float("inf")
        return round(nv / cost, 6)

    if not cand.empty:
        cand["efficiency_ratio"] = cand.apply(_compute_ratio, axis=1)
        cand["is_zero_cost"] = cand["cost"] <= 0

        # Sắp xếp theo thứ tự ưu tiên:
        # a. Chi phí 0 và ΔEV > 0 lên đầu
        # b. Tỷ số ΔEV / C giảm dần
        # c. Giá trị ròng ΔEV giảm dần (tie-breaker)
        # d. Chi phí C tăng dần (ưu tiên phương án rẻ hơn nếu cùng ΔEV)
        # e. customerID, action_code để đảm bảo tính tất định
        cand = cand.sort_values(
            by=["is_zero_cost", "efficiency_ratio", "net_value", "cost", "customerID", "action_code"],
            ascending=[False, False, False, True, True, True],
        ).reset_index(drop=True)

    # 3. Duyệt tham lam: mỗi khách tối đa 1 hành động, tổng chi phí không vượt B
    assigned: Dict[str, dict] = {}
    current_cost = 0.0

    if not cand.empty:
        for _, row in cand.iterrows():
            cid = row["customerID"]
            c = float(row["cost"])
            if cid not in assigned:
                if c <= 0:
                    assigned[cid] = row.to_dict()
                elif current_cost + c <= budget + 1e-9:
                    assigned[cid] = row.to_dict()
                    current_cost += c

    # 4. Những khách chưa được can thiệp (hoặc hết ngân sách) gán về phương án cơ sở A0
    baseline = options[options["is_no_action"]].copy()
    baseline["efficiency_ratio"] = 0.0

    assigned_rows = list(assigned.values())
    unassigned_cids = set(options["customerID"]) - set(assigned.keys())
    unassigned_rows = baseline[baseline["customerID"].isin(unassigned_cids)].to_dict("records")

    all_rows = assigned_rows + unassigned_rows
    result = pd.DataFrame(all_rows)

    # Sắp xếp bảng kết quả: các khách được can thiệp xếp trên (theo ratio/ΔEV), A0 xếp dưới
    result = result.sort_values(
        by=["is_no_action", "efficiency_ratio", "net_value", "customerID"],
        ascending=[True, False, False, True],
    ).reset_index(drop=True)

    result["rank"] = range(1, len(result) + 1)
    return result[BUDGET_COLUMNS]


def solve_optimal_budget_brute_force(
    customers: pd.DataFrame,
    predictions: pd.DataFrame,
    budget: float,
    actions: Optional[Sequence[ActionConfig]] = None,
    margin: float = 0.50,
    horizon_months: int = 12,
) -> Tuple[pd.DataFrame, float]:
    """Tìm nghiệm tối ưu toàn cục bằng duyệt vét cạn (Brute-Force) cho tập mẫu nhỏ (<= 10 khách).

    Lưu ý: Không dùng cho dữ liệu lớn vì độ phức tạp hàm mũ O(K^N).
    """
    if budget < 0:
        raise ValidationError(f"Ngân sách (budget) không được âm, nhận {budget}")

    options = evaluate_actions(customers, predictions, actions, margin, horizon_months)
    if options.empty:
        return pd.DataFrame(columns=BUDGET_COLUMNS), 0.0

    cids = sorted(options["customerID"].unique())
    if len(cids) > 12:
        raise ValueError(
            f"Brute-force chỉ dùng để đối chiếu trên tập nhỏ (<= 12 khách), nhận {len(cids)} khách"
        )

    # Mỗi khách có danh sách lựa chọn: A0 (bắt buộc) và các hành động có net_value > 0
    choices_per_cid: List[List[dict]] = []
    for cid in cids:
        c_opts = options[options["customerID"] == cid]
        a0_row = c_opts[c_opts["is_no_action"]].iloc[0].to_dict()
        a0_row["efficiency_ratio"] = 0.0
        pos_opts = c_opts[(~c_opts["is_no_action"]) & (c_opts["net_value"] > 0)].copy()

        pos_rows = []
        for _, r in pos_opts.iterrows():
            d = r.to_dict()
            cost = float(d["cost"])
            d["efficiency_ratio"] = float("inf") if cost <= 0 else round(float(d["net_value"]) / cost, 6)
            pos_rows.append(d)

        choices_per_cid.append([a0_row] + pos_rows)

    best_net_value = -1.0
    best_combo: Optional[Tuple[dict, ...]] = None

    for combo in itertools.product(*choices_per_cid):
        total_cost = sum(r["cost"] for r in combo)
        if total_cost <= budget + 1e-9:
            total_nv = sum(r["net_value"] for r in combo)
            if total_nv > best_net_value:
                best_net_value = total_nv
                best_combo = combo

    if best_combo is None:
        return pd.DataFrame(columns=BUDGET_COLUMNS), 0.0

    df_opt = pd.DataFrame(list(best_combo))
    df_opt = df_opt.sort_values(
        by=["is_no_action", "efficiency_ratio", "net_value", "customerID"],
        ascending=[True, False, False, True],
    ).reset_index(drop=True)
    df_opt["rank"] = range(1, len(df_opt) + 1)
    return df_opt[BUDGET_COLUMNS], round(best_net_value, 4)


def compute_optimality_gap(greedy_net_value: float, optimal_net_value: float) -> float:
    """Tính Khoảng cách tối ưu (Optimality Gap).

    Gap = (ΔEV_opt - ΔEV_greedy) / ΔEV_opt nếu ΔEV_opt > 0, ngược lại trả về 0.0.
    """
    if optimal_net_value <= 1e-9:
        return 0.0
    return max(0.0, round((optimal_net_value - greedy_net_value) / optimal_net_value, 4))


def compare_greedy_vs_optimal(
    customers: pd.DataFrame,
    predictions: pd.DataFrame,
    budget: float,
    actions: Optional[Sequence[ActionConfig]] = None,
    margin: float = 0.50,
    horizon_months: int = 12,
) -> dict:
    """Đối chiếu nghiệm Tham lam với Nghiệm tối ưu toàn cục trên tập khách hàng nhỏ."""
    greedy_df = select_actions_under_budget(
        customers, predictions, budget, actions, margin, horizon_months
    )
    opt_df, opt_nv = solve_optimal_budget_brute_force(
        customers, predictions, budget, actions, margin, horizon_months
    )

    greedy_nv = float(greedy_df["net_value"].sum()) if not greedy_df.empty else 0.0
    greedy_cost = float(greedy_df["cost"].sum()) if not greedy_df.empty else 0.0
    opt_cost = float(opt_df["cost"].sum()) if not opt_df.empty else 0.0

    gap = compute_optimality_gap(greedy_nv, opt_nv)

    return {
        "budget": budget,
        "n_customers": len(customers),
        "greedy_net_value": round(greedy_nv, 4),
        "greedy_cost": round(greedy_cost, 2),
        "optimal_net_value": round(opt_nv, 4),
        "optimal_cost": round(opt_cost, 2),
        "optimality_gap": gap,
        "is_optimal": abs(opt_nv - greedy_nv) < 1e-4,
    }
