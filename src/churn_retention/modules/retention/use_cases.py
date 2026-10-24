"""Use case giữ chân khách hàng: chạy theo lô (DataFrame), cấu hình hành động nhận từ ngoài.

Quy ước kinh tế (đã thống nhất):
  V  = MonthlyCharges x H x margin            (giá trị thay thế, KHÔNG phải CLV thật)
  C  = contact_cost + acceptance_rate x offer_cost
  r  = acceptance_rate x efficacy
  B  = p x r x V ;  net_value (ΔEV) = B - C ;  p* = C / (r x V)
Chỉ chạy trên khách đang hoạt động (Churn == "No" và tenure > 0). Mỗi khách nhận tối đa
một hành động: hành động hợp lệ có ΔEV > 0 lớn nhất, nếu không có thì chọn không can thiệp.
V luôn tính trên MonthlyCharges gốc; mọi chi phí ưu đãi chỉ nằm trong C (không trừ hai lần).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import pandas as pd

# Cột bắt buộc để tính. Churn và tenure cần thêm để lọc khách đang hoạt động.
REQUIRED_COLUMNS = ("customerID", "MonthlyCharges", "Contract", "InternetService",
                    "TechSupport", "OnlineSecurity", "PaymentMethod")
FILTER_COLUMNS = ("Churn", "tenure")
PREDICTION_COLUMNS = ("customer_id", "churn_prob")

MANUAL_PAYMENTS = ("Electronic check", "Mailed check")


class ValidationError(ValueError):
    """Dữ liệu hoặc cấu hình đầu vào không hợp lệ."""


_OPS = {
    "eq": lambda s, v: s == v,
    "ne": lambda s, v: s != v,
    "in": lambda s, v: s.isin(list(v)),
    "not_in": lambda s, v: ~s.isin(list(v)),
    "ge": lambda s, v: s >= v,
    "gt": lambda s, v: s > v,
    "le": lambda s, v: s <= v,
    "lt": lambda s, v: s < v,
}


@dataclass(frozen=True)
class Condition:
    """Một điều kiện trên một cột dữ liệu thật, ví dụ Condition("Contract", "eq", "Month-to-month")."""
    column: str
    op: str
    value: object

    def __post_init__(self):
        if self.op not in _OPS:
            raise ValidationError(f"Toán tử '{self.op}' không hỗ trợ; chọn trong {sorted(_OPS)}")

    def mask(self, df: pd.DataFrame) -> pd.Series:
        return _OPS[self.op](df[self.column], self.value)


@dataclass(frozen=True)
class ActionConfig:
    """Một hành động giữ chân. Hợp lệ khi thỏa MỌI điều kiện trong all_of
    và (any_of rỗng hoặc thỏa ÍT NHẤT MỘT điều kiện trong any_of)."""
    code: str
    name: str
    contact_cost: float
    offer_cost: float
    acceptance_rate: float      # u
    efficacy: float             # e (mức giảm rủi ro tương đối ở nhóm chấp nhận)
    all_of: tuple = ()
    any_of: tuple = ()
    is_no_action: bool = False

    def __post_init__(self):
        if self.contact_cost < 0 or self.offer_cost < 0:
            raise ValidationError(f"{self.code}: chi phí không được âm")
        for label, v in (("acceptance_rate", self.acceptance_rate), ("efficacy", self.efficacy)):
            if not 0.0 <= v <= 1.0:
                raise ValidationError(f"{self.code}: {label} phải trong [0, 1], nhận {v}")
        if self.is_no_action and (self.expected_cost != 0 or self.effective_reduction != 0
                                  or self.all_of or self.any_of):
            raise ValidationError(f"{self.code}: phương án không can thiệp phải có chi phí 0, "
                                  "không giảm churn và không có điều kiện")

    @property
    def expected_cost(self) -> float:
        return self.contact_cost + self.acceptance_rate * self.offer_cost

    @property
    def effective_reduction(self) -> float:
        return self.acceptance_rate * self.efficacy


def default_actions() -> tuple:
    """Bốn phương án đã thống nhất (tham số là GIẢ ĐỊNH, không đo từ dữ liệu)."""
    return (
        ActionConfig("A0", "Không can thiệp", 0, 0, 0, 0, is_no_action=True),
        ActionConfig("A1", "Hỗ trợ kỹ thuật & bảo mật chủ động", 2, 15, 0.40, 0.25,
                     all_of=(Condition("InternetService", "ne", "No"),),
                     any_of=(Condition("TechSupport", "eq", "No"),
                             Condition("OnlineSecurity", "eq", "No"))),
        ActionConfig("A2", "Tư vấn hợp đồng & thanh toán tự động", 2, 20, 0.35, 0.40,
                     any_of=(Condition("Contract", "eq", "Month-to-month"),
                             Condition("PaymentMethod", "in", MANUAL_PAYMENTS))),
        ActionConfig("A3", "Ưu đãi giảm cước cam kết dài hạn", 3, 40, 0.40, 0.50,
                     all_of=(Condition("Contract", "eq", "Month-to-month"),
                             Condition("MonthlyCharges", "ge", 50.0))),
    )


def actions_from_dicts(items: Iterable[Mapping]) -> tuple:
    """Tạo cấu hình từ list dict (ví dụ đọc từ JSON/YAML)."""
    def conds(d, key):
        out = []
        for c in d.get(key, []):
            v = c["value"]
            out.append(Condition(c["column"], c["op"], tuple(v) if isinstance(v, (list, set)) else v))
        return tuple(out)

    actions = []
    for d in items:
        try:
            actions.append(ActionConfig(
                code=d["code"], name=d["name"], contact_cost=float(d["contact_cost"]),
                offer_cost=float(d["offer_cost"]), acceptance_rate=float(d["acceptance_rate"]),
                efficacy=float(d["efficacy"]), all_of=conds(d, "all_of"), any_of=conds(d, "any_of"),
                is_no_action=bool(d.get("is_no_action", False))))
        except KeyError as e:
            raise ValidationError(f"Cấu hình hành động thiếu trường {e}") from e
    validate_actions(actions)
    return tuple(actions)


def validate_actions(actions: Sequence[ActionConfig]) -> None:
    if not actions:
        raise ValidationError("Danh sách hành động rỗng")
    codes = [a.code for a in actions]
    if len(set(codes)) != len(codes):
        raise ValidationError(f"Mã hành động bị trùng: {codes}")
    if sum(a.is_no_action for a in actions) != 1:
        raise ValidationError("Cần đúng một phương án không can thiệp (is_no_action=True)")


def _sample(values, n=5) -> str:
    vals = list(values)
    return ", ".join(map(str, vals[:n])) + (f" ... (+{len(vals) - n})" if len(vals) > n else "")


_CHURN_MAP = {"Yes": 1, "No": 0, 1: 1, 0: 0, True: 1, False: 0, "1": 1, "0": 0}


def _prepare(customers: pd.DataFrame, predictions: pd.DataFrame, actions) -> pd.DataFrame:
    """Kiểm tra đầu vào, lọc khách đang hoạt động, ghép prediction THEO customer_id."""
    if not isinstance(customers, pd.DataFrame) or not isinstance(predictions, pd.DataFrame):
        raise ValidationError("customers và predictions phải là DataFrame")

    cond_cols = {c.column for a in actions for c in (*a.all_of, *a.any_of)}
    needed = list(dict.fromkeys([*REQUIRED_COLUMNS, *FILTER_COLUMNS, *sorted(cond_cols)]))
    missing = [c for c in needed if c not in customers.columns]
    if missing:
        raise ValidationError(f"Thiếu cột bắt buộc trong dữ liệu khách hàng: {missing}")
    missing_p = [c for c in PREDICTION_COLUMNS if c not in predictions.columns]
    if missing_p:
        raise ValidationError(f"Thiếu cột trong predictions: {missing_p}")

    cust = customers.copy()
    if cust["customerID"].isna().any():
        raise ValidationError("customerID có giá trị rỗng")
    cust["customerID"] = cust["customerID"].astype(str)
    dup = cust.loc[cust["customerID"].duplicated(), "customerID"].unique()
    if len(dup):
        raise ValidationError(f"customerID bị trùng: {_sample(dup)}")

    churn = cust["Churn"].map(_CHURN_MAP)
    if churn.isna().any():
        bad = cust.loc[churn.isna(), "Churn"].unique()
        raise ValidationError(f"Giá trị Churn không hợp lệ (chỉ nhận Yes/No hoặc 1/0): {_sample(bad)}")
    tenure = pd.to_numeric(cust["tenure"], errors="coerce")
    if tenure.isna().any():
        raise ValidationError("tenure có giá trị thiếu hoặc không phải số")
    active = cust[(churn == 0) & (tenure > 0)].copy()

    mc = pd.to_numeric(active["MonthlyCharges"], errors="coerce")
    bad_mc = active.loc[mc.isna() | (mc < 0), "customerID"]
    if len(bad_mc):
        raise ValidationError(f"MonthlyCharges thiếu, không phải số hoặc âm ở khách: {_sample(bad_mc)}")
    active["MonthlyCharges"] = mc
    for col in needed:
        nulls = active.loc[active[col].isna(), "customerID"]
        if len(nulls):
            raise ValidationError(f"Cột '{col}' có giá trị thiếu ở khách: {_sample(nulls)}")

    pred = predictions[list(PREDICTION_COLUMNS)].copy()
    if pred["customer_id"].isna().any():
        raise ValidationError("predictions có customer_id rỗng")
    pred["customer_id"] = pred["customer_id"].astype(str)
    dup = pred.loc[pred["customer_id"].duplicated(), "customer_id"].unique()
    if len(dup):
        raise ValidationError(f"predictions có customer_id trùng: {_sample(dup)}")
    pred["churn_prob"] = pd.to_numeric(pred["churn_prob"], errors="coerce")
    if pred["churn_prob"].isna().any() or not pred["churn_prob"].between(0, 1).all():
        raise ValidationError("churn_prob phải là số trong [0, 1], không được thiếu")
    unknown = set(pred["customer_id"]) - set(cust["customerID"])
    if unknown:
        raise ValidationError(f"predictions chứa ID không có trong dữ liệu khách hàng: {_sample(sorted(unknown))}")
    no_pred = sorted(set(active["customerID"]) - set(pred["customer_id"]))
    if no_pred:
        raise ValidationError(f"Thiếu prediction cho khách đang hoạt động: {_sample(no_pred)}")

    return active.merge(pred.rename(columns={"customer_id": "customerID"}),
                        on="customerID", how="left", validate="one_to_one")


def _eligible_mask(df: pd.DataFrame, a: ActionConfig) -> pd.Series:
    m = pd.Series(True, index=df.index)
    for c in a.all_of:
        m &= c.mask(df)
    if a.any_of:
        anym = pd.Series(False, index=df.index)
        for c in a.any_of:
            anym |= c.mask(df)
        m &= anym
    return m


def _check_params(margin: float, horizon_months: int) -> None:
    if not 0 < margin <= 1:
        raise ValidationError(f"margin phải trong (0, 1], nhận {margin}")
    if int(horizon_months) != horizon_months or horizon_months <= 0:
        raise ValidationError(f"horizon_months phải là số nguyên dương, nhận {horizon_months}")


OPTION_COLUMNS = ["customerID", "action_code", "action_name", "is_no_action", "churn_prob",
                  "proxy_value", "cost", "reduction", "churn_prob_after",
                  "benefit", "net_value", "breakeven_prob"]


def evaluate_actions(customers: pd.DataFrame, predictions: pd.DataFrame,
                     actions: Sequence[ActionConfig] | None = None,
                     margin: float = 0.5, horizon_months: int = 12) -> pd.DataFrame:
    """Bảng dài: mỗi dòng là một cặp (khách, hành động HỢP LỆ). Hành động không hợp lệ bị loại.
    breakeven_prob = C / (r x V): khách có churn_prob dưới mức này thì hành động không có lời
    (nếu > 1 thì không bao giờ có lời với khách đó)."""
    actions = tuple(actions) if actions is not None else default_actions()
    validate_actions(actions)
    _check_params(margin, horizon_months)
    base = _prepare(customers, predictions, actions)
    base["proxy_value"] = base["MonthlyCharges"] * horizon_months * margin

    frames = []
    for a in actions:
        sub = base.loc[_eligible_mask(base, a), ["customerID", "churn_prob", "proxy_value"]].copy()
        sub["action_code"], sub["action_name"], sub["is_no_action"] = a.code, a.name, a.is_no_action
        sub["cost"], sub["reduction"] = a.expected_cost, a.effective_reduction
        frames.append(sub)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=OPTION_COLUMNS)
    if out.empty:
        return pd.DataFrame(columns=OPTION_COLUMNS)

    out["churn_prob_after"] = out["churn_prob"] * (1 - out["reduction"])
    out["benefit"] = out["churn_prob"] * out["reduction"] * out["proxy_value"]
    out["net_value"] = out["benefit"] - out["cost"]
    denom = out["reduction"] * out["proxy_value"]
    out["breakeven_prob"] = out["cost"] / denom.where(denom > 0)
    return out[OPTION_COLUMNS].sort_values(["customerID", "action_code"]).reset_index(drop=True)


RANK_COLUMNS = ["rank", "customerID", "action_code", "action_name", "churn_prob", "proxy_value",
                "cost", "benefit", "net_value", "breakeven_prob", "churn_prob_after"]


def recommend(customers: pd.DataFrame, predictions: pd.DataFrame,
              actions: Sequence[ActionConfig] | None = None,
              margin: float = 0.5, horizon_months: int = 12) -> pd.DataFrame:
    """Mỗi khách đang hoạt động một dòng: hành động hợp lệ có net_value > 0 lớn nhất,
    nếu không có thì chọn không can thiệp (net_value = 0). Xếp hạng theo net_value giảm dần."""
    options = evaluate_actions(customers, predictions, actions, margin, horizon_months)
    if options.empty:
        return pd.DataFrame(columns=RANK_COLUMNS)

    cand = options[(~options["is_no_action"]) & (options["net_value"] > 0)]
    cand = cand.sort_values(["customerID", "net_value", "cost", "action_code"],
                            ascending=[True, False, True, True])
    best = cand.drop_duplicates("customerID", keep="first")
    baseline = options[options["is_no_action"]].drop_duplicates("customerID")
    chosen = pd.concat([best, baseline[~baseline["customerID"].isin(best["customerID"])]])

    chosen = chosen.sort_values(["net_value", "customerID"], ascending=[False, True]).reset_index(drop=True)
    chosen["rank"] = range(1, len(chosen) + 1)
    return chosen[RANK_COLUMNS]


def sensitivity_by_margin(customers: pd.DataFrame, predictions: pd.DataFrame,
                          margins: Sequence[float] = (0.2, 0.3, 0.4, 0.5),
                          actions: Sequence[ActionConfig] | None = None,
                          horizon_months: int = 12) -> pd.DataFrame:
    """Độ nhạy theo biên lợi nhuận: mỗi margin một dòng tổng hợp."""
    actions = tuple(actions) if actions is not None else default_actions()
    codes = [a.code for a in actions if not a.is_no_action]
    no_action = next(a.code for a in actions if a.is_no_action)
    rows = []
    for m in margins:
        rec = recommend(customers, predictions, actions, m, horizon_months)
        act = rec[rec["action_code"] != no_action]
        row = {"margin": m, "n_customers": len(rec), "n_intervened": len(act),
               "total_cost": act["cost"].sum(), "total_benefit": act["benefit"].sum(),
               "total_net_value": act["net_value"].sum()}
        counts = act["action_code"].value_counts()
        row.update({f"n_{c}": int(counts.get(c, 0)) for c in codes})
        rows.append(row)
    return pd.DataFrame(rows)


def select_actions_under_budget(
    customers: pd.DataFrame,
    predictions: pd.DataFrame,
    budget: float,
    actions: Sequence[ActionConfig] | None = None,
    margin: float = 0.50,
    horizon_months: int = 12,
) -> pd.DataFrame:
    """Tối ưu phân bổ hành động giữ chân dưới ràng buộc ngân sách B (Greedy Heuristic)."""
    from churn_retention.modules.retention.budget import (
        select_actions_under_budget as _select,
    )
    return _select(
        customers=customers,
        predictions=predictions,
        budget=budget,
        actions=actions,
        margin=margin,
        horizon_months=horizon_months,
    )


def solve_optimal_budget_brute_force(
    customers: pd.DataFrame,
    predictions: pd.DataFrame,
    budget: float,
    actions: Sequence[ActionConfig] | None = None,
    margin: float = 0.50,
    horizon_months: int = 12,
) -> tuple[pd.DataFrame, float]:
    """Tìm nghiệm tối ưu toàn cục bằng duyệt vét cạn (cho tập mẫu <= 10 khách)."""
    from churn_retention.modules.retention.budget import (
        solve_optimal_budget_brute_force as _solve,
    )
    return _solve(
        customers=customers,
        predictions=predictions,
        budget=budget,
        actions=actions,
        margin=margin,
        horizon_months=horizon_months,
    )


def compare_greedy_vs_optimal(
    customers: pd.DataFrame,
    predictions: pd.DataFrame,
    budget: float,
    actions: Sequence[ActionConfig] | None = None,
    margin: float = 0.50,
    horizon_months: int = 12,
) -> dict:
    """Đối chiếu giải pháp Tham lam với Nghiệm tối ưu toàn cục (Brute-force)."""
    from churn_retention.modules.retention.budget import (
        compare_greedy_vs_optimal as _compare,
    )
    return _compare(
        customers=customers,
        predictions=predictions,
        budget=budget,
        actions=actions,
        margin=margin,
        horizon_months=horizon_months,
    )
