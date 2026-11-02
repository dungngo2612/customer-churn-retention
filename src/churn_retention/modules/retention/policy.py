"""Retention policy and economic evaluation module for Telco Customer Churn."""

from dataclasses import dataclass
from math import isfinite
from numbers import Real


def _validate(value: float, *, lower: float, upper: float | None = None) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not isfinite(value)
        or value < lower
        or (upper is not None and value > upper)
    ):
        raise ValueError("Tham số kinh tế không hợp lệ; cần số hữu hạn trong phạm vi")


@dataclass(frozen=True)
class RetentionActionSpec:
    """Đặc tả tham số kinh tế và hiệu quả giả định của một hành động giữ chân."""

    code: str
    name: str
    contact_cost: float
    offer_cost: float
    acceptance_rate: float
    efficacy_given_accept: float
    description: str

    def __post_init__(self) -> None:
        _validate(self.contact_cost, lower=0)
        _validate(self.offer_cost, lower=0)
        _validate(self.acceptance_rate, lower=0, upper=1)
        _validate(self.efficacy_given_accept, lower=0, upper=1)
        _validate(self.contact_cost + self.acceptance_rate * self.offer_cost, lower=0)

    @property
    def expected_cost(self) -> float:
        """Chi phí kỳ vọng C_k = contact_cost + acceptance_rate * offer_cost."""
        return round(self.contact_cost + (self.acceptance_rate * self.offer_cost), 2)

    @property
    def effective_churn_reduction(self) -> float:
        """Mức giảm rủi ro tương đối hiệu quả r_k = u_k * e_k."""
        return round(self.acceptance_rate * self.efficacy_given_accept, 4)


RETENTION_CATALOG: dict[str, RetentionActionSpec] = {
    "A0": RetentionActionSpec(
        code="A0",
        name="Không can thiệp (Baseline)",
        contact_cost=0.0,
        offer_cost=0.0,
        acceptance_rate=0.0,
        efficacy_given_accept=0.0,
        description="Giữ nguyên hiện trạng, không phát sinh chi phí can thiệp.",
    ),
    "A1": RetentionActionSpec(
        code="A1",
        name="Hỗ trợ Kỹ thuật & Bảo mật chủ động",
        contact_cost=2.0,
        offer_cost=15.0,
        acceptance_rate=0.40,
        efficacy_given_accept=0.25,
        description="Chủ động hỗ trợ thiết bị và cấp quyền TechSupport/OnlineSecurity.",
    ),
    "A2": RetentionActionSpec(
        code="A2",
        name="Tư vấn Hợp đồng & Thanh toán tự động",
        contact_cost=2.0,
        offer_cost=20.0,
        acceptance_rate=0.35,
        efficacy_given_accept=0.40,
        description="Tư vấn chuyển đổi sang gói dài hạn và thanh toán tự động.",
    ),
    "A3": RetentionActionSpec(
        code="A3",
        name="Ưu đãi Giảm cước Cam kết Dài hạn",
        contact_cost=3.0,
        offer_cost=40.0,
        acceptance_rate=0.40,
        efficacy_given_accept=0.50,
        description="Tặng voucher giảm cước khi khách hàng cam kết gia hạn dịch vụ.",
    ),
}


def compute_proxy_clv(
    monthly_charges: float,
    horizon_months: int = 12,
    contribution_margin: float = 0.50,
) -> float:
    """Tính Giá trị thay thế CLV (Proxy CLV) dựa trên lợi nhuận đóng góp.

    Luôn tính trên MonthlyCharges gốc trước ưu đãi để tránh trừ trùng chi phí.
    """
    _validate(monthly_charges, lower=0)
    _validate(horizon_months, lower=1)
    _validate(contribution_margin, lower=0, upper=1)
    if int(horizon_months) != horizon_months or contribution_margin == 0:
        raise ValueError("Kỳ hạn phải nguyên dương, margin phải dương")
    value = float(monthly_charges) * int(horizon_months) * float(contribution_margin)
    _validate(value, lower=0)
    return round(value, 2)


class RetentionPolicy:
    """Đánh giá hiệu quả kinh tế cho quyết định giữ chân một khách hàng đơn lẻ."""

    def __init__(
        self,
        clv: float,
        churn_prob: float,
        cost: float,
        success_rate: float,
    ) -> None:
        _validate(clv, lower=0)
        _validate(churn_prob, lower=0, upper=1)
        _validate(cost, lower=0)
        _validate(success_rate, lower=0, upper=1)
        self.clv = float(clv)
        self.churn_prob = float(churn_prob)
        self.cost = float(cost)
        self.success_rate = float(success_rate)

    def expected_benefit(self) -> float:
        """Lợi ích kinh tế kỳ vọng trước chi phí: B = p * r * V."""
        return self.churn_prob * self.success_rate * self.clv

    def expected_net_value(self) -> float:
        """Giá trị ròng kỳ vọng tăng thêm: ENV = B - C."""
        return self.expected_benefit() - self.cost

    def breakeven_prob(self) -> float | None:
        """Ngưỡng hòa vốn xác suất rời bỏ: p* = C / (r * V)."""
        denom = self.success_rate * self.clv
        if denom <= 0:
            return None
        return round(self.cost / denom, 4)

    def should_intervene(self) -> bool:
        """Quyết định can thiệp khi ENV > 0."""
        return self.expected_net_value() > 0.0


if __name__ == "__main__":
    policy_a1 = RetentionPolicy(clv=480.0, churn_prob=0.70, cost=8.0, success_rate=0.10)
    print("Test policy A1:", policy_a1.expected_net_value(), "ENV (USD)")
