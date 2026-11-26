"""White-Box KHDV Service Advisor: translates ML/TreeSHAP drivers and 19-feature
profiles into 3-tier CSKH explanations (Facts -> Assessment -> Recommendation),
extended telecom care services, and operator-reviewed call scripts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import pandas as pd

FEATURE_LABELS_VI: dict[str, str] = {
    "gender": "Giới tính",
    "SeniorCitizen": "Khách hàng cao tuổi",
    "Partner": "Có bạn đời / gia đình",
    "Dependents": "Có người phụ thuộc",
    "tenure": "Thời gian gắn bó (tháng)",
    "PhoneService": "Dịch vụ điện thoại",
    "MultipleLines": "Nhiều đường dây điện thoại",
    "InternetService": "Đường truyền Internet",
    "OnlineSecurity": "Bảo mật trực tuyến",
    "OnlineBackup": "Sao lưu đám mây",
    "DeviceProtection": "Bảo vệ thiết bị",
    "TechSupport": "Hỗ trợ kỹ thuật",
    "StreamingTV": "Truyền hình trực tuyến",
    "StreamingMovies": "Xem phim trực tuyến",
    "Contract": "Loại hợp đồng",
    "PaperlessBilling": "Hóa đơn điện tử",
    "PaymentMethod": "Phương thức thanh toán",
    "MonthlyCharges": "Cước hàng tháng",
    "TotalCharges": "Tổng cước tích lũy",
}

VALUE_LABELS_VI: dict[str, str] = {
    "Month-to-month": "Theo từng tháng",
    "One year": "Hợp đồng 1 năm",
    "Two year": "Hợp đồng 2 năm",
    "Fiber optic": "Cáp quang (Fiber optic)",
    "DSL": "Cáp đồng (DSL)",
    "No": "Chưa đăng ký",
    "Yes": "Đã đăng ký",
    "Electronic check": "Séc điện tử (thủ công)",
    "Mailed check": "Séc bưu điện (thủ công)",
    "Bank transfer (automatic)": "Chuyển khoản tự động",
    "Credit card (automatic)": "Thẻ tín dụng tự động",
    "No internet service": "Không dùng Internet",
    "No phone service": "Không dùng điện thoại",
}

HIGH_RISK_THRESHOLD = 0.65
NEW_TENURE_MAX_MONTHS = 6


@dataclass(frozen=True)
class ExtendedServiceMatch:
    """One suggested telecom care service expansion derived from customer profile columns."""

    code: str
    title: str
    trigger_columns: str
    description: str
    precondition: str


@dataclass(frozen=True)
class ServiceAdvisory:
    """Structured 3-tier KHDV explanation and Frontstage advisory for one customer."""

    customer_id: str
    facts: tuple[str, ...]
    risk_drivers_vi: tuple[str, ...]
    assessment_summary: str
    recommendation_summary: str
    service_channel: str
    extended_services: tuple[ExtendedServiceMatch, ...]
    call_script: str


def _get_str(row: Mapping[str, Any], key: str, default: str = "") -> str:
    val = row.get(key, default)
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return default
    return str(val).strip()


def _get_float(row: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    val = row.get(key, default)
    try:
        if val is None or pd.isna(val):
            return default
        return float(val)
    except (TypeError, ValueError):
        return default


def _get_int(row: Mapping[str, Any], key: str, default: int = 0) -> int:
    return int(round(_get_float(row, key, float(default))))


def service_channel_for_customer(action_code: str, churn_prob: float) -> str:
    """Route intervention package and risk level to a concrete Frontstage touchpoint."""
    if action_code == "A0":
        return "Kênh hỗ trợ tiêu chuẩn / Tự phục vụ (không chi ưu đãi)"
    if float(churn_prob) >= HIGH_RISK_THRESHOLD:
        return "Tổng đài CSKH Outbound 1-1 (Ưu tiên cao)"
    return "Zalo OA / Ứng dụng & Điện thoại viên tư vấn"


def match_extended_services(customer_row: Mapping[str, Any]) -> tuple[ExtendedServiceMatch, ...]:
    """Identify matching KHDV extended service opportunities from unused profile columns."""
    tenure = _get_int(customer_row, "tenure", 0)
    senior = _get_int(customer_row, "SeniorCitizen", 0)
    partner = _get_str(customer_row, "Partner")
    dependents = _get_str(customer_row, "Dependents")
    multi_lines = _get_str(customer_row, "MultipleLines")
    internet = _get_str(customer_row, "InternetService")
    backup = _get_str(customer_row, "OnlineBackup")
    device_prot = _get_str(customer_row, "DeviceProtection")
    streaming_tv = _get_str(customer_row, "StreamingTV")
    streaming_movies = _get_str(customer_row, "StreamingMovies")

    matches: list[ExtendedServiceMatch] = []

    if tenure <= NEW_TENURE_MAX_MONTHS:
        matches.append(
            ExtendedServiceMatch(
                code="EXT-1",
                title="Đồng hành giai đoạn đầu thuê bao mới",
                trigger_columns=f"tenure = {tenure} tháng (≤ {NEW_TENURE_MAX_MONTHS})",
                description="Kiểm tra chất lượng đường truyền từ xa, hướng dẫn dùng App quản lý Wi-Fi và giải đáp hóa đơn kỳ đầu.",
                precondition="Cần xác nhận mốc chăm sóc, kênh liên hệ và nhân sự CSKH phụ trách.",
            )
        )

    if senior == 1:
        matches.append(
            ExtendedServiceMatch(
                code="EXT-2",
                title="Kênh hỗ trợ thân thiện cho Người cao tuổi",
                trigger_columns="SeniorCitizen = 1",
                description="Ưu tiên kết nối gặp trực tiếp điện thoại viên (hạn chế bấm phím tự động), hướng dẫn bản chữ lớn và nhắc cước thuận tiện.",
                precondition="Cần xác nhận nhu cầu, bảo đảm quyền riêng tư và sự đồng ý trước khi liên hệ người thân.",
            )
        )

    if partner == "Yes" or dependents == "Yes" or multi_lines == "Yes":
        triggers = [
            name
            for name, cond in (
                ("Partner=Yes", partner == "Yes"),
                ("Dependents=Yes", dependents == "Yes"),
                ("MultipleLines=Yes", multi_lines == "Yes"),
            )
            if cond
        ]
        matches.append(
            ExtendedServiceMatch(
                code="EXT-3",
                title="Tư vấn Gói cước Gia đình & Đa thiết bị",
                trigger_columns=", ".join(triggers),
                description="Gợi ý gộp hóa đơn Internet cố định và nhiều đường dây cho hộ gia đình để tối ưu tổng chi phí.",
                precondition="Không suy diễn cứng thành phần hộ gia đình; cần đối chiếu chính sách gói gia đình thật.",
            )
        )

    if internet not in {"", "No"} and (backup == "No" or device_prot == "No"):
        missing = [
            col
            for col, val in (("OnlineBackup=No", backup), ("DeviceProtection=No", device_prot))
            if val == "No"
        ]
        matches.append(
            ExtendedServiceMatch(
                code="EXT-4",
                title="Tư vấn Sao lưu Đám mây & Bảo vệ Thiết bị",
                trigger_columns=", ".join(missing),
                description="Đề xuất gói lưu trữ đám mây kèm kiểm tra, hỗ trợ kỹ thuật định kỳ cho Modem/Router.",
                precondition="Làm rõ phạm vi hỗ trợ thiết bị (DeviceProtection không tự đồng nghĩa bảo hiểm đền bù phần cứng).",
            )
        )

    if internet == "Fiber optic" and (streaming_tv == "No" or streaming_movies == "No"):
        unused = [
            col
            for col, val in (("StreamingTV=No", streaming_tv), ("StreamingMovies=No", streaming_movies))
            if val == "No"
        ]
        matches.append(
            ExtendedServiceMatch(
                code="EXT-5",
                title="Tư vấn Gói Giải trí Cáp quang",
                trigger_columns=f"Fiber optic + {', '.join(unused)}",
                description="Gợi ý tích hợp nội dung truyền hình/phim trên đường truyền tốc độ cao để tăng giá trị cảm nhận thay cho giảm cước tiền mặt.",
                precondition="Cần xác nhận sở thích giải trí thực tế của khách hàng và bản quyền nội dung.",
            )
        )

    return tuple(matches)


def summarize_extended_services(customer_row: Mapping[str, Any]) -> str:
    """Compact pipe-separated titles of matched extended services for table/CSV export."""
    matched = match_extended_services(customer_row)
    return " | ".join(m.title for m in matched) if matched else "Không có gợi ý bổ sung"


def _extract_facts(customer_row: Mapping[str, Any]) -> tuple[str, ...]:
    contract = _get_str(customer_row, "Contract", "N/A")
    tenure = _get_int(customer_row, "tenure", 0)
    monthly = _get_float(customer_row, "MonthlyCharges", 0.0)
    internet = _get_str(customer_row, "InternetService", "No")
    tech = _get_str(customer_row, "TechSupport", "No")
    sec = _get_str(customer_row, "OnlineSecurity", "No")
    pay = _get_str(customer_row, "PaymentMethod", "N/A")
    senior = _get_int(customer_row, "SeniorCitizen", 0)

    facts = [
        f"Hợp đồng: {VALUE_LABELS_VI.get(contract, contract)} ({contract}), đã gắn bó {tenure} tháng.",
        f"Cước hàng tháng: ${monthly:.2f}/tháng; Thanh toán: {VALUE_LABELS_VI.get(pay, pay)}.",
    ]
    if internet != "No":
        facts.append(
            f"Internet: {VALUE_LABELS_VI.get(internet, internet)}; "
            f"TechSupport: {VALUE_LABELS_VI.get(tech, tech)}; "
            f"OnlineSecurity: {VALUE_LABELS_VI.get(sec, sec)}."
        )
    else:
        facts.append("Không sử dụng dịch vụ Internet (chỉ dùng dịch vụ thoại cố định).")
    if senior == 1:
        facts.append("Thuộc nhóm khách hàng cao tuổi (SeniorCitizen = 1).")
    return tuple(facts)


def _translate_drivers(
    customer_row: Mapping[str, Any],
    churn_prob: float,
    local_shap_df: Optional[pd.DataFrame],
) -> tuple[tuple[str, ...], str]:
    risk_level = "Rất cao" if churn_prob >= HIGH_RISK_THRESHOLD else ("Trung bình" if churn_prob >= 0.40 else "Thấp")

    if local_shap_df is not None and not local_shap_df.empty:
        pos_rows = local_shap_df[local_shap_df["shap_value"] > 0.01].head(3)
        neg_rows = local_shap_df[local_shap_df["shap_value"] < -0.01].head(2)
        drivers: list[str] = []
        for r in pos_rows.itertuples():
            fname = FEATURE_LABELS_VI.get(str(r.feature), str(r.feature))
            fval = VALUE_LABELS_VI.get(str(r.value), str(r.value))
            drivers.append(f"Đẩy rủi ro tăng: {fname} = {fval} (SHAP {r.shap_value:+.2f}).")
        for r in neg_rows.itertuples():
            fname = FEATURE_LABELS_VI.get(str(r.feature), str(r.feature))
            fval = VALUE_LABELS_VI.get(str(r.value), str(r.value))
            drivers.append(f"Yếu tố giữ chân: {fname} = {fval} (SHAP {r.shap_value:+.2f}).")
        if not drivers:
            drivers.append("Các yếu tố đóng góp TreeSHAP ở mức trung tính quanh ngưỡng nền.")
        summary = (
            f"Mức rủi ro ước lượng {risk_level} ({churn_prob:.1%}). "
            "Các yếu tố trên rút ra từ đóng góp Local TreeSHAP trên hồ sơ này "
            "(là dấu hiệu cảnh báo tương quan để nhân viên kiểm tra thực tế, không quy chụp nguyên nhân cứng nhắc)."
        )
        return tuple(drivers), summary

    # Fallback rule-based drivers when running without CatBoost SHAP
    fallback: list[str] = []
    if _get_str(customer_row, "Contract") == "Month-to-month":
        fallback.append("Dấu hiệu tăng rủi ro: Hợp đồng từng tháng không có ràng buộc kỳ hạn.")
    if _get_int(customer_row, "tenure", 0) <= NEW_TENURE_MAX_MONTHS:
        fallback.append("Dấu hiệu tăng rủi ro: Thuê bao mới sử dụng trong giai đoạn đầu.")
    if _get_str(customer_row, "InternetService") == "Fiber optic" and (
        _get_str(customer_row, "TechSupport") == "No" or _get_str(customer_row, "OnlineSecurity") == "No"
    ):
        fallback.append("Dấu hiệu tăng rủi ro: Dùng cáp quang cước cao nhưng thiếu hỗ trợ kỹ thuật/bảo mật.")
    if not fallback:
        fallback.append("Hồ sơ chưa ghi nhận yếu tố rủi ro nổi bật ở hợp đồng hoặc thời gian sử dụng.")
    summary = (
        f"Mức rủi ro ước lượng {risk_level} ({churn_prob:.1%}) trong chế độ mô phỏng. "
        "Nhận định mang tính tham khảo để nhân viên vận hành xem xét."
    )
    return tuple(fallback), summary


def _build_call_script(
    customer_id: str,
    action_code: str,
    action_name: str,
    monthly_charges: float,
    extended_services: tuple[ExtendedServiceMatch, ...],
) -> str:
    lead_ext = extended_services[0].title if extended_services else None
    ext_clause = (
        f" Ngoài ra, nếu anh/chị quan tâm, nhà mạng có thể giới thiệu thêm hướng '{lead_ext}' phù hợp với hồ sơ đang sử dụng."
        if lead_ext
        else ""
    )

    if action_code == "A0":
        return (
            f"[Bản nháp nội bộ — Không cần gọi ưu đãi trả phí] Khách hàng {customer_id} đang thuộc nhóm A0. "
            "Chỉ duy trì kênh CSKH tiêu chuẩn khi khách chủ động liên hệ."
        )
    if action_code == "A1":
        return (
            f"“Em chào anh/chị ({customer_id}), em gọi từ bộ phận CSKH. Nhận thấy anh/chị đang sử dụng gói Internet "
            f"cước ${monthly_charges:.0f}/tháng nhưng chưa kích hoạt đầy đủ hỗ trợ kỹ thuật/bảo mật, bên em xin phép kiểm tra nhanh "
            f"chất lượng đường truyền và hướng dẫn kích hoạt gói '{action_name}'.{ext_clause}”"
        )
    if action_code == "A2":
        return (
            f"“Em chào anh/chị ({customer_id}), em gọi từ bộ phận CSKH. Để giúp anh/chị thanh toán thuận tiện hơn hàng tháng "
            f"và tối ưu quyền lợi so với gói theo từng tháng hiện tại, bên em xin tư vấn phương án '{action_name}' "
            f"(anh/chị hoàn toàn chủ động lựa chọn).{ext_clause}”"
        )
    return (
        f"“Em chào anh/chị ({customer_id}), em gọi từ bộ phận CSKH dành cho khách hàng giá trị cao (${monthly_charges:.0f}/tháng). "
        f"Bên em xin gửi anh/chị đề xuất gói '{action_name}' theo chính sách hiện hành nhằm tri ân và đồng hành lâu dài.{ext_clause}”"
    )


def build_service_advisory(
    customer_row: Mapping[str, Any],
    recommendation_row: Mapping[str, Any],
    local_shap_df: Optional[pd.DataFrame] = None,
) -> ServiceAdvisory:
    """Build complete 3-tier KHDV explanation and service recommendation for one customer."""
    cid = _get_str(customer_row, "customerID", "UNKNOWN")
    monthly = _get_float(customer_row, "MonthlyCharges", 0.0)
    churn_prob = _get_float(recommendation_row, "churn_prob", 0.0)
    action_code = _get_str(recommendation_row, "action_code", "A0")
    action_name = _get_str(recommendation_row, "action_name", "Không can thiệp")
    net_val = _get_float(recommendation_row, "net_value", 0.0)
    breakeven = recommendation_row.get("breakeven_prob")

    facts = _extract_facts(customer_row)
    drivers_vi, assessment_summary = _translate_drivers(customer_row, churn_prob, local_shap_df)
    channel = service_channel_for_customer(action_code, churn_prob)
    extended = match_extended_services(customer_row)

    if action_code == "A0":
        rec_summary = (
            "Khuyến nghị A0 (Theo dõi tiêu chuẩn): Lợi ích kỳ vọng từ các gói trả phí chưa đủ bù chi phí "
            "hoặc rủi ro thấp hơn ngưỡng hòa vốn."
        )
    else:
        be_text = f"{float(breakeven):.1%}" if breakeven is not None and pd.notna(breakeven) else "N/A"
        rec_summary = (
            f"Khuyến nghị {action_code} ({action_name}): ΔEV = +${net_val:.2f}, "
            f"rủi ro {churn_prob:.1%} vượt ngưỡng hòa vốn p* = {be_text}. "
            "Cần nhân viên vận hành xem xét trước khi triển khai."
        )

    script = _build_call_script(cid, action_code, action_name, monthly, extended)

    return ServiceAdvisory(
        customer_id=cid,
        facts=facts,
        risk_drivers_vi=drivers_vi,
        assessment_summary=assessment_summary,
        recommendation_summary=rec_summary,
        service_channel=channel,
        extended_services=extended,
        call_script=script,
    )
