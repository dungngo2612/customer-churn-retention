"""Internationalisation helpers – Vietnamese (default) and English."""

import streamlit as st

TRANSLATIONS: dict[str, dict[str, str]] = {
    "vi": {
        # ── Navigation ──────────────────────────────────────────────────────
        "nav_overview": "Tổng quan",
        "nav_customers": "Khách hàng & dữ liệu",
        "nav_analysis": "Phân tích rời bỏ",
        "nav_plan": "Kế hoạch giữ chân",
        "nav_campaigns": "Chiến dịch",
        # ── Sidebar ─────────────────────────────────────────────────────────
        "sidebar_data_label": "Dữ liệu đang chọn",
        "sidebar_no_data": "Chưa có dữ liệu",
        "sidebar_view_label": "Kết quả đang xem",
        "sidebar_mock_badge": "Mô phỏng · không gọi API",
        "sidebar_sim_note": "Kinh tế và can thiệp là mô phỏng. Chưa gửi ưu đãi cho khách hàng.",
        "sidebar_customers_suffix": "khách hàng · Đã kiểm tra",
        "sidebar_reset_btn": "Làm mới phiên làm việc",
        # ── Overview ────────────────────────────────────────────────────────
        "ov_title": "Executive Dashboard",
        "ov_subtitle": "Tổng quan hiệu suất giữ chân khách hàng",
        "ov_caption": "Cập nhật thời gian thực · Dữ liệu mới nhất",
        "ov_breadcrumb": "🏠 **Dashboard** → Tổng quan",
        "ov_empty_title": "Bắt đầu từ dữ liệu khách hàng",
        "ov_empty_desc": (
            "Nhập tệp CSV hoặc chọn dữ liệu mẫu. Sau khi kiểm tra, chủ động "
            "chạy phân tích để xem nguy cơ rời bỏ."
        ),
        "ov_link_import": "Nhập dữ liệu khách hàng →",
        "ov_kpi_title": "🎯 KPI Chính",
        "ov_metric_analyzed": "📈 Khách đã phân tích",
        "ov_metric_high_risk": "⚠️ Nguy cơ cao",
        "ov_metric_active": "✅ Đang hoạt động",
        "ov_metric_campaigns": "🚀 Chiến dịch đã lưu",
        "ov_risk_title": "📊 Phân tích Nguy cơ",
        "ov_risk_caption": "Phân nhóm khách hàng theo điểm rời bỏ dự đoán · Chưa hiệu chuẩn",
        "ov_show_pie": "➕ Hiển thị biểu đồ tròn",
        "ov_pie_title": "#### 🥧 Phân bố theo tỷ lệ %",
        "ov_risk_link": "Xem kết quả và lý do →",
        "ov_plan_title": "💰 Kế hoạch & ROI",
        "ov_plan_caption": "Dựa trên phân tích hiện tại · Mô phỏng theo giả định",
        "ov_plan_expected": "#### Dự kiến hiệu quả",
        "ov_metric_net": "Lợi ích ròng",
        "ov_metric_intervene": "👥 Khách can thiệp",
        "ov_metric_cost": "💰 Chi phí",
        "ov_btn_plan": "📋 Lập kế hoạch chi tiết",
        "ov_btn_campaigns": "🚀 Xem chiến dịch",
        "ov_no_plan_info": "💡 **Chưa có kế hoạch can thiệp nào.** Điều chỉnh ngân sách hoặc giả định để xem phương án khả thi.",
        "ov_exec_toggle": "📋 Xem báo cáo điều hành chi tiết",
        "ov_exec_title": "### 📋 Báo cáo Điều hành",
        "ov_exec_caption": "Phân tích chi phí-hiệu quả theo từng loại can thiệp",
        "ov_exec_download": "Tải báo cáo tổng quan (.md)",
        "ov_workflow_title": "## 🚀 Luồng làm việc được đề xuất",
        "ov_step1_title": "### Bước 1: Chuẩn bị dữ liệu",
        "ov_step1_body": "**📊 Kiểm tra & làm sạch dữ liệu**",
        "ov_step1_caption": "Nhập CSV hoặc dùng dữ liệu mẫu. Đảm bảo dữ liệu đầy đủ trước khi phân tích.",
        "ov_step1_btn": "➡️ Đến Khách hàng",
        "ov_step2_title": "### Bước 2: Phân tích nguy cơ",
        "ov_step2_body": "**🔍 Hiểu nguyên nhân rời bỏ**",
        "ov_step2_caption": "Phân tích từng khách hoặc batch nhỏ. Xem điểm LLM và lý do dự đoán.",
        "ov_step2_btn": "➡️ Đến Phân tích",
        "ov_step3_title": "### Bước 3: Lập kế hoạch hành động",
        "ov_step3_body": "**💰 Tối ưu ngân sách giữ chân**",
        "ov_step3_caption": "Phân bổ ngân sách, chọn can thiệp hiệu quả nhất, lưu chiến dịch.",
        "ov_step3_btn": "➡️ Đến Kế hoạch",
        "ov_tip": "💡 **Mẹo:** Bắt đầu với dữ liệu mẫu để làm quen với luồng làm việc trước khi dùng dữ liệu thật.",
        "ov_insight_prefix": "**Insight:**",
        "ov_risk_low": "Thấp",
        "ov_risk_medium": "Trung bình",
        "ov_risk_high": "Cao",
        # ── Customers ───────────────────────────────────────────────────────
        "cu_eyebrow": "Không gian dữ liệu",
        "cu_title": "Khách hàng & dữ liệu",
        "cu_desc": "Kiểm tra nguồn dữ liệu trước khi phân tích. Tệp được nhập không tự tạo dự đoán.",
        "cu_tab_import": "Nhập dữ liệu",
        "cu_tab_eda": "📊 Khám phá dữ liệu (EDA)",
        "cu_tab_segments": "👥 Phân khúc khách hàng",
        "cu_tab_list": "Danh sách khách hàng",
        "cu_import_title": "Nhập danh sách khách hàng",
        "cu_import_desc": "CSV UTF-8 · tối đa 2 MB, 1.000 dòng · mã khách hàng duy nhất",
        "cu_file_uploader": "Chọn tệp dữ liệu khách hàng",
        "cu_format_expander": "Định dạng tệp & dữ liệu mẫu",
        "cu_format_body": (
            "Tệp cần đúng 19 thuộc tính, customerID và Churn đã quan sát. "
            "Không đưa nhãn dự đoán vào Churn. Chưa hỗ trợ sửa trực tiếp; hãy "
            "sửa tệp rồi nhập lại."
        ),
        "cu_download_sample": "Tải tệp CSV mẫu (dữ liệu giả)",
        "cu_validate_ok": "Đã kiểm tra: {n} dòng hợp lệ. Chưa chạy phân tích.",
        "cu_use_data_btn": "Sử dụng dữ liệu này",
        "cu_accepted": "Đã chọn dữ liệu. Kết quả trước đó vẫn được giữ nguyên.",
        "cu_fix_caption": (
            "Sửa các giá trị/cột được báo trong tệp rồi nhập lại. Dữ liệu đã "
            "chọn trước đó không bị thay thế."
        ),
        "cu_sample_title": "Khám phá bằng dữ liệu mẫu",
        "cu_sample_desc_suffix": "hồ sơ tổng hợp. Đây là dữ liệu giả, dùng để trải nghiệm luồng làm việc.",
        "cu_use_sample_btn": "Dùng dữ liệu mẫu",
        "cu_current_info": "Đang chọn: {source} · {n} khách hàng",
        "cu_link_analysis": "Tiếp tục phân tích rời bỏ →",
        "cu_link_analysis_simple": "Tiếp tục đến phân tích →",
        "cu_empty_list": "Chưa có danh sách khách hàng",
        "cu_empty_list_desc": "Chọn dữ liệu tại mục Nhập dữ liệu để bắt đầu.",
        "cu_badge_sample": "Dữ liệu mẫu · không phải khách hàng thật",
        "cu_badge_ok": "Dữ liệu đã kiểm tra",
        "cu_col_id": "Mã khách hàng",
        "cu_col_contract": "Hợp đồng",
        "cu_col_monthly": "Cước tháng (USD)",
        "cu_col_tenure": "Sử dụng (tháng)",
        "cu_col_churn": "Trạng thái quan sát",
        "cu_search_label": "Tìm theo mã khách hàng",
        "cu_search_placeholder": "Nhập một phần mã khách hàng",
        "cu_contract_label": "Hợp đồng",
        "cu_sort_label": "Sắp xếp",
        "cu_sort_id": "Mã khách tăng dần",
        "cu_sort_monthly": "Cước tháng giảm dần",
        "cu_sort_tenure": "Thời gian sử dụng giảm dần",
        "cu_all": "Tất cả",
        "cu_churn_yes": "Đã rời bỏ",
        "cu_churn_no": "Đang hoạt động",
        "cu_sample_badge": "Dữ liệu mẫu · không phải khách hàng thật",
        "cu_verified_badge": "Dữ liệu đã kiểm tra",
        "cu_contract_all": "Tất cả",
        "cu_sort_charges": "Cước tháng giảm dần",
        # ── Analysis ────────────────────────────────────────────────────────
        "an_title": "Phân tích Rời bỏ",
        "an_subtitle": "Hiểu nguyên nhân & dự đoán nguy cơ",
        "an_caption": "Phân tích khách hàng đã chọn · Xem kết quả và tiếp tục lập kế hoạch",
        "an_eyebrow": "Hiểu nguyên nhân & dự đoán nguy cơ",
        "an_desc": "Phân tích khách hàng đã chọn · Xem kết quả và tiếp tục lập kế hoạch",
        "an_breadcrumb": "🏠 **Dashboard** → Phân tích",
        "an_tab_run": "🚀 Chạy phân tích",
        "an_tab_results": "📊 Kết quả & lý do",
        "an_tab_history": "📚 Lịch sử",
        "an_empty_no_data": "Chưa chọn dữ liệu để phân tích",
        "an_empty_no_data_desc": "Nhập tệp hoặc chọn dữ liệu mẫu tại Khách hàng & dữ liệu.",
        "an_empty_title": "Chưa chọn dữ liệu để phân tích",
        "an_empty_desc": "Nhập tệp hoặc chọn dữ liệu mẫu tại Khách hàng & dữ liệu.",
        "an_link_choose_data": "Chọn dữ liệu →",
        "an_link_fix_data": "Kiểm tra lại dữ liệu →",
        "an_link_data": "Chọn dữ liệu →",
        "an_link_recheck": "Kiểm tra lại dữ liệu →",
        "an_badge_sample": "Dữ liệu mẫu · hồ sơ giả",
        "an_badge_ok": "Dữ liệu đã kiểm tra",
        "an_section_scope": "Chọn phạm vi phân tích",
        "an_scope_label": "Số khách trong lần chạy",
        "an_scope_single": "Một khách hàng",
        "an_scope_batch": "Batch nhỏ (tối đa 5)",
        "an_scope_batch_label": "Theo lô · tối đa 5 khách",
        "an_selectbox_single": "Khách hàng cần phân tích",
        "an_multiselect_batch": "Chọn khách hàng (tối đa 5)",
        "an_col_id": "Mã khách hàng",
        "an_col_contract": "Hợp đồng",
        "an_col_monthly": "Cước tháng (USD)",
        "an_mode_mock_label": "Mô phỏng · không gọi API",
        "an_mode_live_label": "API thật · có thể phát sinh phí",
        "an_info_mock": (
            "Chế độ mô phỏng: không gửi dữ liệu đến API. Kết quả chỉ để trải "
            "nghiệm hệ thống."
        ),
        "an_warn_live": (
            "API thật gửi 19 thuộc tính của khách đã chọn và dữ liệu tham "
            "chiếu đến Groq/Gemini. Có thể phát sinh phí, gồm lời gọi dự "
            "phòng."
        ),
        "an_confirm_label": "Tôi xác nhận phạm vi và chế độ phân tích đã chọn",
        "an_time_caption": (
            "Chỉ chạy khi bấm nút. Tối đa 120 giây/khách; lô 5 khách có thể "
            "cần 10 phút. Không tự gọi lại khi đổi trang."
        ),
        "an_run_btn": "Chạy phân tích",
        "an_new_btn": "Bắt đầu một lượt phân tích mới",
        "an_new_help": "Tạo yêu cầu mới để chủ động phân tích lại. Thao tác này chưa gọi API.",
        "an_spinner": "Đang phân tích {n} khách hàng và lưu kết quả…",
        "an_error_run": (
            "Không thể hoàn tất phân tích. Không tự gọi lại LLM; kết quả đã "
            "lưu được giữ nguyên."
        ),
        "an_empty_results": "Chưa có kết quả phân tích",
        "an_empty_results_desc": (
            "Chọn khách hàng và xác nhận chạy tại mục Chạy phân tích. Kết quả "
            "chỉ xuất hiện khi đã lưu thành công."
        ),
        "an_score_caption": "Điểm LLM chưa hiệu chuẩn. Phân nhóm: cao >70%, trung bình 40–70%, thấp <40%.",
        "an_mismatch_warn": (
            "Dữ liệu đang chọn khác snapshot này. Kết quả dưới đây và kế hoạch"
            " vẫn dùng dữ liệu đã lưu của lần phân tích, không dùng bản mới."
        ),
        "an_search_label": "Tìm mã khách hàng",
        "an_risk_label": "Nguy cơ",
        "an_action_label": "Đề xuất",
        "an_risk_all": "Tất cả",
        "an_risk_high": "Cao",
        "an_risk_medium": "Trung bình",
        "an_risk_low": "Thấp",
        "an_action_all": "Tất cả",
        "an_action_intervene": "Có can thiệp",
        "an_action_no_intervene": "Không can thiệp",
        "an_col_churn_score": "Điểm rời bỏ",
        "an_col_risk": "Nguy cơ",
        "an_col_recommend": "Đề xuất",
        "an_no_recommend": "Không thuộc diện can thiệp",
        "an_export_btn": "Xuất danh sách kết quả đã lọc (.csv)",
        "an_link_plan": "Lập kế hoạch từ lần phân tích này →",
        "an_group_expander": "Quan sát theo nhóm khách hàng",
        "an_group_caption": (
            "Tỷ lệ Churn dưới đây là dữ liệu đã quan sát trong snapshot; không"
            " phải kết quả LLM hay tác động nhân quả."
        ),
        "an_group_contract": "Hợp đồng",
        "an_group_internet": "Internet",
        "an_group_tech": "Hỗ trợ kỹ thuật",
        "an_group_payment": "Thanh toán",
        "an_group_churn_obs": "Churn quan sát",
        "an_history_section": "Các lần phân tích đã lưu",
        "an_history_caption": "Tối đa 50 công việc gần nhất. Mở lại không gọi LLM.",
        "an_history_empty": "Chưa có lịch sử",
        "an_history_empty_desc": "Lần phân tích sẽ được ghi lại khi bạn chủ động chạy.",
        "an_col_run": "Lần phân tích",
        "an_col_source": "Nguồn",
        "an_col_mode": "Chế độ",
        "an_col_status": "Trạng thái",
        "an_history_select": "Chọn lần phân tích",
        "an_history_open_btn": "Mở kết quả đã lưu",
        # ── Plan ────────────────────────────────────────────────────────────
        "pl_eyebrow": "Từ phân tích đến hành động",
        "pl_title": "Kế hoạch giữ chân",
        "pl_desc": "Xây dựng phương án ngân sách và lưu chiến dịch mô phỏng. Chưa thực thi ưu đãi.",
        "pl_empty": "Cần kết quả phân tích trước khi lập kế hoạch",
        "pl_empty_title": "Cần kết quả phân tích trước khi lập kế hoạch",
        "pl_empty_desc": "Mở một lần phân tích đã lưu hoặc chạy phân tích khách hàng đã chọn.",
        "pl_link_analysis": "Đến phân tích rời bỏ →",
        "pl_mismatch_warn": (
            "Phương án dùng snapshot của lần phân tích đang mở; dữ liệu mới "
            "nhập chưa được dùng cho kế hoạch này."
        ),
        "pl_step1": "1 · Thiết lập",
        "pl_step2": "2 · Xem phương án",
        "pl_step3": "3 · Tạo chiến dịch",
        "pl_settings_title": "Thiết lập ngân sách & giả định",
        "pl_settings_caption": "Bấm Áp dụng thiết lập để giữ các giá trị khi chuyển trang. Không gọi LLM.",
        "pl_budget_label": "Ngân sách (USD)",
        "pl_margin_label": "Biên lợi nhuận giả định",
        "pl_horizon_label": "Kỳ hạn đánh giá (tháng)",
        "pl_horizon_note": (
            "USD được giữ nguyên. Tỷ lệ chấp nhận và hiệu quả là giả định, "
            "chưa được đo lường từ chiến dịch."
        ),
        "pl_contact_cost": "Chi phí liên hệ (USD)",
        "pl_offer_cost": "Chi phí ưu đãi (USD)",
        "pl_acceptance_rate": "Tỷ lệ chấp nhận",
        "pl_efficacy": "Mức giảm rời bỏ giả định",
        "pl_apply_btn": "Áp dụng thiết lập",
        "pl_apply_success": "Đã áp dụng và giữ thiết lập. Chọn bước 2 để xem phương án.",
        "pl_actions_expander": "Chi phí và mức giảm kỳ vọng theo gói",
        "pl_col_package": "Gói",
        "pl_col_plan": "Phương án",
        "pl_col_expected_cost": "Chi phí kỳ vọng (USD)",
        "pl_col_reduction": "Mức giảm hiệu dụng",
        "pl_alloc_title": "Phương án phân bổ",
        "pl_budget_chart_title": "### 📊 Phân bổ ngân sách",
        "pl_no_intervene_warn": (
            "Ngân sách hiện tại chưa chọn được gói có lợi ích dương. Tăng ngân"
            " sách hoặc xem lại giả định tại bước 1."
        ),
        "pl_no_intervene_info": (
            "Không có phương án can thiệp lợi ích dương theo điều kiện và giả "
            "định hiện tại. Có thể lưu phương án không can thiệp."
        ),
        "pl_col_customer_id": "Mã khách hàng",
        "pl_col_action": "Phương án",
        "pl_col_churn_score": "Điểm rời bỏ",
        "pl_col_cost": "Chi phí (USD)",
        "pl_col_benefit": "Lợi ích (USD)",
        "pl_col_net": "Lợi ích ròng (USD)",
        "pl_next_step_btn": "Tiếp tục tạo chiến dịch →",
        "pl_sensitivity_expander": "Khảo sát ngân sách & độ nhạy",
        "pl_sensitivity_caption": (
            "Các phương án so sánh dùng cùng snapshot đã lưu; không dự đoán "
            "lại và không bảo đảm nghiệm tối ưu."
        ),
        "pl_summary_budget": "Ngân sách sử dụng",
        "pl_summary_customers": "Khách được chọn",
        "pl_summary_net": "Lợi ích ròng dự kiến",
        "pl_summary_roi": "ROI dự kiến",
        "pl_summary_roi_sim": "Mô phỏng",
        "pl_summary_roi_zero": "Không xác định khi chi phí bằng 0",
        "pl_camp_section": "Lưu phương án thành chiến dịch",
        "pl_camp_desc": (
            "Lưu snapshot bất biến của dữ liệu, kết quả và giả định. Chưa gửi "
            "ưu đãi hoặc thực thi can thiệp."
        ),
        "pl_camp_name_label": "Tên chiến dịch",
        "pl_camp_name_default": "Kế hoạch giữ chân khách hàng",
        "pl_camp_confirm": "Tôi xác nhận lưu phương án mô phỏng với nguồn và ngân sách trên",
        "pl_camp_save_btn": "Lưu chiến dịch",
        "pl_camp_error_missing": "Nhập tên chiến dịch và xác nhận phương án trước khi lưu.",
        "pl_camp_saved": "Đã lưu chiến dịch. Phương án chưa được thực thi.",
        "pl_camp_save_error": (
            "Không thể lưu chiến dịch. Dữ liệu đã lưu được giữ nguyên; bạn có "
            "thể thử lại."
        ),
        "pl_camp_link": "Mở chiến dịch đã lưu →",
        "pl_source_prefix": "**Nguồn:**",
        "pl_budget_prefix": "**Ngân sách:**",
        "pl_customers_prefix": "**Khách được chọn:**",
        "pl_cost_prefix": "**Chi phí dự kiến:**",
        "pl_net_prefix": "**Lợi ích ròng:**",
        # ── Campaigns ───────────────────────────────────────────────────────
        "ca_eyebrow": "Kế hoạch đã lưu",
        "ca_title": "Chiến dịch",
        "ca_desc": "Mở lại phương án và xuất danh sách can thiệp. Chiến dịch đã lưu chưa được thực thi.",
        "ca_tab_archive": "Chiến dịch đã lưu",
        "ca_tab_queue": "Công việc xử lý",
        "ca_empty": "Chưa có chiến dịch",
        "ca_empty_desc": "Sau khi phân tích, lập phương án ngân sách và lưu tại Kế hoạch giữ chân.",
        "ca_link_plan": "Đến kế hoạch giữ chân →",
        "ca_search_label": "Tìm theo tên chiến dịch",
        "ca_status_label": "Trạng thái",
        "ca_mode_label": "Nguồn kết quả",
        "ca_all": "Tất cả",
        "ca_col_campaign": "Chiến dịch",
        "ca_col_status": "Trạng thái",
        "ca_col_created": "Ngày tạo",
        "ca_col_budget": "Ngân sách (USD)",
        "ca_col_customers": "Khách được chọn",
        "ca_col_source": "Nguồn",
        "ca_open_label": "Mở chiến dịch",
        "ca_summary_budget": "Ngân sách",
        "ca_summary_cost": "Chi phí dự kiến",
        "ca_summary_net": "Lợi ích ròng",
        "ca_summary_customers": "Khách được chọn",
        "ca_summary_usd": "USD",
        "ca_summary_sim": "Mô phỏng",
        "ca_summary_not_executed": "Chưa thực thi can thiệp",
        "ca_scope_label": "Phạm vi xem và xuất CSV",
        "ca_scope_intervened": "Khách được can thiệp",
        "ca_scope_all": "Toàn bộ phương án đã lưu",
        "ca_col_cust_id": "Mã khách hàng",
        "ca_col_package": "Gói",
        "ca_col_plan": "Phương án",
        "ca_col_cost": "Chi phí (USD)",
        "ca_col_benefit": "Lợi ích (USD)",
        "ca_col_net": "Lợi ích ròng (USD)",
        "ca_export_note": "Xuất toàn bộ {n} dòng thuộc phạm vi trên; không chỉ trang đang xem.",
        "ca_download_btn": "Tải danh sách chiến dịch (.csv)",
        "ca_export_error": "Không thể ghi nhận yêu cầu xuất CSV. Dữ liệu chiến dịch vẫn được giữ nguyên.",
        "ca_export_note2": (
            "Đã yêu cầu xuất CSV chỉ ghi nhận thao tác tải, không có nghĩa đã "
            "triển khai chiến dịch."
        ),
        "ca_source_expander": "Nguồn kết quả & giả định đã lưu",
        "ca_source_data": "Nguồn dữ liệu: {name}",
        "ca_source_margin": "Biên lợi nhuận: {margin} · Kỳ hạn: {horizon} tháng",
        "ca_open_source_btn": "Mở lần phân tích nguồn",
        "ca_queue_title": "Công việc phân tích",
        "ca_queue_desc": (
            "Hàng đợi xử lý kỹ thuật hiện có. Không phải danh sách phân công "
            "nhân viên; không tự chạy lại API."
        ),
        "ca_queue_empty": "Chưa có công việc xử lý",
        "ca_queue_empty_desc": "Công việc xuất hiện sau khi bạn chủ động chạy phân tích.",
        "ca_col_job_id": "Mã công việc",
        "ca_col_job_status": "Trạng thái",
        "ca_col_created_at": "Tạo lúc",
        "ca_col_retries": "Số lần thử lại",
        "ca_job_select": "Xem tiến độ công việc",
        "ca_cust_col_id": "Mã khách hàng",
        "ca_cust_col_state": "Trạng thái",
        "ca_cust_col_msg": "Thông báo",
        "ca_recover_expander": "Khôi phục công việc bị gián đoạn",
        "ca_recover_warn": (
            "Chỉ đóng khi chắc chắn tiến trình xử lý đã dừng. Không thực hiện "
            "lại các lời gọi API chưa rõ kết quả."
        ),
        "ca_recover_select": "Công việc cần kiểm tra",
        "ca_recover_confirm": "Tôi xác nhận tiến trình xử lý công việc này đã dừng",
        "ca_recover_btn": "Đóng công việc bị gián đoạn",
        "ca_recover_success": "Đã khôi phục trạng thái từ dữ liệu đã lưu. Không gọi lại LLM.",
        "ca_recover_error": "Không thể khôi phục công việc. Dữ liệu đã lưu vẫn được giữ nguyên.",
        # ── App-level errors ─────────────────────────────────────────────────
        "app_safe_runtime_error": "Ứng dụng chỉ hoạt động qua launcher cục bộ được hỗ trợ.",
        "app_workspace_error": (
            "Không thể mở không gian làm việc an toàn. Dữ liệu đã lưu được giữ"
            " nguyên; không tự chạy lại phân tích."
        ),
    },
    "en": {
        # ── Navigation ──────────────────────────────────────────────────────
        "nav_overview": "Overview",
        "nav_customers": "Customers & data",
        "nav_analysis": "Churn analysis",
        "nav_plan": "Retention plan",
        "nav_campaigns": "Campaigns",
        # ── Sidebar ─────────────────────────────────────────────────────────
        "sidebar_data_label": "Selected dataset",
        "sidebar_no_data": "No data selected",
        "sidebar_view_label": "Current results",
        "sidebar_mock_badge": "Simulation · no API call",
        "sidebar_sim_note": "Economics and interventions are simulated. No offers sent to customers.",
        "sidebar_customers_suffix": "customers · Validated",
        "sidebar_reset_btn": "Reset workspace",
        # ── Overview ────────────────────────────────────────────────────────
        "ov_title": "Executive Dashboard",
        "ov_subtitle": "Customer retention performance overview",
        "ov_caption": "Real-time update · Latest data",
        "ov_breadcrumb": "🏠 **Dashboard** → Overview",
        "ov_empty_title": "Start with customer data",
        "ov_empty_desc": (
            "Upload a CSV file or choose sample data. After validation, run "
            "analysis to view churn risk."
        ),
        "ov_link_import": "Import customer data →",
        "ov_kpi_title": "🎯 Key KPIs",
        "ov_metric_analyzed": "📈 Customers analysed",
        "ov_metric_high_risk": "⚠️ High risk",
        "ov_metric_active": "✅ Active",
        "ov_metric_campaigns": "🚀 Saved campaigns",
        "ov_risk_title": "📊 Risk Analysis",
        "ov_risk_caption": "Customer segments by predicted churn score · Uncalibrated",
        "ov_show_pie": "➕ Show pie chart",
        "ov_pie_title": "#### 🥧 Distribution by percentage",
        "ov_risk_link": "View results and reasons →",
        "ov_plan_title": "💰 Plan & ROI",
        "ov_plan_caption": "Based on current analysis · Simulated assumptions",
        "ov_plan_expected": "#### Projected impact",
        "ov_metric_net": "Net benefit",
        "ov_metric_intervene": "👥 Intervened customers",
        "ov_metric_cost": "💰 Cost",
        "ov_btn_plan": "📋 Detailed planning",
        "ov_btn_campaigns": "🚀 View campaigns",
        "ov_no_plan_info": "💡 **No intervention plan yet.** Adjust budget or assumptions to explore feasible options.",
        "ov_exec_toggle": "📋 View detailed executive report",
        "ov_exec_title": "### 📋 Executive Report",
        "ov_exec_caption": "Cost-effectiveness analysis by intervention type",
        "ov_exec_download": "Download executive report (.md)",
        "ov_workflow_title": "## 🚀 Recommended workflow",
        "ov_step1_title": "### Step 1: Prepare data",
        "ov_step1_body": "**📊 Inspect & clean data**",
        "ov_step1_caption": "Upload CSV or use sample data. Ensure data is complete before analysis.",
        "ov_step1_btn": "➡️ Go to Customers",
        "ov_step2_title": "### Step 2: Analyse risk",
        "ov_step2_body": "**🔍 Understand churn reasons**",
        "ov_step2_caption": "Analyse individual or small batch. See LLM scores and prediction reasons.",
        "ov_step2_btn": "➡️ Go to Analysis",
        "ov_step3_title": "### Step 3: Plan action",
        "ov_step3_body": "**💰 Optimise retention budget**",
        "ov_step3_caption": "Allocate budget, choose the most effective interventions, save campaigns.",
        "ov_step3_btn": "➡️ Go to Plan",
        "ov_tip": "💡 **Tip:** Start with sample data to get familiar with the workflow before using real data.",
        "ov_insight_prefix": "**Insight:**",
        "ov_risk_low": "Low",
        "ov_risk_medium": "Medium",
        "ov_risk_high": "High",
        # ── Customers ───────────────────────────────────────────────────────
        "cu_eyebrow": "Data workspace",
        "cu_title": "Customers & data",
        "cu_desc": "Inspect the data source before analysis. Uploaded files do not auto-create predictions.",
        "cu_tab_import": "Import data",
        "cu_tab_eda": "📊 Exploratory Data (EDA)",
        "cu_tab_segments": "👥 Customer Segmentation",
        "cu_tab_list": "Customer list",
        "cu_import_title": "Import customer list",
        "cu_import_desc": "UTF-8 CSV · max 2 MB, 1,000 rows · unique customer ID",
        "cu_file_uploader": "Choose customer data file",
        "cu_format_expander": "File format & sample data",
        "cu_format_body": (
            "File must have exactly 19 attributes, customerID and observed Churn. "
            "Do not put predicted labels in Churn. Direct editing not supported; "
            "edit the file and re-upload."
        ),
        "cu_download_sample": "Download sample CSV (fake data)",
        "cu_validate_ok": "Validated: {n} valid rows. Analysis not yet run.",
        "cu_use_data_btn": "Use this data",
        "cu_accepted": "Data selected. Previous results are preserved.",
        "cu_fix_caption": (
            "Fix the values/columns reported in the file and re-upload. Previously "
            "selected data is not replaced."
        ),
        "cu_sample_title": "Explore with sample data",
        "cu_sample_desc_suffix": "synthetic profiles. This is fake data for exploring the workflow.",
        "cu_use_sample_btn": "Use sample data",
        "cu_current_info": "Selected: {source} · {n} customers",
        "cu_link_analysis": "Continue to churn analysis →",
        "cu_link_analysis_simple": "Continue to analysis →",
        "cu_empty_list": "No customer list yet",
        "cu_empty_list_desc": "Select data in the Import data section to get started.",
        "cu_badge_sample": "Sample data · not real customers",
        "cu_badge_ok": "Validated data",
        "cu_col_id": "Customer ID",
        "cu_col_contract": "Contract",
        "cu_col_monthly": "Monthly charges (USD)",
        "cu_col_tenure": "Tenure (months)",
        "cu_col_churn": "Observed status",
        "cu_search_label": "Search by customer ID",
        "cu_search_placeholder": "Enter part of customer ID",
        "cu_contract_label": "Contract",
        "cu_sort_label": "Sort",
        "cu_sort_id": "Customer ID ascending",
        "cu_sort_monthly": "Monthly charges descending",
        "cu_sort_tenure": "Tenure descending",
        "cu_all": "All",
        "cu_churn_yes": "Churned",
        "cu_churn_no": "Active",
        "cu_sample_badge": "Sample data · not real customers",
        "cu_verified_badge": "Validated data",
        "cu_contract_all": "All",
        "cu_sort_charges": "Monthly charges descending",
        # ── Analysis ────────────────────────────────────────────────────────
        "an_title": "Churn Analysis",
        "an_subtitle": "Understand reasons & predict risk",
        "an_caption": "Analyse selected customers · View results and continue planning",
        "an_eyebrow": "Understand reasons & predict risk",
        "an_desc": "Analyse selected customers · View results and continue planning",
        "an_breadcrumb": "🏠 **Dashboard** → Analysis",
        "an_tab_run": "🚀 Run analysis",
        "an_tab_results": "📊 Results & reasons",
        "an_tab_history": "📚 History",
        "an_empty_no_data": "No data selected for analysis",
        "an_empty_no_data_desc": "Upload a file or choose sample data in Customers & data.",
        "an_empty_title": "No data selected for analysis",
        "an_empty_desc": "Upload a file or choose sample data in Customers & data.",
        "an_link_choose_data": "Choose data →",
        "an_link_fix_data": "Review data →",
        "an_link_data": "Choose data →",
        "an_link_recheck": "Review data →",
        "an_badge_sample": "Sample data · fake profiles",
        "an_badge_ok": "Validated data",
        "an_section_scope": "Choose analysis scope",
        "an_scope_label": "Customers in this run",
        "an_scope_single": "Single customer",
        "an_scope_batch": "Small batch (max 5)",
        "an_scope_batch_label": "Batch · max 5 customers",
        "an_selectbox_single": "Customer to analyse",
        "an_multiselect_batch": "Choose customers (max 5)",
        "an_col_id": "Customer ID",
        "an_col_contract": "Contract",
        "an_col_monthly": "Monthly charges (USD)",
        "an_mode_mock_label": "Simulation · no API call",
        "an_mode_live_label": "Live API · may incur charges",
        "an_info_mock": (
            "Simulation mode: no data sent to API. Results are for exploring "
            "the system only."
        ),
        "an_warn_live": (
            "Live API sends 19 customer attributes and reference data to "
            "Groq/Gemini. May incur charges, including fallback calls."
        ),
        "an_confirm_label": "I confirm the selected scope and analysis mode",
        "an_time_caption": (
            "Only runs on button click. Max 120 s/customer; a 5-customer batch "
            "may need 10 min. Does not auto-retry on page change."
        ),
        "an_run_btn": "Run analysis",
        "an_new_btn": "Start a new analysis request",
        "an_new_help": "Create a new request to actively re-analyse. This does not call the API.",
        "an_spinner": "Analysing {n} customers and saving results…",
        "an_error_run": (
            "Could not complete analysis. LLM not auto-retried; saved results "
            "are preserved."
        ),
        "an_empty_results": "No analysis results yet",
        "an_empty_results_desc": (
            "Select customers and confirm a run in Run analysis. Results only "
            "appear after successful save."
        ),
        "an_score_caption": "LLM scores are uncalibrated. Segments: high >70%, medium 40–70%, low <40%.",
        "an_mismatch_warn": (
            "Selected data differs from this snapshot. Results below and the plan"
            " still use the saved data from this analysis run, not the new version."
        ),
        "an_search_label": "Search customer ID",
        "an_risk_label": "Risk",
        "an_action_label": "Recommendation",
        "an_risk_all": "All",
        "an_risk_high": "High",
        "an_risk_medium": "Medium",
        "an_risk_low": "Low",
        "an_action_all": "All",
        "an_action_intervene": "With intervention",
        "an_action_no_intervene": "No intervention",
        "an_col_churn_score": "Churn score",
        "an_col_risk": "Risk",
        "an_col_recommend": "Recommendation",
        "an_no_recommend": "Not eligible for intervention",
        "an_export_btn": "Export filtered results (.csv)",
        "an_link_plan": "Plan from this analysis →",
        "an_group_expander": "Observed by customer segment",
        "an_group_caption": (
            "Churn rate below is observed data in the snapshot; not LLM output "
            "or causal impact."
        ),
        "an_group_contract": "Contract",
        "an_group_internet": "Internet",
        "an_group_tech": "Tech support",
        "an_group_payment": "Payment method",
        "an_group_churn_obs": "Observed churn",
        "an_history_section": "Saved analysis runs",
        "an_history_caption": "Up to 50 most recent jobs. Re-opening does not call LLM.",
        "an_history_empty": "No history yet",
        "an_history_empty_desc": "Analysis runs are recorded when you actively run them.",
        "an_col_run": "Analysis run",
        "an_col_source": "Source",
        "an_col_mode": "Mode",
        "an_col_status": "Status",
        "an_history_select": "Choose analysis run",
        "an_history_open_btn": "Open saved results",
        # ── Plan ────────────────────────────────────────────────────────────
        "pl_eyebrow": "From analysis to action",
        "pl_title": "Retention plan",
        "pl_desc": "Build a budget plan and save a simulated campaign. Offers not yet executed.",
        "pl_empty": "Analysis results required before planning",
        "pl_empty_title": "Analysis results required before planning",
        "pl_empty_desc": "Open a saved analysis run or run analysis on the selected customers.",
        "pl_link_analysis": "Go to churn analysis →",
        "pl_mismatch_warn": (
            "Plan uses the snapshot from the open analysis run; newly imported "
            "data is not used for this plan."
        ),
        "pl_step1": "1 · Setup",
        "pl_step2": "2 · View plan",
        "pl_step3": "3 · Create campaign",
        "pl_settings_title": "Budget & assumption setup",
        "pl_settings_caption": "Click Apply settings to retain values across pages. No LLM call.",
        "pl_budget_label": "Budget (USD)",
        "pl_margin_label": "Assumed profit margin",
        "pl_horizon_label": "Evaluation horizon (months)",
        "pl_horizon_note": (
            "USD kept as-is. Acceptance and efficacy are assumed, not measured "
            "from campaigns."
        ),
        "pl_contact_cost": "Contact cost (USD)",
        "pl_offer_cost": "Offer cost (USD)",
        "pl_acceptance_rate": "Acceptance rate",
        "pl_efficacy": "Assumed churn reduction",
        "pl_apply_btn": "Apply settings",
        "pl_apply_success": "Settings applied and saved. Go to step 2 to view the plan.",
        "pl_actions_expander": "Expected cost and reduction by package",
        "pl_col_package": "Package",
        "pl_col_plan": "Option",
        "pl_col_expected_cost": "Expected cost (USD)",
        "pl_col_reduction": "Effective reduction",
        "pl_alloc_title": "Allocation plan",
        "pl_budget_chart_title": "### 📊 Budget allocation",
        "pl_no_intervene_warn": (
            "Current budget cannot select a package with positive benefit. Increase "
            "budget or review assumptions in step 1."
        ),
        "pl_no_intervene_info": (
            "No positive-benefit intervention option under current conditions and "
            "assumptions. You may save a no-intervention plan."
        ),
        "pl_col_customer_id": "Customer ID",
        "pl_col_action": "Option",
        "pl_col_churn_score": "Churn score",
        "pl_col_cost": "Cost (USD)",
        "pl_col_benefit": "Benefit (USD)",
        "pl_col_net": "Net benefit (USD)",
        "pl_next_step_btn": "Continue to create campaign →",
        "pl_sensitivity_expander": "Budget sweep & sensitivity",
        "pl_sensitivity_caption": (
            "Comparison options use the same saved snapshot; no re-prediction "
            "and no guaranteed optimal solution."
        ),
        "pl_summary_budget": "Budget used",
        "pl_summary_customers": "Customers selected",
        "pl_summary_net": "Projected net benefit",
        "pl_summary_roi": "Projected ROI",
        "pl_summary_roi_sim": "Simulation",
        "pl_summary_roi_zero": "Undefined when cost is zero",
        "pl_camp_section": "Save plan as campaign",
        "pl_camp_desc": (
            "Save an immutable snapshot of data, results, and assumptions. No "
            "offers sent or interventions executed."
        ),
        "pl_camp_name_label": "Campaign name",
        "pl_camp_name_default": "Customer retention plan",
        "pl_camp_confirm": "I confirm saving this simulated plan with the source and budget shown",
        "pl_camp_save_btn": "Save campaign",
        "pl_camp_error_missing": "Enter a campaign name and confirm the plan before saving.",
        "pl_camp_saved": "Campaign saved. Plan not yet executed.",
        "pl_camp_save_error": (
            "Could not save campaign. Saved data is preserved; you may try again."
        ),
        "pl_camp_link": "Open saved campaign →",
        "pl_source_prefix": "**Source:**",
        "pl_budget_prefix": "**Budget:**",
        "pl_customers_prefix": "**Customers selected:**",
        "pl_cost_prefix": "**Projected cost:**",
        "pl_net_prefix": "**Net benefit:**",
        # ── Campaigns ───────────────────────────────────────────────────────
        "ca_eyebrow": "Saved plans",
        "ca_title": "Campaigns",
        "ca_desc": "Re-open plans and export intervention lists. Saved campaigns are not yet executed.",
        "ca_tab_archive": "Saved campaigns",
        "ca_tab_queue": "Processing jobs",
        "ca_empty": "No campaigns yet",
        "ca_empty_desc": "After analysis, build a budget plan and save it in Retention plan.",
        "ca_link_plan": "Go to retention plan →",
        "ca_search_label": "Search by campaign name",
        "ca_status_label": "Status",
        "ca_mode_label": "Result source",
        "ca_all": "All",
        "ca_col_campaign": "Campaign",
        "ca_col_status": "Status",
        "ca_col_created": "Created",
        "ca_col_budget": "Budget (USD)",
        "ca_col_customers": "Customers selected",
        "ca_col_source": "Source",
        "ca_open_label": "Open campaign",
        "ca_summary_budget": "Budget",
        "ca_summary_cost": "Projected cost",
        "ca_summary_net": "Net benefit",
        "ca_summary_customers": "Customers selected",
        "ca_summary_usd": "USD",
        "ca_summary_sim": "Simulation",
        "ca_summary_not_executed": "Interventions not yet executed",
        "ca_scope_label": "Scope to view and export CSV",
        "ca_scope_intervened": "Intervened customers",
        "ca_scope_all": "Full saved plan",
        "ca_col_cust_id": "Customer ID",
        "ca_col_package": "Package",
        "ca_col_plan": "Option",
        "ca_col_cost": "Cost (USD)",
        "ca_col_benefit": "Benefit (USD)",
        "ca_col_net": "Net benefit (USD)",
        "ca_export_note": "Exporting all {n} rows in the selected scope; not just the current page.",
        "ca_download_btn": "Download campaign list (.csv)",
        "ca_export_error": "Could not record CSV export request. Campaign data is preserved.",
        "ca_export_note2": (
            "Requesting CSV export only records the download action; it does not "
            "mean the campaign has been deployed."
        ),
        "ca_source_expander": "Result source & saved assumptions",
        "ca_source_data": "Data source: {name}",
        "ca_source_margin": "Profit margin: {margin} · Horizon: {horizon} months",
        "ca_open_source_btn": "Open source analysis run",
        "ca_queue_title": "Analysis jobs",
        "ca_queue_desc": (
            "Current technical processing queue. Not a staff assignment list; "
            "does not auto-retry the API."
        ),
        "ca_queue_empty": "No processing jobs yet",
        "ca_queue_empty_desc": "Jobs appear after you actively run an analysis.",
        "ca_col_job_id": "Job ID",
        "ca_col_job_status": "Status",
        "ca_col_created_at": "Created at",
        "ca_col_retries": "Retry count",
        "ca_job_select": "View job progress",
        "ca_cust_col_id": "Customer ID",
        "ca_cust_col_state": "State",
        "ca_cust_col_msg": "Message",
        "ca_recover_expander": "Recover interrupted job",
        "ca_recover_warn": (
            "Only close when you are certain processing has stopped. Do not re-run "
            "API calls with unclear outcomes."
        ),
        "ca_recover_select": "Job to inspect",
        "ca_recover_confirm": "I confirm that processing for this job has stopped",
        "ca_recover_btn": "Close interrupted job",
        "ca_recover_success": "State recovered from saved data. LLM not retried.",
        "ca_recover_error": "Could not recover job. Saved data is preserved.",
        # ── App-level errors ─────────────────────────────────────────────────
        "app_safe_runtime_error": "App only works through a supported local launcher.",
        "app_workspace_error": (
            "Cannot open the secure workspace. Saved data is preserved; "
            "analysis is not auto-rerun."
        ),
    },
}


FALLBACK_LABELS: dict[str, tuple[str, str]] = {
    "cu_tab_import": ("Nhập dữ liệu", "Import data"),
    "cu_tab_list": ("Danh sách khách hàng", "Customer list"),
    "cu_tab_eda": ("📊 Khám phá dữ liệu (EDA)", "📊 Exploratory Data (EDA)"),
    "cu_tab_segments": ("👥 Phân khúc khách hàng", "👥 Customer Segmentation"),
    "an_tab_run": ("🚀 Chạy phân tích", "🚀 Run analysis"),
    "an_tab_results": ("📊 Kết quả & lý do", "📊 Results & reasons"),
    "an_tab_history": ("📚 Lịch sử", "📚 History"),
    "an_eyebrow": ("Hiểu nguyên nhân & dự đoán nguy cơ", "Understand drivers & predict risk"),
    "an_desc": ("Phân tích khách hàng đã chọn · Xem kết quả và tiếp tục lập kế hoạch", "Analyse selected customers · Inspect results and proceed to planning"),
    "an_link_data": ("Chọn dữ liệu →", "Choose data →"),
    "an_link_recheck": ("Kiểm tra lại dữ liệu →", "Review data →"),
    "pl_empty_title": ("Cần kết quả phân tích trước khi lập kế hoạch", "Analysis results required before planning"),
}


def t(key: str) -> str:
    """Return the translation for *key* in the current UI language.

    Reads ``st.session_state.get("ui_lang", "vi")`` on every call so that
    language changes take effect immediately on the next rerun.  If the key is
    missing from the active language dictionary the key itself is returned as a
    safe fallback.
    """
    lang = st.session_state.get("ui_lang", "vi")
    table = TRANSLATIONS.get(lang, TRANSLATIONS["vi"])
    if key in table:
        return table[key]
    if key in FALLBACK_LABELS:
        return FALLBACK_LABELS[key][1 if lang == "en" else 0]
    return key
