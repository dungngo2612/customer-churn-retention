"""User-level navigation for the main workspace acceptance tests."""

from streamlit.testing.v1 import AppTest


def go(at: AppTest, page: str) -> AppTest:
    at.switch_page(f"views/{page}.py").run()
    assert not at.exception
    return at


def sample_analysis(at: AppTest) -> AppTest:
    go(at, "customers")
    at.button(key="data.sample").click().run()
    return go(at, "analysis")


def plan_setup(at: AppTest) -> AppTest:
    go(at, "plan")
    at.radio(key="_plan_step").set_value("1 · Thiết lập").run()
    return at


def apply_settings(at: AppTest) -> AppTest:
    next(button for button in at.button if button.label == "Áp dụng thiết lập").click().run()
    return at


def save_campaign(at: AppTest) -> AppTest:
    go(at, "plan")
    at.radio(key="_plan_step").set_value("3 · Tạo chiến dịch").run()
    at.checkbox(key="plan.save_confirm").check()
    at.button(key="btn_save_e2e_camp").click().run()
    return at
