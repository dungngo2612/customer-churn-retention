"""Offline demo UI cases, including the previously failing all-A0 scenario."""

from pathlib import Path
from unittest.mock import Mock

import pandas as pd
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "apps/streamlit_llm/retention_page.py"


def test_demo_is_labelled_and_reruns_do_not_read_customer_data(monkeypatch) -> None:
    read = Mock(side_effect=AssertionError("Demo must not read a dataset"))
    monkeypatch.setattr(pd, "read_csv", read)
    at = AppTest.from_file(str(PAGE), default_timeout=15).run()
    assert not at.exception and not at.error
    assert any("DỮ LIỆU GIẢ" in info.value for info in at.info)
    assert any("chưa hiệu chuẩn" in warning.value for warning in at.warning)
    assert len(at.dataframe[0].value) == 2
    at.number_input(key="llm_retention.budget").set_value(0.0).run()
    assert not at.exception and set(at.dataframe[0].value["action_code"]) == {"A0"}
    at.number_input(key="llm_retention.budget").set_value(100.0).run()
    at.number_input(key="llm_retention.contact_multiplier").set_value(1000.0).run()
    assert not at.exception and not at.error
    assert any("Tất cả" in info.value for info in at.info)
    at.run()
    assert not at.exception
    read.assert_not_called()


def test_live_source_has_no_implicit_prediction_or_status() -> None:
    at = AppTest.from_file(str(PAGE), default_timeout=15).run()
    at.radio[0].set_value("Kết quả LLM vừa gửi").run()
    assert not at.exception and not at.error and not at.metric
    assert any("Chưa có kết quả LLM" in info.value for info in at.info)
