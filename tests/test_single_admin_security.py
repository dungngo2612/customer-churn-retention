"""Offline single-admin launch and all-route network boundary regression."""

import asyncio
import os
from pathlib import Path
from unittest.mock import Mock

import pytest

from churn_retention.dashboard import demo
from churn_retention.entrypoints.streamlit import demo_security as security


@pytest.fixture
def launch_env(monkeypatch):
    from streamlit import config

    values = {
        "server.address": None,
        "server.baseUrlPath": "",
        "server.sslCertFile": None,
    }
    monkeypatch.setattr(config, "get_option", lambda name: values.get(name))
    monkeypatch.delenv("CHURN_DASHBOARD_STATE", raising=False)
    monkeypatch.setattr(security, "_active", None)
    return values


@pytest.mark.parametrize(
    "address", ["0.0.0.0", "::", "::1", "localhost", "192.168.1.1", "unix://socket"]
)
def test_public_binding_refused_before_state_or_listener(launch_env, address, tmp_path):
    launch_env["server.address"] = address
    with pytest.raises(security.DemoSecurityError):
        demo.configure(8501, str(tmp_path / "no-create"), False)
    assert not (tmp_path / "no-create").exists()


@pytest.mark.parametrize("port", [0, 80, -1, 65536])
def test_port_fail_closed(launch_env, port):
    with pytest.raises(security.DemoSecurityError):
        demo.configure(port, None, False)


def test_default_overrides_inherited_live_and_removes_keys(launch_env, monkeypatch, tmp_path):
    monkeypatch.setenv("CHURN_DASHBOARD_ALLOW_LIVE", "1")
    monkeypatch.setenv("GROQ_API_KEY", "synthetic-unit-test-only")
    monkeypatch.setenv("CHURN_DASHBOARD_DEMO_RUN", "production-run")
    policy = demo.configure(8501, str(tmp_path / "private"), False)
    assert not policy.live
    assert os.environ["CHURN_DASHBOARD_PREVIEW"] == "1"
    assert os.environ["CHURN_DASHBOARD_ALLOW_LIVE"] == "0"
    assert "GROQ_API_KEY" not in os.environ
    assert "CHURN_DASHBOARD_DEMO_RUN" not in os.environ
    assert policy.state.name == "state.sqlite3"
    assert policy.state.parent.stat().st_mode & 0o777 == 0o700


def test_reuse_only_marked_demo_directory(launch_env, tmp_path):
    path = tmp_path / "demo"
    first = demo.demo_directory(str(path))
    (first / "state.sqlite3").write_bytes(b"synthetic state")
    assert demo.demo_directory(str(path)) == first
    assert (first / "state.sqlite3").read_bytes() == b"synthetic state"
    other = tmp_path / "production"
    other.mkdir(mode=0o700)
    (other / "state.sqlite3").write_bytes(b"must not touch")
    with pytest.raises(security.DemoSecurityError):
        demo.demo_directory(str(other))
    assert (other / "state.sqlite3").read_bytes() == b"must not touch"


def test_refuse_environment_database_override(launch_env, monkeypatch, tmp_path):
    monkeypatch.setenv("CHURN_DASHBOARD_STATE", str(tmp_path / "production.db"))
    with pytest.raises(security.DemoSecurityError):
        demo.configure(8501, None, False)
    assert not (tmp_path / "production.db").exists()


@pytest.mark.parametrize("answer", ["", "yes", "enable live api"])
def test_live_requires_exact_terminal_confirmation(launch_env, monkeypatch, answer):
    monkeypatch.setattr("builtins.input", lambda _: answer)
    with pytest.raises(security.DemoSecurityError):
        demo.configure(8501, None, True)


def test_confirmed_live_still_requires_existing_ui_consent(launch_env, monkeypatch, tmp_path):
    monkeypatch.setattr("builtins.input", lambda _: "ENABLE LIVE API")
    policy = demo.configure(8501, str(tmp_path / "demo"), True)
    assert policy.live and os.environ["CHURN_DASHBOARD_PREVIEW"] == "0"
    assert os.environ["CHURN_DASHBOARD_ALLOW_LIVE"] == "1"


@pytest.mark.parametrize(
    "kind,path",
    [
        ("http", "/"),
        ("http", "/_stcore/health"),
        ("http", "/_stcore/upload_file/a/b"),
        ("http", "/media/export.csv"),
        ("websocket", "/_stcore/stream"),
    ],
)
@pytest.mark.parametrize(
    "attack",
    [
        "remote",
        "host",
        "duplicate-host",
        "origin",
        "forwarded",
        "forwarded-for",
        "no-client",
    ],
)
def test_all_routes_reject_remote_rebinding_proxy_and_cross_origin(kind, path, attack):
    headers = [(b"host", b"127.0.0.1:8501"), (b"origin", b"http://127.0.0.1:8501")]
    client = ("127.0.0.1", 45678)
    if attack == "remote":
        client = ("192.0.2.1", 123)
    elif attack == "host":
        headers[0] = (b"host", b"attacker.example:8501")
    elif attack == "duplicate-host":
        headers.append(headers[0])
    elif attack == "origin":
        headers[1] = (b"origin", b"http://attacker.example")
    elif attack == "forwarded":
        headers.append((b"forwarded", b"for=192.0.2.1"))
    elif attack == "forwarded-for":
        headers.append((b"x-forwarded-for", b"127.0.0.1"))
    else:
        client = None
    called, messages = [], []

    async def app(*args):
        called.append(1)

    async def send(item):
        messages.append(item)

    asyncio.run(
        security.LoopbackOnly(app, 8501)(
            {"type": kind, "path": path, "headers": headers, "client": client},
            None,
            send,
        )
    )
    assert not called
    assert messages[0] == (
        {"type": "websocket.close", "code": 1008}
        if kind == "websocket"
        else {"type": "http.response.start", "status": 403, "headers": []}
    )


@pytest.mark.parametrize("kind", ["http", "websocket"])
def test_local_request_allowed(kind):
    called = []

    async def app(*args):
        called.append(1)

    asyncio.run(
        security.LoopbackOnly(app, 8501)(
            {
                "type": kind,
                "headers": [
                    (b"host", b"127.0.0.1:8501"),
                    (b"origin", b"http://127.0.0.1:8501"),
                ],
                "client": ("127.0.0.1", 22),
            },
            None,
            None,
        )
    )
    assert called == [1]


def test_runtime_bypass_denied_before_dashboard_data(monkeypatch):
    from streamlit.runtime import Runtime

    monkeypatch.setattr(Runtime, "exists", lambda: True)
    monkeypatch.setattr(security, "_active", None)
    with pytest.raises(security.DemoSecurityError):
        security.require_safe_runtime()


def test_launcher_installs_actual_backend_middleware(launch_env, monkeypatch, tmp_path):
    import inspect

    from streamlit.web import bootstrap
    from streamlit.web.server import starlette

    backend = inspect.getmodule(starlette.UvicornServer)
    original = backend.create_starlette_app
    load, run = Mock(), Mock()
    monkeypatch.setattr(bootstrap, "load_config_options", load)
    monkeypatch.setattr(bootstrap, "run", run)
    # Undo mutation after test: assert the actual server's global factory changes.
    monkeypatch.setattr(backend, "create_starlette_app", original)
    policy = security.DemoPolicy(8501, tmp_path / "state.sqlite3")
    demo.start(policy)
    assert backend.create_starlette_app is not original
    assert run.call_args.args[2] == []
    options = load.call_args.args[0]
    assert options["server.address"] == "127.0.0.1"
    assert options["server.enableXsrfProtection"]
    assert options["server.enableCORS"]
    assert options["server.maxUploadSize"] == 2
    assert options["server.fileWatcherType"] == "none"
    assert options["client.showErrorDetails"] == "none"


@pytest.mark.parametrize(
    "relative",
    [
        "app.py",
        "views/overview.py",
        "views/customers.py",
        "views/analysis.py",
        "views/plan.py",
        "views/campaigns.py",
    ],
)
def test_raw_streamlit_app_has_no_data_controls_without_launcher(monkeypatch, relative):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setattr(security, "_active", None)
    page = Path(__file__).resolve().parents[1] / "src/churn_retention/dashboard" / relative
    at = AppTest.from_file(str(page), default_timeout=30).run()
    assert not at.exception and len(at.error) == 1
    assert not at.tabs and not at.file_uploader and not at.button


def test_runtime_configuration_drift_is_denied(launch_env, monkeypatch, tmp_path):
    from streamlit.runtime import Runtime

    monkeypatch.setattr(Runtime, "exists", lambda: True)
    policy = demo.configure(8501, str(tmp_path / "demo"), False)
    security.activate(policy)
    launch_env.update({"server.address": "127.0.0.1", "server.port": 8501})
    security.require_safe_runtime()
    monkeypatch.setenv("CHURN_DASHBOARD_ALLOW_LIVE", "1")
    with pytest.raises(security.DemoSecurityError):
        security.require_safe_runtime()


@pytest.mark.parametrize("option", ["0.0.0.0", "::", "192.0.2.1"])
def test_real_cli_refuses_public_config_without_opening_listener(option, tmp_path):
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env.pop("CHURN_DASHBOARD_STATE", None)
    env["PYTHONPATH"] = str(root / "src")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["STREAMLIT_SERVER_ADDRESS"] = option
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "churn_retention.dashboard.demo",
            "--state-dir",
            str(tmp_path / "never-created"),
        ],
        cwd=root,
        check=False,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 1
    assert "Safe local demo could not start" in result.stderr
    assert not (tmp_path / "never-created").exists()
    assert "Traceback" not in result.stderr
    assert str(tmp_path) not in result.stdout + result.stderr
