"""Fail-closed local demo launcher: python -m churn_retention.dashboard.demo."""

import argparse
import os
import stat
import tempfile
from pathlib import Path
from typing import Any

from churn_retention.entrypoints.streamlit.demo_security import (
    DemoPolicy,
    DemoSecurityError,
    LoopbackOnly,
    activate,
)

MARKER = "single-admin-demo-v1\n"


def demo_directory(value: str | None) -> Path:
    path = Path(value) if value else Path(tempfile.mkdtemp(prefix="churn-admin-demo-")).resolve()
    if path.is_symlink():
        raise DemoSecurityError()
    # Resolve parents too; only private, owned directories can be reused.
    if path.absolute() != path.resolve():
        raise DemoSecurityError()
    if not path.exists():
        path.mkdir(mode=0o700, parents=False)
    info = path.stat()
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.getuid()
        or stat.S_IMODE(info.st_mode) != 0o700
    ):
        raise DemoSecurityError()
    marker = path / ".single-admin-demo"
    if not marker.exists():
        if any(path.iterdir()):
            raise DemoSecurityError()
        fd = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as stream:
            stream.write(MARKER)
    if marker.is_symlink() or marker.read_text() != MARKER:
        raise DemoSecurityError()
    return path


def configure(port: int, state_dir: str | None, live: bool) -> DemoPolicy:
    # Streamlit's CLI parses some STREAMLIT_* variables separately from its
    # programmatic bootstrap; reject unsafe intent explicitly before any writes.
    for name in ("STREAMLIT_SERVER_ADDRESS", "STREAMLIT_BROWSER_SERVER_ADDRESS"):
        if os.environ.get(name, "127.0.0.1") != "127.0.0.1":
            raise DemoSecurityError()
    if os.environ.get("STREAMLIT_SERVER_BASE_URL_PATH"):
        raise DemoSecurityError()
    if not 1024 <= port <= 65535 or "CHURN_DASHBOARD_STATE" in os.environ:
        raise DemoSecurityError()
    from streamlit import config

    if config.get_option("server.address") not in {None, "127.0.0.1"}:
        raise DemoSecurityError()
    if config.get_option("server.baseUrlPath") or config.get_option("server.sslCertFile"):
        raise DemoSecurityError()
    if live and input("Type ENABLE LIVE API to permit paid calls: ") != "ENABLE LIVE API":
        raise DemoSecurityError()
    path = demo_directory(state_dir)
    os.environ["CHURN_DASHBOARD_STATE"] = str(path / "state.sqlite3")
    os.environ["CHURN_DASHBOARD_ALLOW_LIVE"] = str(int(live))
    os.environ["CHURN_DASHBOARD_PREVIEW"] = str(int(not live))
    os.environ.pop("CHURN_DASHBOARD_DEMO_RUN", None)
    if not live:
        for name in (
            "GROQ_API_KEY",
            "GEMINI_API_KEY",
            "GOOGLE_API_KEY",
            "OPENAI_API_KEY",
        ):
            os.environ.pop(name, None)
    return DemoPolicy(port, path / "state.sqlite3", live)


def start(policy: DemoPolicy) -> None:
    from streamlit.web import bootstrap
    from streamlit.web.server import starlette

    # This adapter is deliberately fail-closed if Streamlit changes its server.
    if not callable(getattr(starlette, "create_starlette_app", None)):
        raise DemoSecurityError()
    import inspect

    backend: Any = inspect.getmodule(starlette.UvicornServer)
    factory = starlette.create_starlette_app
    if backend is None or getattr(backend, "create_starlette_app", None) is not factory:
        raise DemoSecurityError()

    def guarded_factory(runtime: Any) -> LoopbackOnly:
        from streamlit import config

        if config.get_option("server.address") != "127.0.0.1":
            raise DemoSecurityError()
        return LoopbackOnly(factory(runtime), policy.port)

    options = {
        "server.address": "127.0.0.1",
        "server.port": policy.port,
        "browser.serverAddress": "127.0.0.1",
        "browser.serverPort": policy.port,
        "server.headless": True,
        "server.enableCORS": True,
        "server.enableXsrfProtection": True,
        "server.enableStaticServing": False,
        "server.fileWatcherType": "none",
        "server.maxUploadSize": 2,
        "server.maxMessageSize": 16,
        "client.showErrorDetails": "none",
        "client.toolbarMode": "minimal",
        "browser.gatherUsageStats": False,
        "theme.base": "light",
        "theme.primaryColor": "#3949ab",
        "theme.backgroundColor": "#f5f6fa",
        "theme.secondaryBackgroundColor": "#ffffff",
        "theme.textColor": "#17243b",
        "theme.font": "sans serif",
    }
    bootstrap.load_config_options(options)
    activate(policy)
    backend.create_starlette_app = guarded_factory
    bootstrap.run(str(Path(__file__).with_name("app.py")), False, [], options)


def main() -> None:
    parser = argparse.ArgumentParser(description="Local single-admin demo only.")
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--state-dir", help="Private demo-only directory (0700).")
    parser.add_argument("--enable-live", action="store_true")
    args = parser.parse_args()
    try:
        start(configure(args.port, args.state_dir, args.enable_live))
    except Exception:  # noqa: BLE001 -- no config paths/secrets in startup errors
        parser.exit(1, "Safe local demo could not start; configuration refused.\n")


if __name__ == "__main__":
    main()
