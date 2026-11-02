"""Single-admin demo boundaries. A browser session is never an identity."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any


class DemoSecurityError(ValueError):
    """Static error: never expose configuration paths or credentials."""

    def __init__(self) -> None:
        super().__init__("Unsafe demo configuration; startup refused.")


@dataclass(frozen=True)
class DemoPolicy:
    port: int
    state: Path
    live: bool = False


_active: DemoPolicy | None = None


def activate(policy: DemoPolicy) -> None:
    global _active
    if _active is not None:
        raise DemoSecurityError()
    _active = policy


def require_safe_runtime() -> None:
    """Guard all Dashboard reads, including direct `streamlit run` bypasses.

    Bare scripts/AppTest have no listening Runtime. They are not network servers.
    """
    from streamlit.runtime import Runtime

    if not Runtime.exists():
        return
    from streamlit import config

    if _active is None or config.get_option("server.address") != "127.0.0.1":
        raise DemoSecurityError()
    import os

    if (
        config.get_option("server.port") != _active.port
        or os.environ.get("CHURN_DASHBOARD_STATE") != str(_active.state)
        or os.environ.get("CHURN_DASHBOARD_ALLOW_LIVE") != str(int(_active.live))
        or os.environ.get("CHURN_DASHBOARD_PREVIEW") != str(int(not _active.live))
    ):
        raise DemoSecurityError()


class LoopbackOnly:
    """ASGI boundary covering HTTP, upload/media/download and WebSockets.

    Reject forwarding rather than trusting a local reverse proxy. Host checks
    also reject DNS rebinding; Origin checks reject cross-site WS connections.
    This is not authentication against another process on this machine.
    """

    def __init__(self, app: Any, port: int):
        self.app, self.authority = app, f"127.0.0.1:{port}"

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] not in {"http", "websocket"}:
            await self.app(scope, receive, send)
            return
        headers = scope.get("headers", [])
        hosts = [v for k, v in headers if k.lower() == b"host"]
        origins = [v for k, v in headers if k.lower() == b"origin"]
        peer = scope.get("client")
        safe = (
            peer is not None
            and peer[0] == "127.0.0.1"
            and hosts == [self.authority.encode()]
            and not any(
                k.lower() == b"forwarded" or k.lower().startswith(b"x-forwarded-")
                for k, _ in headers
            )
            and (
                not origins
                and scope["type"] == "http"
                or origins == [f"http://{self.authority}".encode()]
            )
        )
        if not safe:
            if scope["type"] == "websocket":
                await send({"type": "websocket.close", "code": 1008})
            else:
                await send(
                    {"type": "http.response.start", "status": 403, "headers": []}
                )
                await send({"type": "http.response.body", "body": b"Local demo only."})
            return
        await self.app(scope, receive, send)
