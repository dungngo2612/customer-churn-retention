"""Manual/CI local-only probe of a running single-admin demo (no LLM calls)."""

import http.client
import json
import socket
import sys


def probe(port: int) -> dict[str, object]:
    def status(path: str, headers: dict[str, str], method: str = "GET") -> int:
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        try:
            connection.request(method, path, headers=headers)
            response = connection.getresponse()
            code = response.status
            if code != 101:
                response.read()
            return code
        finally:
            connection.close()

    authority = f"127.0.0.1:{port}"
    assert status("/", {}) == 200
    assert status("/_stcore/health", {}) == 200
    checked = 0
    for path in (
        "/",
        "/_stcore/health",
        "/_stcore/upload_file/a/b",
        "/media/export.csv",
    ):
        for headers in (
            {"Host": "attacker.example"},
            {"Origin": "http://attacker.example"},
            {"X-Forwarded-For": "127.0.0.1"},
            {"Forwarded": "for=127.0.0.1"},
        ):
            assert status(path, headers) == 403
            checked += 1
    assert status("/_stcore/upload_file/a/b", {"Origin": "http://attacker.example"}, "POST") == 403
    assert status("/_stcore/upload_file/a/b", {"X-Forwarded-For": "127.0.0.1"}, "POST") == 403
    checked += 2
    websocket = {
        "Connection": "Upgrade",
        "Upgrade": "websocket",
        "Sec-WebSocket-Version": "13",
        "Sec-WebSocket-Key": "c3ludGhldGljLXByb2JlIQ==",
    }
    assert status("/_stcore/stream", websocket) == 403
    websocket["Origin"] = "http://attacker.example"
    assert status("/_stcore/stream", websocket) == 403
    websocket["Origin"] = f"http://{authority}"
    assert status("/_stcore/stream", websocket) == 101
    # Same host's LAN address must not reach the loopback listener.
    lan = sys.argv[2] if len(sys.argv) > 2 else None
    if lan:
        with socket.socket() as connection:
            connection.settimeout(2)
            assert connection.connect_ex((lan, port)) != 0
    return {
        "http_health": 200,
        "rejected_http_probes": checked,
        "rejected_websocket_probes": 2,
        "local_websocket": 101,
        "lan_connection_refused": bool(lan),
        "llm_calls": 0,
    }


if __name__ == "__main__":
    print(json.dumps(probe(int(sys.argv[1]))))
