"""Application I/O guards; never log or include rejected values in exceptions."""

import html
import os
import re
from collections.abc import Mapping, Sequence
from urllib.parse import unquote

_KEY_NAMES = ("GROQ_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY")
_CREDENTIAL = re.compile(
    r"(?<![A-Za-z0-9_-])(?:gsk_[A-Za-z0-9]{20,}|AIza[A-Za-z0-9_-]{30,}|sk-(?:proj-)?[A-Za-z0-9_-]{24,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)"
)


class SecurityBoundaryError(ValueError):
    """Static failure with no customer values, keys or provider content."""


def reject_credentials(value: object) -> None:
    """Reject known configured keys and recognizable credential formats, without rewriting data.

    Short placeholder environment values are not credentials; avoid matching ordinary
    letters in mock test configurations. No .env files or external services are read.
    """
    if isinstance(value, Mapping):
        for key, item in value.items():
            reject_credentials(key)
            reject_credentials(item)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            reject_credentials(item)
        return
    if not isinstance(value, str):
        return
    decoded = html.unescape(unquote(value))
    secrets = tuple(
        v for name in _KEY_NAMES if len(v := os.environ.get(name, "").strip()) >= 8
    )
    if (
        any(secret in value or secret in decoded for secret in secrets)
        or _CREDENTIAL.search(value)
        or _CREDENTIAL.search(decoded)
    ):
        raise SecurityBoundaryError(
            "Nội dung có thể chứa credential; không được lưu hoặc hiển thị."
        )


def safe_source_name(value: str) -> str:
    """Upload names are display metadata only, never filesystem paths."""
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > 255
        or any(ord(c) < 32 or ord(c) == 127 for c in value)
    ):
        raise SecurityBoundaryError("Tên nguồn dữ liệu không hợp lệ.")
    reject_credentials(value)
    # Browsers normally send a basename; remove directories from custom clients too.
    name = value.replace("\\", "/").rsplit("/", 1)[-1]
    if name in {"", ".", ".."}:
        raise SecurityBoundaryError("Tên nguồn dữ liệu không hợp lệ.")
    return name
