"""Owner-only SQLite files for local Dashboard state, without following file links."""

import os
import sqlite3
import stat
from pathlib import Path


class LocalStateError(ValueError):
    """Safe configuration failure, never includes a filesystem path."""


def connect_local(path: str | Path) -> sqlite3.Connection:
    path = Path(path)
    try:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        parent = path.parent.stat()
        if hasattr(os, "getuid") and (
            parent.st_uid != os.getuid() or parent.st_mode & 0o022
        ):
            raise LocalStateError(
                "Thư mục dữ liệu cần thuộc người vận hành và không cho người khác ghi."
            )
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
        if path.is_symlink():
            raise LocalStateError("Không dùng liên kết file cho dữ liệu Dashboard.")
        fd = os.open(path, flags, 0o600)
        try:
            info = os.fstat(fd)
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_nlink != 1
                or (hasattr(os, "getuid") and info.st_uid != os.getuid())
            ):
                raise LocalStateError("File dữ liệu Dashboard không an toàn.")
            if hasattr(os, "fchmod"):
                os.fchmod(fd, 0o600)
        finally:
            os.close(fd)
        connection = sqlite3.connect(str(path), timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        return connection
    except LocalStateError:
        raise
    except (OSError, sqlite3.Error):
        raise LocalStateError("Không thể mở dữ liệu Dashboard an toàn.") from None
