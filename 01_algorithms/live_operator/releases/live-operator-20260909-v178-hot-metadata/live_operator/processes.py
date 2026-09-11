"""Private lifecycle state and process identity helpers."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ProcessIdentity:
    pid: int
    start_token: str
    pgid: int | None = None
    owner_token: str | None = None

    @classmethod
    def from_value(cls, value: dict[str, Any]) -> "ProcessIdentity":
        return cls(
            pid=int(value["pid"]),
            start_token=str(value["start_token"]),
            pgid=int(value["pgid"]) if value.get("pgid") is not None else None,
            owner_token=str(value["owner_token"]) if value.get("owner_token") else None,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class StateStore:
    """Atomically persist non-secret operator lifecycle state."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)

    def load(self) -> dict[str, Any] | None:
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        if not isinstance(value, dict):
            raise ValueError("operator state must be a JSON object")
        return value

    def save(self, value: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(
            prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent
        )
        temporary = Path(name)
        try:
            if os.name == "posix":
                os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                descriptor = -1
                json.dump(value, handle, ensure_ascii=True, separators=(",", ":"))
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
            os.chmod(self.path, 0o600)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            temporary.unlink(missing_ok=True)

    def remove(self) -> None:
        self.path.unlink(missing_ok=True)

    @contextmanager
    def lock(self):
        lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = lock_path.open("a+b")
        try:
            if lock_path.stat().st_size == 0:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            try:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                handle.close()


def process_start_token(pid: int) -> str | None:
    if os.name == "posix":
        try:
            raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
            fields_after_comm = raw[raw.rfind(")") + 2 :].split()
            return fields_after_comm[19]
        except (OSError, IndexError):
            return None
    try:
        import psutil  # type: ignore

        return f"{psutil.Process(pid).create_time():.6f}"
    except Exception:
        return None


def capture_identity(
    process: subprocess.Popen[Any], *, owner_token: str | None = None
) -> ProcessIdentity:
    token = process_start_token(process.pid)
    if token is None:
        process.terminate()
        raise RuntimeError(f"cannot capture start identity for pid {process.pid}")
    pgid = os.getpgid(process.pid) if os.name == "posix" else None
    return ProcessIdentity(process.pid, token, pgid, owner_token)


def is_same_process(identity: ProcessIdentity) -> bool:
    return process_start_token(identity.pid) == identity.start_token


def owned_process_group_exists(identity: ProcessIdentity) -> bool:
    if os.name != "posix" or identity.pgid is None or not identity.owner_token:
        return False
    marker = f"JIAN_KONG_OWNER_TOKEN={identity.owner_token}".encode()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if os.getpgid(int(entry.name)) != identity.pgid:
                continue
            if marker in (entry / "environ").read_bytes().split(b"\0"):
                return True
        except (OSError, ProcessLookupError, PermissionError):
            continue
    return False


def stop_process(identity: ProcessIdentity, timeout: float = 10.0) -> None:
    if not is_same_process(identity) and not owned_process_group_exists(identity):
        return
    if os.name == "posix" and identity.pgid is not None:
        os.killpg(identity.pgid, signal.SIGTERM)
    else:
        os.kill(identity.pid, signal.SIGTERM)
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and (
        is_same_process(identity) or owned_process_group_exists(identity)
    ):
        time.sleep(0.05)
    if (is_same_process(identity) or owned_process_group_exists(identity)) and os.name == "posix":
        if identity.pgid is not None:
            os.killpg(identity.pgid, signal.SIGKILL)
        else:
            os.kill(identity.pid, signal.SIGKILL)
