"""Password verification for the privileged camera-configuration action.

The dashboard itself remains an unprivileged process.  It may request the
fixed, root-owned publish command only after this small verifier has accepted
an operator-provided management password.  Only a salted PBKDF2 digest is
stored; neither the password nor the camera credentials are returned by the
HTTP API.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import stat
import tempfile
from pathlib import Path


class CameraAuthError(ValueError):
    pass


class CameraManagementAuthenticator:
    """Initialize once, then verify a local management password."""

    _ITERATIONS = 350_000
    _DKLEN = 32

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)

    @property
    def configured(self) -> bool:
        return self.path.is_file()

    @staticmethod
    def _validate_password(password: object) -> str:
        if not isinstance(password, str) or not 10 <= len(password) <= 256:
            raise CameraAuthError("管理口令需为 10 至 256 个字符")
        return password

    def _load(self) -> tuple[bytes, bytes]:
        try:
            status = self.path.lstat()
            if not stat.S_ISREG(status.st_mode) or stat.S_IMODE(status.st_mode) != 0o600:
                raise CameraAuthError("管理口令文件权限异常")
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or payload.get("version") != 1:
                raise CameraAuthError("管理口令文件格式异常")
            salt = base64.b64decode(str(payload["salt"]), validate=True)
            digest = base64.b64decode(str(payload["digest"]), validate=True)
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            if isinstance(error, CameraAuthError):
                raise
            raise CameraAuthError("无法读取管理口令") from error
        if len(salt) < 16 or len(digest) != self._DKLEN:
            raise CameraAuthError("管理口令文件格式异常")
        return salt, digest

    @classmethod
    def _derive(cls, password: str, salt: bytes) -> bytes:
        # PBKDF2 is available in both the production Conda interpreter and
        # the local macOS interpreter used for release tests.  It avoids a
        # runtime OpenSSL/scrypt feature mismatch while still applying a
        # deliberately expensive, salted one-way derivation.
        return hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), salt, cls._ITERATIONS,
            dklen=cls._DKLEN,
        )

    def _save(self, salt: bytes, digest: bytes) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(
            prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent
        )
        temporary = Path(name)
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                descriptor = -1
                json.dump(
                    {
                        "version": 1,
                        "salt": base64.b64encode(salt).decode("ascii"),
                        "digest": base64.b64encode(digest).decode("ascii"),
                    },
                    handle,
                    separators=(",", ":"),
                )
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
            os.chmod(self.path, 0o600)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            temporary.unlink(missing_ok=True)

    def authenticate_or_initialize(self, password: object) -> bool:
        """Return ``True`` only when this call performed initial setup."""

        value = self._validate_password(password)
        if not self.configured:
            salt = secrets.token_bytes(24)
            self._save(salt, self._derive(value, salt))
            return True
        salt, expected = self._load()
        if not hmac.compare_digest(self._derive(value, salt), expected):
            raise CameraAuthError("管理口令不正确")
        return False
