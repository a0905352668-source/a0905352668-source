"""Descriptor-pinned publication and lock authorities for the privileged reload.

Every ancestor is opened with O_DIRECTORY|O_NOFOLLOW. All mutations are relative
to the retained parent descriptor. Revalidation rejects renamed/replaced parents.
Lock order: root reload transaction lock -> the normal operator state lock.
The transaction lock remains held after releasing the state lock, through systemd
restoration. Other lifecycle consumers still synchronize on the same state lock.
"""
from __future__ import annotations

import fcntl
import json
import os
import signal
import stat
import uuid
from contextlib import contextmanager
from pathlib import Path

from live_operator.processes import StateStore


def identity(value):
    return value.st_dev, value.st_ino


def regular(value):
    if not stat.S_ISREG(value.st_mode) or value.st_nlink != 1:
        raise ValueError("file authority must be regular with exactly one link")


@contextmanager
def publication_signal_guard():
    """Defer handled exit signals until the rename and retained pin agree.

    The old mask is restored even on I/O failure. A pending handled signal then
    reaches the caller with either the old pin or a proven known-publication pin.
    """
    old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM, signal.SIGHUP})
    try:
        yield
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)


class PinnedDirectory:
    def __init__(self, path: Path, *, trusted_owner: int | None = None):
        self.path = Path(path)
        self.fds = []
        self.chain = []
        self.trusted_owner = trusted_owner
        if not self.path.is_absolute() or ".." in self.path.parts:
            raise ValueError("directory authority must be absolute without traversal")
        try:
            fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            self.fds.append(fd)
            for name in self.path.parts[1:]:
                parent = fd
                fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
                self.fds.append(fd)
                self.chain.append((parent, name, identity(os.fstat(fd))))
            self.fd = fd
            self.verify()
        except BaseException:
            self.close()
            raise

    def verify(self):
        for parent, name, expected in self.chain:
            value = os.stat(name, dir_fd=parent, follow_symlinks=False)
            if not stat.S_ISDIR(value.st_mode) or identity(value) != expected:
                raise ValueError("directory authority changed")
        details = os.fstat(self.fd)
        if self.trusted_owner is not None and (
                details.st_uid != self.trusted_owner or stat.S_IMODE(details.st_mode) & 0o022):
            raise ValueError("trusted directory must be owner-controlled and not group/world writable")

    def close(self):
        for fd in reversed(self.fds):
            os.close(fd)
        self.fds.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class PinnedFile:
    def __init__(self, path: Path, *, symlink: bool = False, parent: PinnedDirectory | None = None):
        self.path = Path(path)
        if parent is not None and parent.path != self.path.parent:
            raise ValueError("file and pinned parent disagree")
        self.owns_parent = parent is None
        self.parent = parent or PinnedDirectory(self.path.parent)
        self.name = self.path.name
        self.symlink = symlink
        try:
            self.metadata = self._stat()
            if symlink:
                if not stat.S_ISLNK(self.metadata.st_mode):
                    raise ValueError("selector must be a symlink")
                self.content = os.readlink(self.name, dir_fd=self.parent.fd)
            else:
                self.content, self.metadata = self.read_with_metadata()
            self.expected = identity(self.metadata)
        except BaseException:
            self.close()
            raise

    def _stat(self):
        self.parent.verify()
        return os.stat(self.name, dir_fd=self.parent.fd, follow_symlinks=False)

    def read_with_metadata(self):
        self.parent.verify()
        fd = os.open(self.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=self.parent.fd)
        try:
            details = os.fstat(fd)
            regular(details)
            if details.st_size > 64 * 1024 * 1024:
                raise ValueError("configuration file exceeds bounded read size")
            with os.fdopen(fd, "rb") as stream:
                fd = -1
                content = stream.read(64 * 1024 * 1024 + 1)
            if len(content) > 64 * 1024 * 1024 or identity(self._stat()) != identity(details):
                raise ValueError("file authority changed during read")
            return content, details
        finally:
            if fd >= 0:
                os.close(fd)

    def verify(self, *, content=True):
        current = self._stat()
        if identity(current) != self.expected:
            raise ValueError("file authority changed")
        if self.symlink:
            if not stat.S_ISLNK(current.st_mode):
                raise ValueError("selector type changed")
            value = os.readlink(self.name, dir_fd=self.parent.fd)
        else:
            value, current = self.read_with_metadata()
            if (current.st_uid, current.st_gid, stat.S_IMODE(current.st_mode)) != (
                    self.metadata.st_uid, self.metadata.st_gid, stat.S_IMODE(self.metadata.st_mode)):
                raise ValueError("file authority metadata changed")
        if content and value != self.content:
            raise ValueError("file contents changed")
        return value

    def replace(self, content: bytes, *, restore=False):
        self.parent.verify()
        if not restore:
            self.verify(content=False)
        name = f".{self.name}.{uuid.uuid4().hex}.tmp"
        fd = -1
        try:
            fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                         stat.S_IMODE(self.metadata.st_mode), dir_fd=self.parent.fd)
            if os.geteuid() == 0:
                os.fchown(fd, self.metadata.st_uid, self.metadata.st_gid)
            os.fchmod(fd, stat.S_IMODE(self.metadata.st_mode))
            with os.fdopen(fd, "wb") as stream:
                fd = -1
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
                # This is the exclusive temporary file we created and wrote,
                # captured before rename; never infer its identity from a later
                # arbitrary occupant of the authoritative destination path.
                published_metadata = os.fstat(stream.fileno())
            with publication_signal_guard():
                try:
                    self.parent.verify()
                    os.replace(name, self.name, src_dir_fd=self.parent.fd, dst_dir_fd=self.parent.fd)
                    self._reconcile_publication(content, published_metadata)
                    os.fsync(self.parent.fd)
                    self.verify()
                except BaseException:
                    # replace can succeed before an exception is delivered (or
                    # directory fsync can fail). Advance only to our exact known
                    # inode, bytes and metadata so rollback can read that state.
                    try:
                        self._reconcile_publication(content, published_metadata)
                    except BaseException:
                        pass  # Unknown replacements remain rejected by the pin.
                    raise
        finally:
            if fd >= 0:
                os.close(fd)
            try:
                os.unlink(name, dir_fd=self.parent.fd)
            except FileNotFoundError:
                pass

    def _reconcile_publication(self, content: bytes, published_metadata):
        current_content, current = self.read_with_metadata()
        if (identity(current) != identity(published_metadata) or current_content != content
                or (current.st_uid, current.st_gid, stat.S_IMODE(current.st_mode)) != (
                    published_metadata.st_uid, published_metadata.st_gid,
                    stat.S_IMODE(published_metadata.st_mode))):
            raise ValueError("destination is not the exact known publication")
        self.content = content
        self.expected = identity(published_metadata)

    def replace_link(self, target: str | Path, *, restore=False):
        self.parent.verify()
        if not restore:
            self.verify()
        name = f".{self.name}.{uuid.uuid4().hex}.tmp"
        try:
            os.symlink(str(target), name, dir_fd=self.parent.fd)
            self.parent.verify()
            os.replace(name, self.name, src_dir_fd=self.parent.fd, dst_dir_fd=self.parent.fd)
            os.fsync(self.parent.fd)
            self.content = str(target)
            self.expected = identity(self._stat())
            self.verify()
        finally:
            try:
                os.unlink(name, dir_fd=self.parent.fd)
            except FileNotFoundError:
                pass

    def close(self):
        if self.owns_parent:
            self.parent.close()


@contextmanager
def pinned_lock(parent: PinnedDirectory, name: str, *, owner: int):
    parent.verify()
    fd = os.open(name, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600, dir_fd=parent.fd)
    try:
        details = os.fstat(fd)
        regular(details)
        if details.st_uid != owner or stat.S_IMODE(details.st_mode) & 0o022:
            raise ValueError("lock must have its exact trusted owner and private writes")
        key = identity(details)
        def verify():
            parent.verify()
            current = os.stat(name, dir_fd=parent.fd, follow_symlinks=False)
            regular(current)
            if identity(current) != key or current.st_uid != owner or stat.S_IMODE(current.st_mode) & 0o022:
                raise ValueError("lock authority changed")
        fcntl.flock(fd, fcntl.LOCK_EX)
        verify()
        yield verify
        verify()
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


class ReloadTransactionLock:
    """Private root-owned lock; callers cannot pick its path through the CLI."""
    def __init__(self, path: Path, *, owner: int = 0):
        self.path = path
        self.owner = owner

    def __enter__(self):
        self.parent = PinnedDirectory(self.path.parent, trusted_owner=self.owner)
        try:
            self.lock = pinned_lock(self.parent, self.path.name, owner=self.owner)
            self.lock.__enter__()
        except BaseException:
            self.parent.close()
            raise
        return self

    def __exit__(self, *args):
        try:
            return self.lock.__exit__(*args)
        finally:
            self.parent.close()


class ReloadStateStore(StateStore):
    """Same state and lock names as StateStore, with no-follow pinned authority."""
    def __init__(self, path: Path):
        super().__init__(path)
        self.file = PinnedFile(self.path)
        self.verify_lock = None
        if stat.S_IMODE(self.file.metadata.st_mode) != 0o600:
            self.file.close()
            raise ValueError("operator state must have private 0600 mode")

    def load(self):
        if self.verify_lock is not None:
            self.verify_lock()
        content = self.file.verify()
        value = json.loads(content)
        if not isinstance(value, dict):
            raise ValueError("operator state must be an object")
        return value

    def save(self, value):
        if self.verify_lock is None:
            raise ValueError("state publication requires its authoritative lock")
        self.verify_lock()
        self.file.replace((json.dumps(value, separators=(",", ":")) + "\n").encode())
        self.verify_lock()

    @contextmanager
    def lock(self):
        name = self.path.name + ".lock"
        # Existing locks are service-owned. If absent, create via O_EXCL and
        # initialize ownership on the descriptor before any other user opens it.
        try:
            fd = os.open(name, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=self.file.parent.fd)
        except FileExistsError:
            pass
        else:
            try:
                if os.geteuid() == 0:
                    os.fchown(fd, self.file.metadata.st_uid, self.file.metadata.st_gid)
            finally:
                os.close(fd)
        with pinned_lock(self.file.parent, name, owner=self.file.metadata.st_uid) as verify:
            self.verify_lock = verify
            try:
                # An earlier serialized lifecycle may have atomically replaced state
                # while we waited. Re-pin the state only after obtaining the same lock.
                content, metadata = self.file.read_with_metadata()
                if (metadata.st_uid, metadata.st_gid, stat.S_IMODE(metadata.st_mode)) != (
                        self.file.metadata.st_uid, self.file.metadata.st_gid, 0o600):
                    raise ValueError("state ownership changed while acquiring lock")
                self.file.content = content
                self.file.expected = identity(metadata)
                yield
            finally:
                self.verify_lock = None

    def close(self):
        self.file.close()
