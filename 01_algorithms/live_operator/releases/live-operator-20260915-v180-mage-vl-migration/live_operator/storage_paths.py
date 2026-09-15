"""Frozen per-run metadata placement; media always stays in the original run.

The administrator provisions the hot root. Never repair marked storage: a
missing mount or damaged identity must stop writers rather than split a run.
Initialization is serialized by a run-local advisory lock. Root directories
must be administrator controlled; symlinks are rejected, not followed.
"""
from dataclasses import dataclass
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile


DEFAULT_HOT_ROOT = Path('/var/lib/jiankong/hot-metadata')
_RUN_ID = re.compile(r'live_[0-9]{8}_[0-9]{6}\Z')
# Verified administrator migration alias, not a caller-configurable bypass.
_TRUSTED_PROJECT_ALIASES = {
    Path('/media/boshi/Data/JianKong'):
        Path('/media/boshi/Data/00_active_projects/JianKong'),
}


@dataclass(frozen=True)
class RunStorage:
    run_dir: Path
    metadata_dir: Path
    clips_dir: Path
    inference_dir: Path
    index_path: Path
    hot: bool


def _safe_path(path):
    path = Path(path)
    if '..' in path.parts:
        raise ValueError('parent traversal is not allowed in storage paths')
    path = path.absolute()
    for component in (*reversed(path.parents), path):
        if component.is_symlink():
            raise ValueError(f'symlink in storage path: {component}')
    return path


def _root(hot_root):
    root = _safe_path(hot_root if hot_root is not None else
                      os.environ.get('JIANKONG_HOT_METADATA_ROOT', DEFAULT_HOT_ROOT))
    if root == Path(root.anchor) or not root.is_dir():
        raise ValueError('hot root must be a provisioned, non-root directory')
    if root.stat().st_mode & 0o022:
        raise ValueError('hot root must not be group/world writable')
    return root


def _run_path(run_dir):
    path = Path(run_dir)
    if '..' in path.parts:
        raise ValueError('parent traversal is not allowed in storage paths')
    path = path.absolute()
    for alias, target in _TRUSTED_PROJECT_ALIASES.items():
        if path.is_relative_to(alias) and alias.is_symlink():
            _safe_path(alias.parent)
            _safe_path(target)
            link_target = alias.readlink()
            if '..' in link_target.parts:
                raise ValueError(f'parent traversal in trusted project alias: {alias}')
            if not link_target.is_absolute():
                link_target = alias.parent / link_target
            if link_target != target:
                raise ValueError(f'trusted project alias target mismatch: {alias}')
            path = target / path.relative_to(alias)
            break
    return _safe_path(path)


def _read_json(path):
    _safe_path(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd) as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError(f'not a regular storage identity: {path}')
        try:
            return json.load(stream)
        except (json.JSONDecodeError, UnicodeError) as exc:
            raise ValueError(f'invalid storage identity: {path}') from exc


def _sync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _publish(path, value):
    fd, temporary = tempfile.mkstemp(prefix=f'.{path.name}.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, sort_keys=True)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _sync_dir(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _identity(run, metadata):
    return dict(version=1, run_id=run.name, run_dir=str(run), metadata_dir=str(metadata))


def _paths(run, root=None):
    # The one project-prefix exception must never exempt run children.
    _safe_path(run / 'dashboard/clips')
    _safe_path(run / 'inference')
    return RunStorage(run, root / 'runs' / run.name / 'dashboard' if root else run / 'dashboard',
                      run / 'dashboard' / 'clips', run / 'inference',
                      root / 'index/event_index.sqlite3' if root else
                      run.parent / '.jiankong_dashboard/event_index.sqlite3', root is not None)


def _validate_hot(paths):
    for directory in (paths.metadata_dir.parent.parent, paths.metadata_dir.parent,
                      paths.metadata_dir, paths.index_path.parent):
        _safe_path(directory)
        if not directory.is_dir():
            raise ValueError(f'missing committed hot storage: {directory}')
        if directory.stat().st_mode & 0o022:
            raise ValueError(f'hot storage must not be group/world writable: {directory}')
    _safe_path(paths.index_path)
    ready = _read_json(paths.metadata_dir.parent / '.storage_ready.json')
    if ready != _identity(paths.run_dir, paths.metadata_dir) or type(ready.get('version')) is not int:
        raise ValueError('hot storage readiness identity mismatch')


def resolve_run_storage(run_dir: str | Path, *, hot_root: str | Path | None = None) -> RunStorage:
    run = _run_path(run_dir)
    marker = _safe_path(run / 'storage_layout.json')
    if not marker.exists():
        return _paths(run)
    value = _read_json(marker)
    if (not isinstance(value, dict) or set(value) != {'version', 'run_id'} or
            type(value['version']) is not int or value['version'] != 1 or
            value['run_id'] != run.name or not _RUN_ID.fullmatch(run.name)):
        raise ValueError('invalid storage layout marker')
    paths = _paths(run, _root(hot_root))
    _validate_hot(paths)
    return paths


def _mkdir(path):
    _safe_path(path)
    try:
        path.mkdir(mode=0o700)
    except FileExistsError:
        if not path.is_dir():
            raise ValueError(f'not a storage directory: {path}')
    if path.stat().st_mode & 0o022:
        raise ValueError(f'hot storage must not be group/world writable: {path}')
    _sync_dir(path)
    _sync_dir(path.parent)


def initialize_run_storage(run_dir: str | Path, *, hot_root: str | Path | None = None,
                           min_free_bytes: int = 20 * 1024**3) -> RunStorage:
    run = _run_path(run_dir)
    if not run.is_dir() or not _RUN_ID.fullmatch(run.name):
        raise ValueError('hot storage requires an existing live_YYYYMMDD_HHMMSS run')
    if min_free_bytes < 0:
        raise ValueError('min_free_bytes must be nonnegative')
    lock = _safe_path(run / '.storage_layout.lock')
    fd = os.open(lock, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    root_fd = None
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        current = resolve_run_storage(run, hot_root=hot_root)
        if current.hot:
            return current
        root = _root(hot_root)
        if root == run or run in root.parents:
            raise ValueError('hot root must be outside the run directory')
        # Also serialize distinct source paths with colliding run IDs. A
        # run-local lock alone cannot protect their shared readiness identity.
        root_lock = _safe_path(root / '.storage_init.lock')
        root_fd = os.open(root_lock, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        fcntl.flock(root_fd, fcntl.LOCK_EX)
        dashboard = _safe_path(run / 'dashboard')
        if dashboard.exists():
            if not dashboard.is_dir() or any(p.name != 'clips' for p in dashboard.iterdir()):
                raise ValueError('nonempty legacy metadata cannot migrate')
            _safe_path(dashboard / 'clips')
        if shutil.disk_usage(root).free < min_free_bytes:
            raise OSError('insufficient free space for hot metadata')
        paths = _paths(run, root)
        # Preflight every destination before creating anything beneath the root.
        for path in (paths.metadata_dir, paths.index_path, paths.metadata_dir.parent / '.storage_ready.json'):
            _safe_path(path)
        for directory in (root / 'runs', paths.metadata_dir.parent, paths.metadata_dir, root / 'index'):
            _mkdir(directory)
        ready = paths.metadata_dir.parent / '.storage_ready.json'
        if ready.exists():
            _validate_hot(paths)
        else:
            if any(paths.metadata_dir.iterdir()):
                raise ValueError('unidentified nonempty hot metadata cannot be adopted')
            _publish(ready, _identity(run, paths.metadata_dir))
        _validate_hot(paths)
        _publish(run / 'storage_layout.json', dict(version=1, run_id=run.name))
        return paths
    finally:
        if root_fd is not None:
            os.close(root_fd)
        os.close(fd)
