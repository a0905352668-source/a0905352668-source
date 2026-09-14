"""Read-only, bounded storage telemetry; never a runtime restart decision."""

import os
import shutil
import time

from live_operator.storage_paths import resolve_run_storage

RESERVE_BYTES = 20 * 1024**3


class StorageHealth:
    def __init__(self, run_dir):
        self.run_dir = run_dir
        self._storage = resolve_run_storage(run_dir)
        self._next_usage_at = 0.0
        self._usage = (0, False)

    def snapshot(self):
        try:
            storage = resolve_run_storage(self.run_dir)
            if storage != self._storage:
                raise ValueError("storage authority changed")
            if not storage.hot:
                return {"state": "legacy", "hot": False}
            free = shutil.disk_usage(storage.metadata_dir).free
            now = time.monotonic()
            if now >= self._next_usage_at:
                size, visited, complete = 0, 0, True
                def traversal_error(_error):
                    nonlocal complete
                    complete = False
                # Bound work and do not traverse symlink directories or count
                # symlink targets. Cache usage; free space remains a cheap probe.
                for directory, directories, files in os.walk(
                    storage.metadata_dir, followlinks=False, onerror=traversal_error
                ):
                    visited += 1 + len(directories) + len(files)
                    if visited > 10000:
                        complete = False
                        break
                    directories[:] = [d for d in directories if not os.path.islink(os.path.join(directory, d))]
                    for name in files:
                        path = os.path.join(directory, name)
                        if not os.path.islink(path):
                            size += os.stat(path, follow_symlinks=False).st_size
                self._usage = size, complete
                self._next_usage_at = now + 60.0
            return {"hot": True, "state": "low_space" if free < RESERVE_BYTES else "healthy",
                    "free_bytes": free, "reserve_bytes": RESERVE_BYTES,
                    "metadata_bytes": self._usage[0], "metadata_usage_complete": self._usage[1]}
        except (OSError, ValueError):
            return {"hot": True, "state": "unavailable"}
