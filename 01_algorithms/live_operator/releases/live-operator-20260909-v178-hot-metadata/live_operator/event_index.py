"""Persistent, incremental index for events produced by inference runs."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


SCHEMA_VERSION = 1
_LOADER_FAILED = "loader_failed"
_INDEX_UNAVAILABLE = "index_unavailable"


class EventIndexError(RuntimeError):
    """A sanitized failure that allows callers to use their fallback path."""


class _CorruptIndex(Exception):
    pass


class _ArchiveError(OSError):
    pass


@dataclass(frozen=True)
class RunSignature:
    events_mtime_ns: int
    events_size: int
    reviews_mtime_ns: int
    reviews_size: int
    clips_mtime_ns: int


@dataclass(frozen=True)
class IndexedRecord:
    source_run_id: str
    event_id: str
    occurred_at: datetime
    payload: dict[str, Any]


@dataclass(frozen=True)
class IndexSummary:
    loaded_runs: int
    skipped_runs: int
    removed_runs: int
    total_records: int
    fallback_active: bool = False
    last_error: str | None = None


@dataclass(frozen=True)
class _Candidate:
    source_run_id: str
    run_path: str
    signature: RunSignature
    original: Any


class EventIndex:
    def __init__(
        self,
        database_path: Path,
        *,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._database_path = Path(database_path)
        self._clock = clock
        self._lock = threading.RLock()
        self._database_path.parent.mkdir(parents=True, exist_ok=True)
        self.summary = IndexSummary(0, 0, 0, 0)
        with self._lock:
            self._initialize_database(allow_recovery=True)
            self.summary = IndexSummary(0, 0, 0, self._record_count())

    def refresh(
        self,
        candidates: Iterable[Any],
        loader: Callable[[Any], Iterable[IndexedRecord]],
    ) -> IndexSummary:
        with self._lock:
            return self._refresh(candidates, loader)

    def _refresh(
        self,
        candidates: Iterable[Any],
        loader: Callable[[Any], Iterable[IndexedRecord]],
    ) -> IndexSummary:
        normalized = [_normalize_candidate(candidate) for candidate in candidates]
        by_run_id = {candidate.source_run_id: candidate for candidate in normalized}
        if len(by_run_id) != len(normalized):
            raise ValueError("candidate source_run_id values must be unique")

        loaded_runs = 0
        skipped_runs = 0
        try:
            with self._connection() as connection:
                existing = {
                    row["source_run_id"]: RunSignature(
                        row["events_mtime_ns"],
                        row["events_size"],
                        row["reviews_mtime_ns"],
                        row["reviews_size"],
                        row["clips_mtime_ns"],
                    )
                    for row in connection.execute(
                        """
                        SELECT source_run_id, events_mtime_ns, events_size,
                               reviews_mtime_ns, reviews_size, clips_mtime_ns
                        FROM runs
                        """
                    )
                }

                for candidate in normalized:
                    if existing.get(candidate.source_run_id) == candidate.signature:
                        skipped_runs += 1
                        continue
                    try:
                        records = list(loader(candidate.original))
                        _validate_records(candidate.source_run_id, records)
                        record_rows = [
                            (
                                item.source_run_id,
                                item.event_id,
                                item.occurred_at.isoformat(timespec="microseconds"),
                                json.dumps(
                                    item.payload,
                                    ensure_ascii=False,
                                    sort_keys=True,
                                    separators=(",", ":"),
                                ),
                            )
                            for item in records
                        ]
                    except Exception:
                        self._activate_fallback(_LOADER_FAILED)
                        raise EventIndexError("event loader failed") from None
                    with connection:
                        connection.execute(
                            "DELETE FROM runs WHERE source_run_id = ?",
                            (candidate.source_run_id,),
                        )
                        connection.execute(
                            """
                            INSERT INTO runs(
                              source_run_id, run_path, events_mtime_ns, events_size,
                              reviews_mtime_ns, reviews_size, clips_mtime_ns, indexed_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                candidate.source_run_id,
                                candidate.run_path,
                                candidate.signature.events_mtime_ns,
                                candidate.signature.events_size,
                                candidate.signature.reviews_mtime_ns,
                                candidate.signature.reviews_size,
                                candidate.signature.clips_mtime_ns,
                                self._clock(),
                            ),
                        )
                        connection.executemany(
                            """
                            INSERT INTO events(source_run_id, event_id, occurred_at, payload_json)
                            VALUES (?, ?, ?, ?)
                            """,
                            record_rows,
                        )
                    loaded_runs += 1

                missing_run_ids = sorted(set(existing) - set(by_run_id))
                if missing_run_ids:
                    with connection:
                        connection.executemany(
                            "DELETE FROM runs WHERE source_run_id = ?",
                            [(run_id,) for run_id in missing_run_ids],
                        )
                total_records = connection.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        except EventIndexError:
            raise
        except sqlite3.Error:
            self._activate_fallback(_INDEX_UNAVAILABLE)
            raise EventIndexError("event index unavailable") from None

        self.summary = IndexSummary(
            loaded_runs=loaded_runs,
            skipped_runs=skipped_runs,
            removed_runs=len(missing_run_ids),
            total_records=total_records,
            fallback_active=False,
            last_error=None,
        )
        return self.summary

    def records(self) -> list[IndexedRecord]:
        with self._lock:
            try:
                with self._connection(recover_corruption=False) as connection:
                    rows = connection.execute(
                        """
                        SELECT source_run_id, event_id, occurred_at, payload_json
                        FROM events
                        """
                    ).fetchall()
                records = [
                    IndexedRecord(
                        source_run_id=row["source_run_id"],
                        event_id=row["event_id"],
                        occurred_at=datetime.fromisoformat(row["occurred_at"]),
                        payload=json.loads(row["payload_json"]),
                    )
                    for row in rows
                ]
            except _CorruptIndex:
                self._activate_fallback(_INDEX_UNAVAILABLE, total_records=0)
                self._recover_database()
                raise EventIndexError("event index unavailable") from None
            except (sqlite3.Error, TypeError, ValueError):
                self._activate_fallback(_INDEX_UNAVAILABLE)
                raise EventIndexError("event index unavailable") from None
            return sorted(
                records,
                key=lambda item: (item.occurred_at, item.event_id, item.source_run_id),
            )

    def rebuild(self) -> None:
        with self._lock:
            try:
                self._archive_database()
                self._initialize_database(allow_recovery=False)
            except (EventIndexError, OSError, sqlite3.Error):
                self._activate_fallback(_INDEX_UNAVAILABLE, total_records=0)
                raise EventIndexError("event index unavailable") from None
            self.summary = IndexSummary(0, 0, 0, 0)

    def _record_count(self) -> int:
        try:
            with self._connection() as connection:
                return connection.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        except sqlite3.Error:
            self._activate_fallback(_INDEX_UNAVAILABLE)
            raise EventIndexError("event index unavailable") from None

    @contextmanager
    def _connection(
        self,
        *,
        recover_corruption: bool = True,
    ) -> Iterator[sqlite3.Connection]:
        try:
            with self._open_checked_connection(validate_integrity=False) as connection:
                yield connection
                return
        except _CorruptIndex:
            if not recover_corruption:
                raise
            self._recover_database()
        with self._open_checked_connection(validate_integrity=False) as connection:
            yield connection

    @contextmanager
    def _open_checked_connection(
        self,
        *,
        validate_integrity: bool,
    ) -> Iterator[sqlite3.Connection]:
        connection: sqlite3.Connection | None = None
        try:
            connection = sqlite3.connect(str(self._database_path))
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("PRAGMA busy_timeout=5000")
            if validate_integrity:
                check_rows = connection.execute("PRAGMA quick_check").fetchall()
                if not check_rows or any(row[0] != "ok" for row in check_rows):
                    raise _CorruptIndex
        except sqlite3.Error as error:
            if connection is not None:
                connection.close()
            if _is_corruption_error(error):
                raise _CorruptIndex from None
            raise
        except BaseException:
            if connection is not None:
                connection.close()
            raise
        try:
            yield connection
        finally:
            connection.close()

    def _initialize_database(self, *, allow_recovery: bool) -> None:
        try:
            with self._open_checked_connection(validate_integrity=True) as connection:
                tables = {
                    row["name"]
                    for row in connection.execute(
                        """
                        SELECT name
                        FROM sqlite_master
                        WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
                        """
                    )
                }
                if not tables:
                    _create_schema(connection)
                _validate_schema(connection)
                connection.commit()
        except _CorruptIndex:
            if not allow_recovery:
                self._activate_fallback(_INDEX_UNAVAILABLE)
                raise EventIndexError("event index unavailable") from None
            self._recover_database()
        except sqlite3.Error:
            self._activate_fallback(_INDEX_UNAVAILABLE)
            raise EventIndexError("event index unavailable") from None

    def _recover_database(self) -> None:
        try:
            self._archive_database()
            self._initialize_database(allow_recovery=False)
        except (EventIndexError, OSError, sqlite3.Error):
            self._activate_fallback(_INDEX_UNAVAILABLE, total_records=0)
            raise EventIndexError("event index unavailable") from None

    def _archive_database(self) -> None:
        sources = (
            self._database_path,
            Path(f"{self._database_path}-wal"),
            Path(f"{self._database_path}-shm"),
        )
        if not any(path.exists() for path in sources):
            return
        timestamp = datetime.fromtimestamp(self._clock(), timezone.utc).strftime("%Y%m%d%H%M%S")
        counter = 0
        while True:
            suffix = "" if counter == 0 else f"-{counter}"
            backup = Path(f"{self._database_path}.corrupt-{timestamp}{suffix}")
            destinations = (backup, Path(f"{backup}-wal"), Path(f"{backup}-shm"))
            if not any(path.exists() for path in destinations):
                break
            counter += 1
        planned_moves = [
            (source, destination)
            for source, destination in zip(sources, destinations)
            if source.exists()
        ]
        for source, destination in planned_moves:
            if source.parent != destination.parent or destination.exists():
                raise _ArchiveError

        moved: list[tuple[Path, Path]] = []
        try:
            for source, destination in planned_moves:
                if not source.exists() or destination.exists():
                    raise _ArchiveError
                source.rename(destination)
                moved.append((source, destination))
        except (OSError, sqlite3.Error):
            for source, destination in reversed(moved):
                try:
                    if source.exists() or not destination.exists():
                        break
                    destination.rename(source)
                except (OSError, sqlite3.Error):
                    break
            raise _ArchiveError from None

    def _activate_fallback(
        self,
        category: str,
        *,
        total_records: int | None = None,
    ) -> None:
        self.summary = IndexSummary(
            loaded_runs=0,
            skipped_runs=0,
            removed_runs=0,
            total_records=(
                self.summary.total_records if total_records is None else total_records
            ),
            fallback_active=True,
            last_error=category,
        )


def _normalize_candidate(candidate: Any) -> _Candidate:
    if isinstance(candidate, Mapping):
        source_run_id = candidate["source_run_id"]
        run_path = candidate["run_path"]
        signature = candidate["signature"]
    else:
        source_run_id = candidate.source_run_id
        run_path = candidate.run_path
        signature = candidate.signature
    if not isinstance(source_run_id, str):
        raise TypeError("candidate source_run_id must be a string")
    if not isinstance(signature, RunSignature):
        raise TypeError("candidate signature must be a RunSignature")
    return _Candidate(source_run_id, str(run_path), signature, candidate)


def _validate_records(source_run_id: str, records: Iterable[IndexedRecord]) -> None:
    for item in records:
        if not isinstance(item, IndexedRecord):
            raise TypeError("loader must return IndexedRecord instances")
        if item.source_run_id != source_run_id:
            raise ValueError("record source_run_id must match its candidate")
        if item.occurred_at.tzinfo is None or item.occurred_at.utcoffset() is None:
            raise ValueError("record occurred_at must be timezone-aware")


def _validate_schema(connection: sqlite3.Connection) -> None:
    expected_columns = {
        "meta": [
            ("key", "TEXT", 0, None, 1),
            ("value", "TEXT", 1, None, 0),
        ],
        "runs": [
            ("source_run_id", "TEXT", 0, None, 1),
            ("run_path", "TEXT", 1, None, 0),
            ("events_mtime_ns", "INTEGER", 1, None, 0),
            ("events_size", "INTEGER", 1, None, 0),
            ("reviews_mtime_ns", "INTEGER", 1, None, 0),
            ("reviews_size", "INTEGER", 1, None, 0),
            ("clips_mtime_ns", "INTEGER", 1, None, 0),
            ("indexed_at", "REAL", 1, None, 0),
        ],
        "events": [
            ("source_run_id", "TEXT", 1, None, 1),
            ("event_id", "TEXT", 1, None, 2),
            ("occurred_at", "TEXT", 1, None, 0),
            ("payload_json", "TEXT", 1, None, 0),
        ],
    }
    for table, expected in expected_columns.items():
        actual = [
            (
                row["name"],
                row["type"].upper(),
                row["notnull"],
                row["dflt_value"],
                row["pk"],
            )
            for row in connection.execute(f"PRAGMA table_info({table})")
        ]
        if actual != expected:
            raise _CorruptIndex

    foreign_keys = [
        (
            row["table"],
            row["from"],
            row["to"],
            row["on_update"],
            row["on_delete"],
            row["match"],
        )
        for row in connection.execute("PRAGMA foreign_key_list(events)")
    ]
    if foreign_keys != [
        ("runs", "source_run_id", "source_run_id", "NO ACTION", "CASCADE", "NONE")
    ]:
        raise _CorruptIndex
    if list(connection.execute("PRAGMA foreign_key_list(meta)")) or list(
        connection.execute("PRAGMA foreign_key_list(runs)")
    ):
        raise _CorruptIndex

    index_row = next(
        (
            row
            for row in connection.execute("PRAGMA index_list(events)")
            if row["name"] == "events_order"
        ),
        None,
    )
    if (
        index_row is None
        or index_row["unique"] != 0
        or index_row["origin"] != "c"
        or index_row["partial"] != 0
    ):
        raise _CorruptIndex
    index_columns = [
        row["name"]
        for row in connection.execute("PRAGMA index_info(events_order)")
    ]
    if index_columns != ["occurred_at", "event_id", "source_run_id"]:
        raise _CorruptIndex

    version_rows = connection.execute(
        "SELECT value FROM meta WHERE key = 'schema_version'"
    ).fetchall()
    if len(version_rows) != 1 or version_rows[0]["value"] != str(SCHEMA_VERSION):
        raise _CorruptIndex


def _create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE meta (
          key TEXT PRIMARY KEY,
          value TEXT NOT NULL
        );
        CREATE TABLE runs (
          source_run_id TEXT PRIMARY KEY,
          run_path TEXT NOT NULL,
          events_mtime_ns INTEGER NOT NULL,
          events_size INTEGER NOT NULL,
          reviews_mtime_ns INTEGER NOT NULL,
          reviews_size INTEGER NOT NULL,
          clips_mtime_ns INTEGER NOT NULL,
          indexed_at REAL NOT NULL
        );
        CREATE TABLE events (
          source_run_id TEXT NOT NULL,
          event_id TEXT NOT NULL,
          occurred_at TEXT NOT NULL,
          payload_json TEXT NOT NULL,
          PRIMARY KEY (source_run_id, event_id),
          FOREIGN KEY (source_run_id) REFERENCES runs(source_run_id) ON DELETE CASCADE
        );
        CREATE INDEX events_order
        ON events(occurred_at, event_id, source_run_id);
        """
    )
    connection.execute(
        "INSERT INTO meta(key, value) VALUES ('schema_version', ?)",
        (str(SCHEMA_VERSION),),
    )


def _is_corruption_error(error: sqlite3.Error) -> bool:
    corruption_codes = {
        getattr(sqlite3, "SQLITE_CORRUPT", 11),
        getattr(sqlite3, "SQLITE_NOTADB", 26),
    }
    if getattr(error, "sqlite_errorcode", None) in corruption_codes:
        return True
    message = str(error).lower()
    return any(
        marker in message
        for marker in (
            "database disk image is malformed",
            "file is not a database",
            "file is encrypted",
            "malformed database schema",
        )
    )
