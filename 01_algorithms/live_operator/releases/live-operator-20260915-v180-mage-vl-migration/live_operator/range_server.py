"""HTTP byte-range response helpers for dashboard media files."""

from __future__ import annotations

import mimetypes
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Mapping


_RANGE_PATTERN = re.compile(r"bytes=(\d*)-(\d*)")


@dataclass(frozen=True)
class HTTPResponse:
    status: int
    headers: Mapping[str, str]
    body: bytes | Iterable[bytes]

    def iter_body(self) -> Iterator[bytes]:
        if isinstance(self.body, bytes):
            if self.body:
                yield self.body
            return
        yield from self.body


@dataclass(frozen=True)
class FileBody:
    path: Path
    start: int
    length: int
    chunk_size: int = 64 * 1024

    def __iter__(self) -> Iterator[bytes]:
        remaining = self.length
        with self.path.open("rb") as handle:
            handle.seek(self.start)
            while remaining > 0:
                chunk = handle.read(min(self.chunk_size, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
                yield chunk


def file_response(
    path: str | Path, *, range_header: str | None = None, head_only: bool = False
) -> HTTPResponse:
    """Return a complete or single-range response without starting a server."""

    source = Path(path)
    if not source.is_file():
        return HTTPResponse(404, {"Content-Length": "0"}, b"")

    size = source.stat().st_size
    content_type = mimetypes.guess_type(source.name)[0] or "application/octet-stream"
    if content_type in {"application/json", "text/javascript", "text/css", "text/html"}:
        content_type = f"{content_type}; charset=utf-8"
    base_headers = {
        "Accept-Ranges": "bytes",
        "Content-Type": content_type,
        "Cache-Control": "public, max-age=31536000, immutable",
    }
    if range_header is None:
        body: bytes | Iterable[bytes] = (
            b"" if head_only else FileBody(source, start=0, length=size)
        )
        return HTTPResponse(
            200, {**base_headers, "Content-Length": str(size)}, body
        )

    byte_range = _parse_range(range_header, size)
    if byte_range is None:
        return HTTPResponse(
            416,
            {
                **base_headers,
                "Content-Range": f"bytes */{size}",
                "Content-Length": "0",
            },
            b"",
        )
    start, end = byte_range
    length = end - start + 1
    body = b"" if head_only else FileBody(source, start=start, length=length)
    return HTTPResponse(
        206,
        {
            **base_headers,
            "Content-Range": f"bytes {start}-{end}/{size}",
            "Content-Length": str(length),
        },
        body,
    )


def _parse_range(value: str, size: int) -> tuple[int, int] | None:
    if size <= 0 or "," in value:
        return None
    match = _RANGE_PATTERN.fullmatch(value.strip())
    if match is None:
        return None
    raw_start, raw_end = match.groups()
    if not raw_start and not raw_end:
        return None
    if not raw_start:
        suffix = int(raw_end)
        if suffix <= 0:
            return None
        start = max(0, size - suffix)
        return start, size - 1
    start = int(raw_start)
    if start >= size:
        return None
    end = size - 1 if not raw_end else min(int(raw_end), size - 1)
    if end < start:
        return None
    return start, end
