"""Production-first admission for the single shared Mage-VL model."""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Literal


WorkKind = Literal["production", "offline"]


class GateLease:
    def __init__(self, gate: "ProductionFirstGate", kind: WorkKind) -> None:
        self._gate = gate
        self.kind = kind
        self.cancelled = threading.Event()
        self._released = False

    def release(self, latency_seconds: float, error: bool = False) -> None:
        self._gate._release(self, latency_seconds=latency_seconds, error=error)


class ProductionFirstGate:
    def __init__(
        self,
        clock: Callable[[], float],
        quiet_seconds: float = 30.0,
    ) -> None:
        self._clock = clock
        self.quiet_seconds = float(quiet_seconds)
        self._lock = threading.Lock()
        self._active: GateLease | None = None
        self._last_production_at = float(clock())
        self._production_waiting = 0
        self._admitted = {"production": 0, "offline": 0}
        self._rejected = {"production": 0, "offline": 0}
        self._errors = {"production": 0, "offline": 0}
        self._latency_seconds = {"production": 0.0, "offline": 0.0}
        self._last_latency_seconds: dict[WorkKind, float | None] = {
            "production": None,
            "offline": None,
        }

    def production_arrived(self) -> None:
        with self._lock:
            self._last_production_at = float(self._clock())
            self._production_waiting += 1
            if self._active is not None and self._active.kind == "offline":
                self._active.cancelled.set()

    def try_acquire(self, kind: WorkKind) -> GateLease | None:
        if kind not in ("production", "offline"):
            raise ValueError("invalid inference kind")
        with self._lock:
            now = float(self._clock())
            if kind == "production" and self._production_waiting:
                self._production_waiting -= 1
            if self._active is not None:
                self._rejected[kind] += 1
                return None
            if (
                kind == "offline"
                and now - self._last_production_at < self.quiet_seconds
            ):
                self._rejected[kind] += 1
                return None
            lease = GateLease(self, kind)
            self._active = lease
            self._admitted[kind] += 1
            return lease

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            quiet_remaining = max(
                0.0,
                self.quiet_seconds - (float(self._clock()) - self._last_production_at),
            )
            return {
                "active_kind": None if self._active is None else self._active.kind,
                "production_waiting": self._production_waiting,
                "offline_cancel_requested": bool(
                    self._active is not None
                    and self._active.kind == "offline"
                    and self._active.cancelled.is_set()
                ),
                "quiet_remaining_seconds": quiet_remaining,
                "production_admitted": self._admitted["production"],
                "offline_admitted": self._admitted["offline"],
                "production_rejected": self._rejected["production"],
                "offline_rejected": self._rejected["offline"],
                "production_errors": self._errors["production"],
                "offline_errors": self._errors["offline"],
                "production_latency_seconds": self._latency_seconds["production"],
                "offline_latency_seconds": self._latency_seconds["offline"],
                "production_last_latency_seconds": self._last_latency_seconds[
                    "production"
                ],
                "offline_last_latency_seconds": self._last_latency_seconds["offline"],
            }

    def _release(
        self,
        lease: GateLease,
        *,
        latency_seconds: float,
        error: bool,
    ) -> None:
        with self._lock:
            if lease._released:
                raise RuntimeError("inference lease already released")
            if self._active is not lease:
                raise RuntimeError("inference lease is not active")
            lease._released = True
            self._active = None
            self._latency_seconds[lease.kind] += float(latency_seconds)
            self._last_latency_seconds[lease.kind] = float(latency_seconds)
            if error:
                self._errors[lease.kind] += 1
