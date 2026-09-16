from __future__ import annotations

import pytest

from live_operator.inference_priority import ProductionFirstGate


class FakeClock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_offline_admission_requires_a_complete_quiet_period() -> None:
    clock = FakeClock()
    gate = ProductionFirstGate(clock, quiet_seconds=30.0)

    assert gate.try_acquire("offline") is None
    clock.advance(29.999)
    assert gate.try_acquire("offline") is None
    clock.advance(0.001)

    lease = gate.try_acquire("offline")
    assert lease is not None
    lease.release(latency_seconds=0.25)


def test_production_arrival_cancels_active_offline_before_retry() -> None:
    clock = FakeClock()
    gate = ProductionFirstGate(clock, quiet_seconds=30.0)
    clock.advance(30.0)
    offline = gate.try_acquire("offline")
    assert offline is not None

    gate.production_arrived()

    assert offline.cancelled.is_set()
    assert gate.try_acquire("production") is None
    offline.release(latency_seconds=0.5)
    production = gate.try_acquire("production")
    assert production is not None
    production.release(latency_seconds=0.1)


def test_validation_mode_starts_immediately_and_finishes_current_offline() -> None:
    clock = FakeClock()
    gate = ProductionFirstGate(
        clock,
        quiet_seconds=0.0,
        cancel_offline_on_production=False,
    )
    offline = gate.try_acquire("offline")
    assert offline is not None

    gate.production_arrived()

    assert not offline.cancelled.is_set()
    assert gate.try_acquire("production") is None
    assert gate.snapshot()["production_waiting"] == 1
    offline.release(latency_seconds=2.0)
    production = gate.try_acquire("production")
    assert production is not None
    production.release(latency_seconds=0.5)


def test_validation_reserves_next_slot_then_returns_turn_to_production() -> None:
    clock = FakeClock()
    gate = ProductionFirstGate(
        clock,
        quiet_seconds=0.0,
        cancel_offline_on_production=False,
        reserve_offline_turn=True,
    )
    gate.production_arrived()
    active_production = gate.try_acquire("production")
    assert active_production is not None

    gate.offline_arrived()
    gate.offline_arrived()
    assert gate.try_acquire("offline") is None
    assert gate.snapshot()["offline_reserved"] is True

    active_production.release(latency_seconds=1.0)
    gate.production_arrived()
    assert gate.try_acquire("production") is None

    offline = gate.try_acquire("offline")
    assert offline is not None
    assert gate.snapshot()["offline_reserved"] is False
    gate.production_arrived()
    offline.release(latency_seconds=2.0)

    gate.offline_arrived()
    assert gate.try_acquire("offline") is None
    production = gate.try_acquire("production")
    assert production is not None
    production.release(latency_seconds=0.5)


def test_abandoned_offline_reservation_expires() -> None:
    clock = FakeClock()
    gate = ProductionFirstGate(
        clock,
        quiet_seconds=0.0,
        cancel_offline_on_production=False,
        reserve_offline_turn=True,
        offline_reservation_seconds=5.0,
    )
    gate.production_arrived()
    active = gate.try_acquire("production")
    assert active is not None
    gate.offline_arrived()
    active.release(latency_seconds=1.0)

    clock.advance(5.001)
    gate.production_arrived()
    production = gate.try_acquire("production")

    assert production is not None
    assert gate.snapshot()["offline_reserved"] is False
    production.release(latency_seconds=0.5)


def test_production_can_start_without_waiting_for_quiet_time() -> None:
    clock = FakeClock()
    gate = ProductionFirstGate(clock, quiet_seconds=30.0)

    gate.production_arrived()
    assert gate.snapshot()["production_waiting"] == 1
    lease = gate.try_acquire("production")

    assert lease is not None
    assert gate.snapshot()["production_waiting"] == 0
    lease.release(latency_seconds=1.25)


def test_snapshot_reports_active_kind_counts_latency_and_errors() -> None:
    clock = FakeClock()
    gate = ProductionFirstGate(clock, quiet_seconds=30.0)
    gate.production_arrived()
    production = gate.try_acquire("production")
    assert production is not None
    assert gate.snapshot()["active_kind"] == "production"
    production.release(latency_seconds=1.5, error=True)

    clock.advance(30.0)
    offline = gate.try_acquire("offline")
    assert offline is not None
    offline.release(latency_seconds=2.0)

    snapshot = gate.snapshot()
    assert snapshot["active_kind"] is None
    assert snapshot["production_admitted"] == 1
    assert snapshot["offline_admitted"] == 1
    assert snapshot["production_errors"] == 1
    assert snapshot["offline_errors"] == 0
    assert snapshot["production_latency_seconds"] == 1.5
    assert snapshot["offline_latency_seconds"] == 2.0


def test_offline_rejection_is_counted_and_double_release_is_rejected() -> None:
    clock = FakeClock()
    gate = ProductionFirstGate(clock, quiet_seconds=30.0)
    assert gate.try_acquire("offline") is None
    assert gate.snapshot()["offline_rejected"] == 1

    production = gate.try_acquire("production")
    assert production is not None
    production.release(latency_seconds=0.1)
    with pytest.raises(RuntimeError, match="already released"):
        production.release(latency_seconds=0.1)


def test_invalid_work_kind_is_rejected() -> None:
    gate = ProductionFirstGate(FakeClock())

    with pytest.raises(ValueError, match="invalid inference kind"):
        gate.try_acquire("batch")  # type: ignore[arg-type]
