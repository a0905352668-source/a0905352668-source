"""Sanitized system-health projection for the operator dashboard."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any


DEFAULT_STATE_PATH = Path(
    "/media/boshi/Data/JianKong/02_configs/runtime/live_operator_state.json"
)
DEFAULT_HEALTH_PATH = Path(
    "/media/boshi/Data/JianKong/02_configs/runtime/live_operator_health.json"
)
STATUS_STALE_AFTER_SECONDS = 5.0
RECENT_INCIDENT_SECONDS = 30.0

_SAFE_IDENTIFIER = re.compile(r"[A-Za-z0-9_.-]{1,128}")
_SAFE_DISPLAY_NAME = re.compile(r"[\w .()（）·-]{1,128}")
_SAFE_CODE = re.compile(r"[a-z0-9][a-z0-9_.:-]{0,127}")
_STATUS_FIELDS = ("run_id", "state", "aggregate_fps", "updated_at")
_CAMERA_FIELDS = (
    "relay",
    "view",
    "status",
    "fps",
    "source_errors",
    "p95_latency_ms",
    "p95_latency",
)
_HEALTH_STATES = {
    "healthy",
    "recovering",
    "degraded",
    "blocked",
    "offline",
    "unknown",
}
_INTERNAL_OBSERVATION_CODES = {"unknown_process_observing"}
_SENSITIVE_TEXT = re.compile(
    r"(?:rtsp://|(?:^|[\s=:])(?:owner_?token|start_?token|password|pid|pgid)"
    r"|(?:^|[\s=:])(?:/[^ ]+|[A-Za-z]:\\))",
    re.IGNORECASE,
)


def build_public_health(
    raw_status: object,
    lifecycle_state: object,
    watchdog_health: object,
    index_summary: object,
    now: float,
) -> dict[str, Any]:
    """Return the complete, strictly whitelisted public status payload."""

    status = raw_status if isinstance(raw_status, Mapping) else {}
    lifecycle = lifecycle_state if isinstance(lifecycle_state, Mapping) else {}
    watchdog = watchdog_health if isinstance(watchdog_health, Mapping) else {}

    public_status = _public_status(status)
    target_fps = _positive_number(lifecycle.get("target_fps_per_stream"), 0)
    history_index = _public_index_summary(index_summary)
    status_age = _status_age(public_status.get("updated_at"), now)
    if status_age is not None and status_age > STATUS_STALE_AFTER_SECONDS:
        public_status = _stale_public_status(public_status)
    health = _public_watchdog_health(watchdog, status_age)
    alerts = _alerts(public_status, health, history_index, now)

    result = {
        **public_status,
        "health": health,
        "alerts": alerts,
        "target_fps_per_stream": target_fps,
        "history_index": history_index,
    }
    storage = status.get("storage")
    if isinstance(storage, Mapping):
        state = storage.get("state")
        result["storage"] = {
            "state": state if state in {"legacy", "healthy", "low_space", "unavailable"} else "unavailable",
            "hot": storage.get("hot") is True,
        }
        for key in ("free_bytes", "reserve_bytes", "metadata_bytes"):
            value = storage.get(key)
            if type(value) is int and 0 <= value <= 2**63 - 1:
                result["storage"][key] = value
        if type(storage.get("metadata_usage_complete")) is bool:
            result["storage"]["metadata_usage_complete"] = storage["metadata_usage_complete"]
    return result


def _public_status(status: Mapping[str, Any]) -> dict[str, Any]:
    run_id = _identifier(status.get("run_id"), "")
    state = _identifier(status.get("state"), "unknown")
    aggregate_fps = _nonnegative_number(status.get("aggregate_fps"), 0.0)
    updated_at = _finite_number(status.get("updated_at"))
    cameras = status.get("cameras")
    return {
        "run_id": run_id,
        "state": state,
        "aggregate_fps": aggregate_fps,
        "updated_at": updated_at,
        "cameras": [
            _public_camera(camera)
            for camera in (cameras if isinstance(cameras, list) else [])
            if isinstance(camera, Mapping)
        ],
    }


def _public_camera(camera: Mapping[str, Any]) -> dict[str, Any]:
    public: dict[str, Any] = {}
    for field in _CAMERA_FIELDS:
        if field not in camera:
            continue
        value = camera[field]
        if field in {"relay", "status"}:
            public[field] = _identifier(value, "")
        elif field == "view":
            public[field] = _display_name(value, "")
        elif field == "source_errors":
            public[field] = int(_nonnegative_number(value, 0))
        else:
            number = _finite_number(value)
            if number is not None:
                public[field] = number
    return public


def _stale_public_status(status: dict[str, Any]) -> dict[str, Any]:
    """Prevent old inference measurements from looking live."""
    stale = dict(status)
    stale["state"] = "stale"
    stale["aggregate_fps"] = 0.0
    stale["cameras"] = [
        {
            **camera,
            "status": "unknown",
            "fps": 0.0,
        }
        for camera in status.get("cameras", [])
        if isinstance(camera, Mapping)
    ]
    return stale


def _public_watchdog_health(
    watchdog: Mapping[str, Any],
    status_age: float | None,
) -> dict[str, Any]:
    state_value = watchdog.get("state")
    state = state_value if state_value in _HEALTH_STATES else "unknown"
    checked_at = _finite_number(
        watchdog.get("checked_at", watchdog.get("updated_at"))
    )
    health: dict[str, Any] = {
        "state": state,
        "checked_at": checked_at,
        "reason_code": _code(watchdog.get("reason_code"), "unknown"),
        "message": _safe_message(watchdog.get("message")),
        "last_action": _code(watchdog.get("last_action"), "none"),
        "retry_after": _nonnegative_number(watchdog.get("retry_after"), 0),
        "status_age_seconds": status_age,
    }
    last_action_at = _finite_number(watchdog.get("last_action_at"))
    if last_action_at is not None:
        health["last_action_at"] = last_action_at
    incident = _public_incident(watchdog.get("last_incident"))
    if incident is not None:
        health["last_incident"] = incident
    return health


def _public_incident(value: object) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    observed_at = _finite_number(value.get("observed_at"))
    if observed_at is None:
        return None
    incident: dict[str, Any] = {
        "active": value.get("active") is True,
        "observed_at": observed_at,
        "reason_code": _code(value.get("reason_code"), "unknown"),
        "message": _safe_message(value.get("message")),
    }
    resolved_at = _finite_number(value.get("resolved_at"))
    if resolved_at is not None:
        incident["resolved_at"] = resolved_at
    return incident


def _public_index_summary(value: object) -> dict[str, Any]:
    summary = value if isinstance(value, Mapping) else {}
    return {
        "runs_indexed": int(_nonnegative_number(summary.get("runs_indexed"), 0)),
        "events_indexed": int(
            _nonnegative_number(summary.get("events_indexed"), 0)
        ),
        "last_refresh_ms": _nonnegative_number(
            summary.get("last_refresh_ms"), 0.0
        ),
        "changed_runs": int(_nonnegative_number(summary.get("changed_runs"), 0)),
        "fallback_active": summary.get("fallback_active") is True,
        "last_error": _code(summary.get("last_error"), ""),
    }


def _alerts(
    status: Mapping[str, Any],
    health: Mapping[str, Any],
    history_index: Mapping[str, Any],
    now: float,
) -> list[dict[str, Any]]:
    alerts: list[dict[str, Any]] = []
    age = health.get("status_age_seconds")
    status_is_stale = (
        isinstance(age, (int, float)) and age > STATUS_STALE_AFTER_SECONDS
    )
    if status_is_stale:
        alerts.append(
            {
                "code": "status_stale",
                "severity": "critical",
                "message": "运行状态已停止更新",
            }
        )

    cameras = status.get("cameras")
    for camera in cameras if isinstance(cameras, list) and not status_is_stale else []:
        if not isinstance(camera, Mapping):
            continue
        camera_id = _identifier(
            camera.get("relay") or camera.get("view"),
            "camera",
        )
        if camera.get("status") not in {"online", "degraded"}:
            alerts.append(
                {
                    "code": "camera_offline",
                    "severity": "critical",
                    "message": f"{camera_id} 离线",
                    "camera": camera_id,
                }
            )
        if _nonnegative_number(camera.get("source_errors"), 0) > 0:
            alerts.append(
                {
                    "code": "camera_source_error",
                    "severity": "critical",
                    "message": f"{camera_id} 拉流异常",
                    "camera": camera_id,
                }
            )

    watchdog_state = health.get("state")
    reason_code = _code(health.get("reason_code"), "unknown")
    if (
        watchdog_state == "blocked"
        and reason_code not in _INTERNAL_OBSERVATION_CODES
    ):
        incident = health.get("last_incident")
        observed = incident.get("observed_at") if isinstance(incident, Mapping) else None
        brief_unknown = (
            reason_code == "worker_unknown"
            and status.get("state") == "running"
            and age is not None and not status_is_stale
            and isinstance(incident, Mapping) and incident.get("active") is True
            and incident.get("reason_code") == reason_code
            and isinstance(observed, (int, float))
            and _finite_number(now) is not None
            and 0 <= now - observed < 60
        )
        alerts.append(
            {
                "code": "watchdog_blocked",
                "severity": "warning" if brief_unknown else "critical",
                "message": ("正在确认事件处理状态，实时检测仍在运行"
                            if brief_unknown else f"自动恢复已阻止：{reason_code}"),
            }
        )
    elif watchdog_state == "degraded":
        alerts.append(
            {
                "code": "watchdog_degraded",
                "severity": "warning",
                "message": f"守护状态降级：{reason_code}",
            }
        )

    incident = health.get("last_incident")
    if (
        watchdog_state == "healthy"
        and isinstance(incident, Mapping)
        and incident.get("active") is False
        and _finite_number(now) is not None
        and isinstance(incident.get("resolved_at"), (int, float))
        and _code(incident.get("reason_code"), "unknown")
        not in _INTERNAL_OBSERVATION_CODES
        and 0.0 <= float(now) - float(incident["resolved_at"])
        <= RECENT_INCIDENT_SECONDS
    ):
        alerts.append(
            {
                "code": "watchdog_recovered",
                "severity": "info",
                "message": (
                    "守护异常已恢复："
                    f"{_code(incident.get('reason_code'), 'unknown')}"
                ),
            }
        )

    if history_index.get("fallback_active") is True:
        alerts.append(
            {
                "code": "history_index_degraded",
                "severity": "warning",
                "message": "历史事件索引降级，当前使用文件扫描",
            }
        )
    return sorted(
        alerts,
        key=lambda alert: 0 if alert["severity"] == "critical" else 1,
    )


def _status_age(updated_at: object, now: float) -> float | None:
    updated = _finite_number(updated_at)
    current = _finite_number(now)
    if updated is None or current is None:
        return None
    return max(0.0, current - updated)


def _safe_message(value: object) -> str:
    if not isinstance(value, str) or not value or len(value) > 500:
        return ""
    if _SENSITIVE_TEXT.search(value):
        return "Health details withheld."
    return value


def _identifier(value: object, default: str) -> str:
    return value if isinstance(value, str) and _SAFE_IDENTIFIER.fullmatch(value) else default


def _display_name(value: object, default: str) -> str:
    return value if isinstance(value, str) and _SAFE_DISPLAY_NAME.fullmatch(value) else default


def _code(value: object, default: str) -> str:
    return value if isinstance(value, str) and _SAFE_CODE.fullmatch(value) else default


def _finite_number(value: object) -> int | float | None:
    if (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    ):
        return value
    return None


def _nonnegative_number(value: object, default: int | float) -> int | float:
    number = _finite_number(value)
    return number if number is not None and number >= 0 else default


def _positive_number(value: object, default: int | float) -> int | float:
    number = _finite_number(value)
    return number if number is not None and number > 0 else default
