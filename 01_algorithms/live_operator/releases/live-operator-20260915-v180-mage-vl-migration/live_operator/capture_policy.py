"""Conservative aggregation for dual-stage screen-capture review."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from live_operator.capture_visibility import VisibilityRelation


class StageTwoLabel(StrEnum):
    CAPTURE_POSSIBLE = "CAPTURE_POSSIBLE"
    IMPOSSIBLE_FLAT_OR_DOWN = "IMPOSSIBLE_FLAT_OR_DOWN"
    IMPOSSIBLE_AWAY_FROM_SCREEN = "IMPOSSIBLE_AWAY_FROM_SCREEN"
    IMPOSSIBLE_BLOCKED = "IMPOSSIBLE_BLOCKED"
    NOT_PHONE_OR_NO_CAPTURE_ACTION = "NOT_PHONE_OR_NO_CAPTURE_ACTION"
    UNCERTAIN = "UNCERTAIN"


class FinalOutcome(StrEnum):
    FILTER_NOT_PHONE = "FILTER_NOT_PHONE"
    FILTER_CAPTURE_IMPOSSIBLE = "FILTER_CAPTURE_IMPOSSIBLE"
    KEEP_CAPTURE_POSSIBLE = "KEEP_CAPTURE_POSSIBLE"
    KEEP_SUSPECTED_CAPTURE = "KEEP_SUSPECTED_CAPTURE"


@dataclass(frozen=True)
class FinalDecision:
    outcome: FinalOutcome
    reason: str


def aggregate_capture_decision(
    first_stage_result: str,
    visibility: VisibilityRelation,
    stage_two: StageTwoLabel,
) -> FinalDecision:
    if first_stage_result not in {"pass", "filter", "uncertain"}:
        raise ValueError("invalid first-stage result")
    if first_stage_result == "filter":
        return FinalDecision(FinalOutcome.FILTER_NOT_PHONE, "stage_one_filtered")
    if visibility.state == "blocked":
        return FinalDecision(
            FinalOutcome.FILTER_CAPTURE_IMPOSSIBLE,
            "verified_spatial_block",
        )
    if first_stage_result == "uncertain":
        return FinalDecision(
            FinalOutcome.KEEP_SUSPECTED_CAPTURE,
            "stage_one_uncertain",
        )
    if visibility.state == "unknown":
        return FinalDecision(
            FinalOutcome.KEEP_SUSPECTED_CAPTURE,
            "visibility_unknown",
        )
    if stage_two is StageTwoLabel.CAPTURE_POSSIBLE:
        return FinalDecision(
            FinalOutcome.KEEP_CAPTURE_POSSIBLE,
            "capture_pose_supported",
        )
    if stage_two in {
        StageTwoLabel.IMPOSSIBLE_FLAT_OR_DOWN,
        StageTwoLabel.IMPOSSIBLE_AWAY_FROM_SCREEN,
        StageTwoLabel.NOT_PHONE_OR_NO_CAPTURE_ACTION,
    }:
        return FinalDecision(
            FinalOutcome.FILTER_CAPTURE_IMPOSSIBLE,
            f"stage_two_{stage_two.value.lower()}",
        )
    return FinalDecision(
        FinalOutcome.KEEP_SUSPECTED_CAPTURE,
        "stage_two_uncertain_or_conflicting",
    )
