from __future__ import annotations

import pytest

from live_operator.capture_policy import (
    FinalOutcome,
    StageTwoLabel,
    aggregate_capture_decision,
)
from live_operator.capture_visibility import VisibilityRelation


def relation(state: str) -> VisibilityRelation:
    return VisibilityRelation(
        screen_id="screen_01",
        state=state,
        zone_id="zone_01" if state != "unknown" else None,
        occluder_id="partition_a" if state == "blocked" else None,
    )


@pytest.mark.parametrize(
    ("first_stage", "visibility", "stage_two", "expected"),
    [
        ("filter", "possible", "CAPTURE_POSSIBLE", "FILTER_NOT_PHONE"),
        ("pass", "blocked", "CAPTURE_POSSIBLE", "FILTER_CAPTURE_IMPOSSIBLE"),
        ("uncertain", "blocked", "UNCERTAIN", "FILTER_CAPTURE_IMPOSSIBLE"),
        ("pass", "possible", "CAPTURE_POSSIBLE", "KEEP_CAPTURE_POSSIBLE"),
        ("pass", "possible", "UNCERTAIN", "KEEP_SUSPECTED_CAPTURE"),
        ("uncertain", "possible", "CAPTURE_POSSIBLE", "KEEP_SUSPECTED_CAPTURE"),
        (
            "uncertain",
            "possible",
            "IMPOSSIBLE_FLAT_OR_DOWN",
            "KEEP_SUSPECTED_CAPTURE",
        ),
        (
            "pass",
            "possible",
            "IMPOSSIBLE_FLAT_OR_DOWN",
            "FILTER_CAPTURE_IMPOSSIBLE",
        ),
        (
            "pass",
            "possible",
            "IMPOSSIBLE_AWAY_FROM_SCREEN",
            "FILTER_CAPTURE_IMPOSSIBLE",
        ),
        (
            "pass",
            "possible",
            "NOT_PHONE_OR_NO_CAPTURE_ACTION",
            "FILTER_CAPTURE_IMPOSSIBLE",
        ),
        (
            "pass",
            "possible",
            "IMPOSSIBLE_BLOCKED",
            "KEEP_SUSPECTED_CAPTURE",
        ),
        (
            "pass",
            "unknown",
            "IMPOSSIBLE_AWAY_FROM_SCREEN",
            "KEEP_SUSPECTED_CAPTURE",
        ),
    ],
)
def test_conservative_decision_table(
    first_stage: str,
    visibility: str,
    stage_two: str,
    expected: str,
) -> None:
    decision = aggregate_capture_decision(
        first_stage,
        relation(visibility),
        StageTwoLabel(stage_two),
    )

    assert decision.outcome is FinalOutcome(expected)
    assert decision.reason


def test_invalid_first_stage_result_is_rejected() -> None:
    with pytest.raises(ValueError, match="invalid first-stage result"):
        aggregate_capture_decision(
            "success",
            relation("possible"),
            StageTwoLabel.CAPTURE_POSSIBLE,
        )
