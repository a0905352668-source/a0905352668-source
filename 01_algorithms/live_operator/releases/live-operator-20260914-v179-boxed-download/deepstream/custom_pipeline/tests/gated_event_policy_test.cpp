#include "gated_event_policy.hpp"

#include <cstdlib>
#include <iostream>

using namespace jiankong::custom_pipeline;

static void require(bool value, const char* message) {
    if (!value) {
        std::cerr << message << '\n';
        std::exit(1);
    }
}

static GatedFrameEvidence strong() {
    GatedFrameEvidence item;
    item.phone_valid = true;
    item.person_association = EvidenceState::true_value;
    item.hand_relation = EvidenceState::true_value;
    item.screen_intent = EvidenceState::true_value;
    item.high_confidence_phone = true;
    item.screen_ray_hit = true;
    item.static_state = StaticEvidenceState::moving_or_handheld;
    return item;
}

int main() {
    GatedEventPolicy main_path;
    GatedEventDecision decision;
    for (int i = 0; i < 9; ++i) {
        decision = main_path.update(strong());
        require(!decision.alarm, "main path must not alarm before ten processed frames");
    }
    decision = main_path.update(strong());
    require(decision.alarm, "main path must alarm after stable core evidence");
    require(decision.state == GatedEventState::alarm, "alarm state");

    GatedEventPolicy ten_fps;
    ten_fps.set_infer_fps(10.0);
    for (int i = 0; i < 12; ++i) {
        decision = ten_fps.update(strong());
        require(!decision.alarm, "10 FPS path must preserve the 8 FPS time window");
    }
    decision = ten_fps.update(strong());
    require(decision.alarm, "10 FPS path must alarm after the scaled time window");

    GatedEventPolicy low_raw_confidence;
    for (int i = 0; i < 10; ++i) {
        auto item = strong();
        item.high_confidence_phone = false;
        decision = low_raw_confidence.update(item);
    }
    require(!decision.alarm, "low raw phone confidence must not alarm");
    require(decision.reject_reason == "raw_phone_confidence_not_confirmed",
            "raw confidence reject reason");

    GatedEventPolicy no_screen_ray;
    for (int i = 0; i < 10; ++i) {
        auto item = strong();
        item.screen_ray_hit = false;
        decision = no_screen_ray.update(item);
    }
    require(!decision.alarm, "angle-only screen evidence must not alarm");
    require(!decision.review, "screen-ray-free evidence must remain internal observation");

    GatedEventPolicy far_from_hand;
    for (int i = 0; i < 16; ++i) {
        auto item = strong();
        item.hand_relation = EvidenceState::false_value;
        decision = far_from_hand.update(item);
    }
    require(!decision.alarm, "false hand evidence must never be compensated");
    require(decision.reject_reason == "hand_relation_false", "false hand reject reason");

    GatedEventPolicy wrong_direction;
    for (int i = 0; i < 16; ++i) {
        auto item = strong();
        item.screen_intent = EvidenceState::false_value;
        decision = wrong_direction.update(item);
    }
    require(!decision.alarm, "false screen intent must never be compensated");
    require(!decision.review, "wrong-direction evidence must remain internal observation");

    GatedEventPolicy static_phone;
    for (int i = 0; i < 8; ++i) decision = static_phone.update(strong());
    auto static_item = strong();
    static_item.static_state = StaticEvidenceState::static_confirmed;
    decision = static_phone.update(static_item);
    require(!decision.alarm, "confirmed static phone must suppress alarm");
    require(decision.state == GatedEventState::static_suppressed, "static state");

    GatedEventPolicy degraded;
    for (int i = 0; i < 16; ++i) {
        auto item = strong();
        item.hand_relation = EvidenceState::unknown;
        item.transition_observed = i == 3;
        decision = degraded.update(item);
    }
    require(!decision.alarm, "degraded evidence must not raise production alarm");
    require(decision.review, "degraded evidence with transition must request review");

    GatedEventPolicy edge_handheld;
    for (int i = 0; i < 10; ++i) {
        auto item = strong();
        item.screen_intent = EvidenceState::unknown;
        item.screen_ray_hit = false;
        decision = edge_handheld.update(item);
    }
    require(!decision.alarm, "edge handheld evidence must not bypass the strict alarm gate");
    require(!decision.review, "screen-unknown handheld evidence must remain internal observation");

    GatedEventPolicy edge_handheld_ten_fps;
    edge_handheld_ten_fps.set_infer_fps(10.0);
    for (int i = 0; i < 12; ++i) {
        auto item = strong();
        item.screen_intent = EvidenceState::unknown;
        item.screen_ray_hit = false;
        decision = edge_handheld_ten_fps.update(item);
        require(!decision.review, "10 FPS screen-unknown evidence must remain internal");
    }
    auto edge_tenth_fps_item = strong();
    edge_tenth_fps_item.screen_intent = EvidenceState::unknown;
    edge_tenth_fps_item.screen_ray_hit = false;
    decision = edge_handheld_ten_fps.update(edge_tenth_fps_item);
    require(!decision.review, "10 FPS screen-unknown evidence must not become a review event");

    // Candidate 10 FPS gate: 9/13 associated phone-person-hand frames,
    // 7/13 screen-intent frames, 8/13 high-confidence phone frames and
    // 2/13 screen-ray hits.
    GatedEventPolicy relaxed_screen_gate;
    relaxed_screen_gate.set_infer_fps(10.0);
    for (int i = 0; i < 13; ++i) {
        auto item = strong();
        item.screen_intent = i < 7
            ? EvidenceState::true_value : EvidenceState::false_value;
        item.screen_ray_hit = i < 2;
        decision = relaxed_screen_gate.update(item);
    }
    require(decision.associated_phone_hits == 13, "associated evidence count");
    require(decision.screen_intent_hits == 7, "seven screen-intent hits");
    require(decision.screen_ray_hits == 2, "two screen-ray hits");
    require(decision.alarm,
            "7/13 screen intent and 2/13 rays must pass despite six direction misses");

    GatedEventPolicy insufficient_screen_intent;
    insufficient_screen_intent.set_infer_fps(10.0);
    for (int i = 0; i < 13; ++i) {
        auto item = strong();
        item.screen_intent = i < 6
            ? EvidenceState::true_value : EvidenceState::false_value;
        item.screen_ray_hit = i < 2;
        decision = insufficient_screen_intent.update(item);
    }
    require(decision.screen_intent_hits == 6, "six screen-intent hits");
    require(!decision.alarm, "6/13 screen-intent hits must not alarm");

    GatedEventPolicy relaxed_association_gate;
    relaxed_association_gate.set_infer_fps(10.0);
    for (int i = 0; i < 13; ++i) {
        auto item = strong();
        item.person_association = i < 4
            ? EvidenceState::false_value : EvidenceState::true_value;
        decision = relaxed_association_gate.update(item);
    }
    require(decision.associated_phone_hits == 9, "nine associated evidence hits");
    require(decision.alarm, "9/13 associated evidence hits must alarm");

    GatedEventPolicy insufficient_association;
    insufficient_association.set_infer_fps(10.0);
    for (int i = 0; i < 13; ++i) {
        auto item = strong();
        item.person_association = i < 5
            ? EvidenceState::false_value : EvidenceState::true_value;
        decision = insufficient_association.update(item);
    }
    require(decision.associated_phone_hits == 8, "eight associated evidence hits");
    require(!decision.alarm, "8/13 associated evidence hits must not alarm");

    GatedEventPolicy insufficient_screen_ray;
    insufficient_screen_ray.set_infer_fps(10.0);
    for (int i = 0; i < 13; ++i) {
        auto item = strong();
        item.screen_intent = i < 7
            ? EvidenceState::true_value : EvidenceState::false_value;
        item.screen_ray_hit = i == 0;
        decision = insufficient_screen_ray.update(item);
    }
    require(decision.screen_ray_hits == 1, "one screen-ray hit");
    require(!decision.alarm, "1/13 screen-ray hit must not alarm");

    // Regression from event 202608130125, camera10 track 2683.  The person is
    // clipped by the lower frame edge: phone/person/hand evidence is strong,
    // while screen direction alternates between false and unknown.
    GatedEventPolicy camera10_lower_edge;
    camera10_lower_edge.set_infer_fps(10.0);
    const bool lower_edge_valid[13] = {
        true, false, false, false, false, true, true,
        true, true, true, true, true, true,
    };
    const bool lower_edge_high[13] = {
        true, false, false, false, false, true, true,
        true, true, true, true, true, true,
    };
    const EvidenceState lower_edge_screen[13] = {
        EvidenceState::false_value, EvidenceState::false_value,
        EvidenceState::unknown, EvidenceState::false_value,
        EvidenceState::false_value, EvidenceState::false_value,
        EvidenceState::false_value, EvidenceState::unknown,
        EvidenceState::false_value, EvidenceState::false_value,
        EvidenceState::false_value, EvidenceState::unknown,
        EvidenceState::unknown,
    };
    for (int i = 0; i < 13; ++i) {
        auto item = strong();
        item.phone_valid = lower_edge_valid[i];
        item.high_confidence_phone = lower_edge_high[i];
        item.screen_intent = lower_edge_screen[i];
        item.screen_ray_hit = false;
        decision = camera10_lower_edge.update(item);
    }
    require(!decision.alarm, "camera10 lower-edge regression must not bypass strict alarm");
    require(!decision.review, "camera10 lower-edge evidence without screen proof stays internal");
    require(decision.handheld_review_hits == 9,
            "camera10 lower-edge regression must preserve nine strong hits");

    GatedEventPolicy low_confidence_handheld;
    for (int i = 0; i < 16; ++i) {
        auto item = strong();
        item.high_confidence_phone = false;
        item.screen_intent = EvidenceState::unknown;
        item.screen_ray_hit = false;
        decision = low_confidence_handheld.update(item);
    }
    require(!decision.review, "low-confidence candidates must not enter handheld review");

    GatedEventPolicy uncertain_hand;
    for (int i = 0; i < 16; ++i) {
        auto item = strong();
        item.hand_relation = EvidenceState::unknown;
        item.screen_intent = EvidenceState::unknown;
        item.screen_ray_hit = false;
        decision = uncertain_hand.update(item);
    }
    require(!decision.review, "unknown hand association must not enter handheld review without transition");

    GatedEventPolicy contradicted_handheld;
    for (int i = 0; i < 10; ++i) {
        auto item = strong();
        item.screen_intent = EvidenceState::unknown;
        item.screen_ray_hit = false;
        if (i == 8) item.screen_intent = EvidenceState::false_value;
        decision = contradicted_handheld.update(item);
    }
    require(!decision.review, "screen direction must not create a separate review candidate");

    GatedEventPolicy static_handheld;
    for (int i = 0; i < 9; ++i) {
        auto item = strong();
        item.screen_intent = EvidenceState::unknown;
        item.screen_ray_hit = false;
        decision = static_handheld.update(item);
    }
    auto fixed_item = strong();
    fixed_item.screen_intent = EvidenceState::unknown;
    fixed_item.screen_ray_hit = false;
    fixed_item.static_state = StaticEvidenceState::static_confirmed;
    decision = static_handheld.update(fixed_item);
    require(!decision.review, "confirmed static phone must not enter handheld review");

    GatedEventPolicy intermittent;
    for (int i = 0; i < 16; ++i) {
        auto item = strong();
        if (i % 3 != 0) {
            item.screen_intent = EvidenceState::false_value;
            item.screen_ray_hit = false;
        }
        decision = intermittent.update(item);
    }
    require(!decision.alarm, "fewer than five of ten screen-intent hits must not alarm");

    return 0;
}
