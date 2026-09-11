#pragma once

#include <algorithm>
#include <cmath>
#include <deque>
#include <string>

namespace jiankong::custom_pipeline {

enum class EvidenceState { false_value, unknown, true_value };
enum class StaticEvidenceState { unknown, moving_or_handheld, static_confirmed };

enum class GatedEventState {
    no_phone,
    phone_probation,
    hand_confirmed,
    screen_aim_confirmed,
    stable_aiming,
    alarm,
    static_suppressed,
    degraded_observation,
    review
};

struct GatedFrameEvidence {
    bool phone_valid = false;
    // The detection threshold admits a phone-like candidate.  These two
    // stricter, independent observations are only required for an alarm.
    bool high_confidence_phone = false;
    bool screen_ray_hit = false;
    EvidenceState person_association = EvidenceState::unknown;
    EvidenceState hand_relation = EvidenceState::unknown;
    EvidenceState screen_intent = EvidenceState::unknown;
    StaticEvidenceState static_state = StaticEvidenceState::unknown;
    bool transition_observed = false;
};

struct GatedEventDecision {
    GatedEventState state = GatedEventState::no_phone;
    bool alarm = false;
    bool review = false;
    int phone_hits = 0;
    int hand_hits = 0;
    int aim_hits = 0;
    int core_hits = 0;
    int associated_phone_hits = 0;
    int screen_intent_hits = 0;
    int high_confidence_phone_hits = 0;
    int screen_ray_hits = 0;
    int handheld_review_hits = 0;
    std::string reject_reason = "no_phone";
};

inline const char* evidence_name(EvidenceState state) {
    switch (state) {
        case EvidenceState::true_value: return "true";
        case EvidenceState::false_value: return "false";
        default: return "unknown";
    }
}

inline const char* static_evidence_name(StaticEvidenceState state) {
    switch (state) {
        case StaticEvidenceState::moving_or_handheld: return "moving_or_handheld";
        case StaticEvidenceState::static_confirmed: return "static_confirmed";
        default: return "unknown";
    }
}

inline const char* gated_event_state_name(GatedEventState state) {
    switch (state) {
        case GatedEventState::phone_probation: return "S1_PHONE_PROBATION";
        case GatedEventState::hand_confirmed: return "S2_HAND_CONFIRMED";
        case GatedEventState::screen_aim_confirmed: return "S3_SCREEN_AIM_CONFIRMED";
        case GatedEventState::stable_aiming: return "S4_STABLE_AIMING";
        case GatedEventState::alarm: return "S5_ALARM";
        case GatedEventState::static_suppressed: return "STATIC_SUPPRESSED";
        case GatedEventState::degraded_observation: return "DEGRADED_OBSERVATION";
        case GatedEventState::review: return "REVIEW";
        default: return "S0_NO_PHONE";
    }
}

class GatedEventPolicy {
public:
    void set_infer_fps(double infer_fps) {
        infer_fps_ = infer_fps == 10.0 ? 10.0 : 8.0;
    }

    GatedEventDecision update(const GatedFrameEvidence& evidence) {
        history_.push_back(evidence);
        while (history_.size() > scaled(16)) history_.pop_front();

        GatedEventDecision decision;
        decision.phone_hits = count_last(scaled(4), [](const auto& item) { return item.phone_valid; });
        decision.hand_hits = count_last(scaled(6), [](const auto& item) {
            return item.hand_relation == EvidenceState::true_value;
        });
        decision.aim_hits = count_last(scaled(8), [](const auto& item) {
            return item.screen_intent == EvidenceState::true_value;
        });
        decision.core_hits = count_last(scaled(10), [](const auto& item) { return core_true(item); });
        decision.associated_phone_hits = count_last(scaled(10), [](const auto& item) {
            return item.phone_valid &&
                item.person_association == EvidenceState::true_value &&
                item.hand_relation == EvidenceState::true_value &&
                item.static_state != StaticEvidenceState::static_confirmed;
        });
        decision.screen_intent_hits = count_last(scaled(10), [](const auto& item) {
            return item.screen_intent == EvidenceState::true_value;
        });
        decision.high_confidence_phone_hits = count_last(scaled(10), [](const auto& item) {
            return item.high_confidence_phone;
        });
        decision.screen_ray_hits = count_last(scaled(10), [](const auto& item) {
            return item.screen_ray_hit;
        });
        decision.handheld_review_hits = count_last(scaled(10), [](const auto& item) {
            return item.phone_valid && item.high_confidence_phone &&
                item.person_association == EvidenceState::true_value &&
                item.hand_relation == EvidenceState::true_value &&
                item.static_state != StaticEvidenceState::static_confirmed;
        });

        if (evidence.static_state == StaticEvidenceState::static_confirmed) {
            decision.state = GatedEventState::static_suppressed;
            decision.reject_reason = "static_phone";
            alarm_latched_ = false;
            return decision;
        }

        if (consecutive_last(scaled(4), [](const auto& item) { return !item.phone_valid; })) {
            reset();
            return decision;
        }
        if (consecutive_last(scaled(4), [](const auto& item) {
                return item.hand_relation == EvidenceState::false_value;
            })) {
            decision.reject_reason = "hand_relation_false";
            alarm_latched_ = false;
            return decision;
        }
        if (alarm_latched_) {
            decision.state = GatedEventState::alarm;
            decision.alarm = true;
            decision.reject_reason.clear();
            return decision;
        }

        if (history_.size() >= scaled(10) &&
            decision.associated_phone_hits >= static_cast<int>(scaled(7)) &&
            decision.screen_intent_hits >= static_cast<int>(scaled(5)) &&
            decision.high_confidence_phone_hits >= static_cast<int>(scaled(6)) &&
            decision.screen_ray_hits >= static_cast<int>(scaled(1)) &&
            !any_last(scaled(4), [](const auto& item) {
                return !item.phone_valid ||
                    item.person_association == EvidenceState::false_value ||
                    item.hand_relation == EvidenceState::false_value ||
                    item.static_state == StaticEvidenceState::static_confirmed;
            })) {
            alarm_latched_ = true;
            decision.state = GatedEventState::alarm;
            decision.alarm = true;
            decision.reject_reason.clear();
            return decision;
        }

        const int degraded_hits = count_last(scaled(16), [](const auto& item) {
            return item.phone_valid &&
                item.person_association != EvidenceState::false_value &&
                item.hand_relation != EvidenceState::false_value &&
                item.screen_intent != EvidenceState::false_value &&
                item.static_state != StaticEvidenceState::static_confirmed;
        });
        const bool transition_seen = any_last(scaled(16), [](const auto& item) {
            return item.transition_observed;
        });
        if (history_.size() >= scaled(16) &&
            degraded_hits >= static_cast<int>(scaled(12)) && transition_seen) {
            decision.state = GatedEventState::review;
            decision.review = true;
            decision.reject_reason = "degraded_review";
            return decision;
        }

        if (decision.phone_hits < static_cast<int>(scaled(3))) {
            decision.state = GatedEventState::no_phone;
            decision.reject_reason = "phone_not_confirmed";
        } else if (evidence.person_association == EvidenceState::false_value) {
            decision.state = GatedEventState::phone_probation;
            decision.reject_reason = "person_association_false";
        } else if (decision.hand_hits < static_cast<int>(scaled(4))) {
            decision.state = evidence.hand_relation == EvidenceState::unknown
                ? GatedEventState::degraded_observation
                : GatedEventState::phone_probation;
            decision.reject_reason = evidence.hand_relation == EvidenceState::unknown
                ? "hand_relation_unknown" : "hand_not_confirmed";
        } else if (decision.screen_intent_hits < static_cast<int>(scaled(5))) {
            decision.state = evidence.screen_intent == EvidenceState::unknown
                ? GatedEventState::degraded_observation
                : GatedEventState::hand_confirmed;
            decision.reject_reason = evidence.screen_intent == EvidenceState::unknown
                ? "screen_intent_unknown" : "screen_aim_not_confirmed";
        } else if (decision.associated_phone_hits < static_cast<int>(scaled(7))) {
            decision.state = GatedEventState::screen_aim_confirmed;
            decision.reject_reason = "phone_person_hand_not_confirmed";
        } else if (decision.high_confidence_phone_hits < static_cast<int>(scaled(6))) {
            decision.state = GatedEventState::stable_aiming;
            decision.reject_reason = "raw_phone_confidence_not_confirmed";
        } else if (decision.screen_ray_hits < static_cast<int>(scaled(1))) {
            decision.state = GatedEventState::stable_aiming;
            decision.reject_reason = "screen_ray_not_confirmed";
        } else {
            decision.state = GatedEventState::stable_aiming;
            decision.reject_reason = "alarm_duration_not_met";
        }
        return decision;
    }

    void reset() {
        history_.clear();
        alarm_latched_ = false;
    }

private:
    std::size_t scaled(std::size_t frames_at_8fps) const {
        return static_cast<std::size_t>(std::ceil(
            static_cast<double>(frames_at_8fps) * infer_fps_ / 8.0));
    }

    double infer_fps_ = 8.0;
    static bool core_true(const GatedFrameEvidence& item) {
        return item.phone_valid &&
            item.person_association == EvidenceState::true_value &&
            item.hand_relation == EvidenceState::true_value &&
            item.screen_intent == EvidenceState::true_value &&
            item.static_state != StaticEvidenceState::static_confirmed;
    }

    static bool core_false(const GatedFrameEvidence& item) {
        return !item.phone_valid ||
            item.person_association == EvidenceState::false_value ||
            item.hand_relation == EvidenceState::false_value ||
            item.screen_intent == EvidenceState::false_value ||
            item.static_state == StaticEvidenceState::static_confirmed;
    }

    template <typename Predicate>
    int count_last(std::size_t count, Predicate predicate) const {
        int hits = 0;
        const std::size_t begin = history_.size() > count ? history_.size() - count : 0;
        for (std::size_t i = begin; i < history_.size(); ++i) hits += predicate(history_[i]) ? 1 : 0;
        return hits;
    }

    template <typename Predicate>
    bool any_last(std::size_t count, Predicate predicate) const {
        return count_last(count, predicate) > 0;
    }

    template <typename Predicate>
    bool consecutive_last(std::size_t count, Predicate predicate) const {
        if (history_.size() < count) return false;
        for (std::size_t i = history_.size() - count; i < history_.size(); ++i) {
            if (!predicate(history_[i])) return false;
        }
        return true;
    }

    std::deque<GatedFrameEvidence> history_;
    bool alarm_latched_ = false;
};

}  // namespace jiankong::custom_pipeline
