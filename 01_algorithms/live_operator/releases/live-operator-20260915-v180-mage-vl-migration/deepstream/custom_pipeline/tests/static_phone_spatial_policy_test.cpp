#include "static_phone_spatial_policy.hpp"

#include <cmath>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

using namespace jiankong::custom_pipeline;

static void require(bool value, const char* message = "static spatial policy test failed") {
    if (!value) {
        std::cerr << message << '\n';
        std::exit(1);
    }
}

struct RegressionSample {
    double time_sec = 0.0;
    SpatialRect phone{};
    SpatialRect person{};
    float risk_score = 0.0f;
    bool accepted = false;
};

static std::vector<RegressionSample> load_camera04_regression_fixture() {
    const std::filesystem::path fixture_path =
        std::filesystem::path(__FILE__).parent_path() / "fixtures" /
        "camera04_person299_static.json";
    std::ifstream input(fixture_path);
    require(input.good(), "camera04 regression fixture must be readable");
    std::ostringstream buffer;
    buffer << input.rdbuf();
    const std::string text = buffer.str();
    require(text.find("\"camera\": \"camera04\"") != std::string::npos,
            "camera04 regression fixture identity");
    require(text.find("\"track_id\": 299") != std::string::npos,
            "camera04 regression fixture person track");
    require(text.find("\"infer_fps\": 8.0") != std::string::npos,
            "camera04 regression fixture FPS");
    require(text.find("\"phone_id\"") == std::string::npos,
            "camera04 regression fixture must not require a phone ID");

    const std::string number = "([-+0-9.eE]+)";
    const std::regex sample_pattern(
        "\\{\\s*\\\"time_sec\\\"\\s*:\\s*" + number +
        "\\s*,\\s*\\\"phone_bbox\\\"\\s*:\\s*\\[\\s*" + number +
        "\\s*,\\s*" + number + "\\s*,\\s*" + number + "\\s*,\\s*" + number +
        "\\s*\\]\\s*,\\s*\\\"person_bbox\\\"\\s*:\\s*\\[\\s*" + number +
        "\\s*,\\s*" + number + "\\s*,\\s*" + number + "\\s*,\\s*" + number +
        "\\s*\\]\\s*,\\s*\\\"risk_score\\\"\\s*:\\s*" + number +
        "\\s*,\\s*\\\"accepted\\\"\\s*:\\s*(true|false)\\s*\\}");

    std::vector<RegressionSample> samples;
    for (std::sregex_iterator it(text.begin(), text.end(), sample_pattern), end;
         it != end; ++it) {
        const std::smatch& match = *it;
        RegressionSample sample;
        sample.time_sec = std::stod(match[1].str());
        sample.phone = SpatialRect{
            std::stof(match[2].str()), std::stof(match[3].str()),
            std::stof(match[4].str()), std::stof(match[5].str())};
        sample.person = SpatialRect{
            std::stof(match[6].str()), std::stof(match[7].str()),
            std::stof(match[8].str()), std::stof(match[9].str())};
        sample.risk_score = std::stof(match[10].str());
        sample.accepted = match[11].str() == "true";
        samples.push_back(sample);
    }
    require(samples.size() == 73, "camera04 regression fixture must contain 73 samples");
    require(samples.front().time_sec == 0.0 && samples.back().time_sec == 9.0,
            "camera04 regression fixture must cover the reviewed interval");
    return samples;
}

static SpatialFrameObservation sample(double time_sec, bool detected,
                                      float phone_dx, float person_dx,
                                      float wrist_dx, bool wrist_valid = true) {
    SpatialFrameObservation frame;
    frame.time_sec = time_sec;
    frame.person = SpatialRect{100.0f + person_dx, 80.0f, 220.0f + person_dx, 420.0f};
    frame.wrist_valid = wrist_valid;
    frame.wrist_x = 175.0f + wrist_dx;
    frame.wrist_y = 310.0f;
    if (detected) {
        frame.phones.push_back(SpatialPhoneCandidate{
            SpatialRect{168.0f + phone_dx, 292.0f, 198.0f + phone_dx, 326.0f},
            0.90f, 0.82f, true});
    }
    return frame;
}

static void add_phone(SpatialFrameObservation& frame, float phone_dx, float confidence = 0.90f) {
    frame.phones.push_back(SpatialPhoneCandidate{
        SpatialRect{168.0f + phone_dx, 292.0f, 198.0f + phone_dx, 326.0f},
        confidence, 0.82f, true});
}

static SpatialPoint center_of(const SpatialRect& box) {
    return SpatialPoint{(box.x1 + box.x2) * 0.5f, (box.y1 + box.y2) * 0.5f};
}

static bool matches_primary_cluster(const SpatialStaticDecision& decision,
                                    const SpatialRect& box) {
    const SpatialPoint center = center_of(box);
    for (const SpatialPoint& primary_center : decision.primary_cluster_centers) {
        if (std::hypot(center.x - primary_center.x, center.y - primary_center.y) <=
            decision.primary_cluster_tolerance_px) {
            return true;
        }
    }
    return false;
}

int main() {
    require(classify_fixed_template_spatial_evidence(
                2.0f, 28.0f, 1.05f, 0.95f) ==
            FixedTemplateEvidence::matched,
            "a stable phone at a confirmed template location must match");
    require(classify_fixed_template_spatial_evidence(
                30.0f, 28.0f, 1.0f, 1.0f) ==
            FixedTemplateEvidence::mismatch,
            "a phone moved away from the confirmed location must release");
    require(classify_fixed_template_spatial_evidence(
                2.0f, 28.0f, 2.0f, 1.0f) ==
            FixedTemplateEvidence::mismatch,
            "an implausibly different box at the anchor must not match");
    require(classify_fixed_template_spatial_evidence(
                100.0f, 28.0f, 1.0f, 1.0f) ==
            FixedTemplateEvidence::none,
            "an unrelated distant phone must not inherit template evidence");

    SpatialStaticConfig config;
    SpatialStaticPolicy static_phone(config);
    SpatialStaticDecision last;
    for (int i = 0; i < 27; ++i) {
        const bool detected = i % 5 != 0;
        const float jitter = static_cast<float>((i % 3) - 1);
        last = static_phone.observe(sample(i / 8.0, detected, jitter, i * 0.7f, i * 1.0f), 0.0f);
    }
    require(last.phase == SpatialStaticPhase::suppressed, "static jitter must suppress");
    require(last.metrics.detection_ratio >= 0.60f, "static jitter detection ratio");
    require(last.metrics.center_spread_px <= 8.0f, "static jitter center spread");
    require(last.metrics.motion_decoupled, "static jitter motion decoupling");

    SpatialStaticPolicy handheld(config);
    for (int i = 0; i < 25; ++i) {
        last = handheld.observe(sample(i / 8.0, true, i * 2.0f, i * 2.0f, i * 2.0f), 0.0f);
    }
    require(last.phase == SpatialStaticPhase::handheld_or_moving, "handheld must exit");
    require(last.replay_shadow, "handheld must replay shadow");

    SpatialStaticPolicy ambiguous(config);
    for (int i = 0; i < 25; ++i) {
        last = ambiguous.observe(sample(i / 8.0, true, 0.0f, 0.0f, 0.0f), 0.0f);
    }
    require(last.phase != SpatialStaticPhase::suppressed, "ambiguous must not suppress");
    require(last.replay_shadow, "ambiguous must replay shadow");

    SpatialStaticPolicy hotspot(config);
    last = hotspot.observe(sample(0.0, true, 0.0f, 0.0f, 0.0f), 1.0f);
    require(last.phase != SpatialStaticPhase::suppressed, "hotspot must not suppress immediately");
    last = hotspot.observe(sample(1.0 / 8.0, true, 0.5f, 0.0f, 0.0f), 1.0f);
    last = hotspot.observe(sample(2.0 / 8.0, true, -0.5f, 0.0f, 0.0f), 1.0f);
    require(last.phase == SpatialStaticPhase::pending && last.hold_candidate,
            "a known hotspot with three stable detections must hold alarm history early");
    require(std::string(last.reason) == "known_hotspot_pending",
            "the early hotspot hold must be distinguishable in diagnostics");
    for (int i = 3; i < 25; ++i) {
        last = hotspot.observe(sample(i / 8.0, true, 0.0f, 0.0f, 0.0f), 1.0f);
    }
    require(last.phase == SpatialStaticPhase::suppressed, "hotspot must suppress after short window");
    require(last.discard_shadow, "hotspot suppression must discard shadow");

    SpatialStaticPolicy fixed_template(config);
    for (int i = 0; i < 3; ++i) {
        SpatialFrameObservation frame =
            sample(i / 8.0, true, 0.0f, 0.0f, 0.0f);
        frame.phones[0].fixed_template_evidence =
            FixedTemplateEvidence::matched;
        frame.phones[0].fixed_template_score = 0.91f;
        last = fixed_template.observe(frame, 0.0f);
    }
    require(last.phase == SpatialStaticPhase::pending && last.hold_candidate,
            "three location-matched template observations must hold early");
    require(std::string(last.reason) == "known_hotspot_pending",
            "manual template match uses the conservative hotspot hold path");

    SpatialFrameObservation changed_context =
        sample(3.0 / 8.0, true, 0.0f, 0.0f, 0.0f);
    changed_context.phones[0].fixed_template_evidence =
        FixedTemplateEvidence::mismatch;
    changed_context.phones[0].fixed_template_score = 0.31f;
    last = fixed_template.observe(changed_context, 4.0f);
    require(last.phase == SpatialStaticPhase::handheld_or_moving,
            "template location mismatch must override a saturated hotspot");
    require(last.replay_shadow,
            "template location mismatch must restore held behavior history");
    require(std::string(last.reason) == "fixed_template_mismatch",
            "template mismatch must be explicit in diagnostics");

    SpatialStaticConfig manual_only_config;
    manual_only_config.manual_templates_only = true;
    SpatialStaticPolicy manual_only_unknown(manual_only_config);
    for (int i = 0; i < 27; ++i) {
        last = manual_only_unknown.observe(
            sample(i / 8.0, true, 0.0f, i * 0.7f, i * 1.0f), 4.0f);
    }
    require(last.phase == SpatialStaticPhase::handheld_or_moving,
            "an unconfirmed stationary phone must not be automatically suppressed");
    require(std::string(last.reason) == "manual_template_required",
            "manual-only mode must identify the missing fixed-phone template");

    SpatialStaticPolicy manual_only_template(manual_only_config);
    for (int i = 0; i < 25; ++i) {
        SpatialFrameObservation frame =
            sample(i / 8.0, true, 0.0f, i * 0.7f, i * 1.0f);
        frame.phones[0].fixed_template_evidence = FixedTemplateEvidence::matched;
        last = manual_only_template.observe(frame, 0.0f);
    }
    require(last.phase == SpatialStaticPhase::suppressed,
            "a user-confirmed stationary-phone template must still suppress");

    SpatialStaticPolicy new_location(config);
    for (int i = 0; i < 3; ++i) {
        last = new_location.observe(sample(i / 8.0, true, 0.0f, 0.0f, 0.0f), 0.0f);
    }
    require(last.phase == SpatialStaticPhase::observed && !last.hold_candidate,
            "a new location must keep the normal probation window");

    SpatialStaticPolicy hotspot_pickup(config);
    for (int i = 0; i < 3; ++i) {
        last = hotspot_pickup.observe(sample(i / 8.0, true, 0.0f, 0.0f, 0.0f), 4.0f);
    }
    hotspot_pickup.append_shadow(ShadowRiskSample{0.25, 0.82f, true});
    for (int i = 3; i < 6; ++i) {
        const float motion = 20.0f * static_cast<float>(i - 2);
        last = hotspot_pickup.observe(
            sample(i / 8.0, true, motion, 0.0f, motion), 4.0f);
    }
    require(last.phase == SpatialStaticPhase::handheld_or_moving,
            "a phone picked up from a hotspot must leave early hold");
    require(last.replay_shadow,
            "a phone picked up from a hotspot must restore held behavior history");

    SpatialStaticPolicy picked_up(config);
    for (int i = 0; i < 16; ++i) {
        last = picked_up.observe(sample(i / 8.0, true, 0.0f, 0.0f, 0.0f), 0.0f);
    }
    picked_up.append_shadow(ShadowRiskSample{2.0, 0.82f, true});
    for (int i = 0; i < 3; ++i) {
        const float motion = 20.0f * static_cast<float>(i + 1);
        last = picked_up.observe(sample((16 + i) / 8.0, true, motion, 0.0f, motion), 0.0f);
    }
    require(last.phase == SpatialStaticPhase::handheld_or_moving, "picked up phone must exit");
    require(last.replay_shadow, "picked up phone must replay shadow");

    SpatialStaticPolicy disappeared(config);
    for (int i = 0; i < 8; ++i) {
        last = disappeared.observe(sample(i / 8.0, true, 0.0f, 0.0f, 0.0f), 0.0f);
    }
    last = disappeared.observe(sample(1.75, false, 0.0f, 0.0f, 0.0f), 0.0f);
    require(last.phase == SpatialStaticPhase::observed);
    require(last.replay_shadow);

    SpatialStaticPolicy shadow(config);
    shadow.append_shadow(ShadowRiskSample{0.25, 0.40f, false});
    shadow.append_shadow(ShadowRiskSample{0.50, 0.80f, true});
    std::vector<ShadowRiskSample> replay = shadow.take_replay();
    require(replay.size() == 2);
    require(replay[0].time_sec == 0.25);
    require(replay[1].risk_score == 0.80f);
    require(shadow.take_replay().empty());

    shadow.append_shadow(ShadowRiskSample{0.75, 0.90f, true});
    shadow.discard_shadow();
    require(shadow.take_replay().empty());
    shadow.append_shadow(ShadowRiskSample{1.00, 0.90f, true});
    shadow.reset_short_state();
    require(shadow.take_replay().empty());
    last = shadow.observe(sample(2.0, true, 0.0f, 0.0f, 0.0f), 0.0f);
    require(last.phase == SpatialStaticPhase::observed);

    SpatialStaticPolicy second_phone(config);
    for (int i = 0; i < 25; ++i) {
        SpatialFrameObservation frame = sample(i / 8.0, true, 0.0f, i * 0.7f, i * 1.0f);
        if (i >= 22) add_phone(frame, 80.0f + 8.0f * static_cast<float>(i - 22), 0.99f);
        last = second_phone.observe(frame, 1.0f);
    }
    require(last.phase == SpatialStaticPhase::suppressed,
            "a moving secondary phone must not release the dominant static cluster");

    SpatialStaticPolicy primary_index_policy(config);
    for (int i = 0; i < 27; ++i) {
        SpatialFrameObservation frame = sample(
            i / 8.0, false, 0.0f, i * 0.7f, i * 1.0f);
        const bool primary_detected = i != 21;
        const bool secondary_detected = i % 3 == 0;
        std::size_t primary_index = 0;
        const auto append_primary = [&]() {
            primary_index = frame.phones.size();
            add_phone(frame, static_cast<float>((i % 3) - 1), 0.55f);
        };
        const auto append_secondary = [&]() {
            add_phone(frame, 80.0f + 12.0f * static_cast<float>(i), 0.99f);
        };
        if (i % 2 == 0) {
            if (primary_detected) append_primary();
            if (secondary_detected) append_secondary();
        } else {
            if (secondary_detected) append_secondary();
            if (primary_detected) append_primary();
        }

        last = primary_index_policy.observe(frame, 1.0f);
        if (i < 6) continue;
        if (primary_detected) {
            require(last.primary_candidate_index.has_value(),
                    "the historical primary cluster must identify its current candidate");
            require(*last.primary_candidate_index == primary_index,
                    "the primary candidate index must follow current observation order");
        } else {
            require(!last.primary_candidate_index.has_value(),
                    "a secondary candidate must not replace a dropped primary cluster");
        }
    }
    require(last.phase == SpatialStaticPhase::suppressed,
            "the stable primary cluster must remain suppressed after reordered inputs");

    SpatialStaticConfig diagonal_config;
    diagonal_config.radius_ratio = 0.0f;
    diagonal_config.min_radius_px = 8.0f;
    SpatialStaticPolicy primary_centers_policy(diagonal_config);
    std::vector<SpatialRect> primary_boxes;
    SpatialRect secondary_box{};
    for (int i = 0; i <= 24; ++i) {
        SpatialFrameObservation frame = sample(
            i / 8.0, false, 0.0f, i * 0.7f, i * 1.0f);
        const float primary_dx = static_cast<float>((i % 3) - 1);
        add_phone(frame, primary_dx, 0.99f);
        primary_boxes.push_back(frame.phones.back().box);
        add_phone(frame, 12.0f, 0.55f);
        frame.phones.back().box.y1 += 12.0f;
        frame.phones.back().box.y2 += 12.0f;
        secondary_box = frame.phones.back().box;
        last = primary_centers_policy.observe(frame, 1.0f);
    }
    require(last.phase == SpatialStaticPhase::suppressed,
            "the stable primary cluster must suppress beside a diagonal secondary");
    require(last.primary_cluster_tolerance_px == 16.0f,
            "cluster membership tolerance must equal twice the analysis radius");
    require(!last.primary_cluster_centers.empty(),
            "a final primary cluster must expose representative centers");
    for (const SpatialRect& box : primary_boxes) {
        require(matches_primary_cluster(last, box),
                "primary cluster centers must match every primary representative center");
    }
    require(!matches_primary_cluster(last, secondary_box),
            "a diagonal secondary beyond the Euclidean tolerance must remain separate");

    SpatialStaticConfig duplicate_config;
    duplicate_config.radius_ratio = 0.0f;
    SpatialStaticPolicy duplicate_boxes(duplicate_config);
    SpatialFrameObservation duplicated = sample(0.0, true, 0.0f, 0.0f, 0.0f);
    add_phone(duplicated, 4.0f, 0.91f);
    add_phone(duplicated, 8.0f, 0.92f);
    last = duplicate_boxes.observe(duplicated, 0.0f);
    require(last.phase == SpatialStaticPhase::observed,
            "same-frame duplicate boxes must not count as sustained motion");

    SpatialStaticPolicy single_outlier(duplicate_config);
    for (int i = 0; i < 8; ++i) {
        last = single_outlier.observe(sample(i / 8.0, true, 0.0f, 0.0f, 0.0f), 0.0f);
    }
    last = single_outlier.observe(sample(1.0, true, 8.0f, 0.0f, 0.0f), 0.0f);
    require(last.phase != SpatialStaticPhase::handheld_or_moving,
            "one spatial outlier must not count as sustained motion");

    SpatialStaticPolicy leading_empty(config);
    for (int i = 0; i <= 16; ++i) {
        last = leading_empty.observe(sample(i / 8.0, false, 0.0f, 0.0f, 0.0f), 1.0f);
    }
    for (int i = 17; i <= 31; ++i) {
        last = leading_empty.observe(sample(i / 8.0, true, 0.0f, 0.0f, 0.0f), 1.0f);
    }
    require(last.phase != SpatialStaticPhase::suppressed,
            "leading phone-free history must not complete the static window");

    SpatialStaticPolicy pending_hold(config);
    for (int i = 0; i <= 6; ++i) {
        last = pending_hold.observe(sample(i / 8.0, true, 0.0f, 0.0f, 0.0f), 0.0f);
    }
    require(last.phase == SpatialStaticPhase::pending && last.hold_candidate,
            "0.75 seconds of stable detections must enter pending and hold risk");

    SpatialStaticPolicy low_detection_rate(config);
    for (int i = 0; i <= 16; ++i) {
        last = low_detection_rate.observe(
            sample(i / 8.0, i % 2 == 0, 0.0f, 0.0f, 0.0f), 1.0f);
        require(last.phase != SpatialStaticPhase::pending &&
                    last.phase != SpatialStaticPhase::suppressed,
                "a detection ratio below 0.60 must not enter static handling");
    }

    SpatialStaticPolicy long_path(config);
    for (int i = 0; i <= 48; ++i) {
        const float slow_person_motion = 0.25f * static_cast<float>(i);
        last = long_path.observe(
            sample(i / 8.0, true, 0.0f, slow_person_motion, slow_person_motion), 0.0f);
        if (i < 48) {
            require(last.phase != SpatialStaticPhase::suppressed,
                    "the long static path must not confirm before six seconds");
        }
    }
    require(last.phase == SpatialStaticPhase::suppressed,
            "the six-second path must use retained motion-decoupling evidence");

    SpatialStaticPolicy invalid_input(config);
    for (int i = 0; i <= 6; ++i) {
        last = invalid_input.observe(sample(i / 8.0, true, 0.0f, 0.0f, 0.0f), 0.0f);
    }
    SpatialFrameObservation invalid_time = sample(
        std::numeric_limits<double>::quiet_NaN(), true, 0.0f, 0.0f, 0.0f);
    last = invalid_input.observe(invalid_time, 0.0f);
    require(last.phase == SpatialStaticPhase::pending && last.hold_candidate,
            "a non-finite timestamp must not corrupt pending state");
    SpatialFrameObservation invalid_coordinate = sample(1.0, true, 0.0f, 0.0f, 0.0f);
    add_phone(invalid_coordinate, std::numeric_limits<float>::quiet_NaN(), 1.0f);
    last = invalid_input.observe(invalid_coordinate, std::numeric_limits<float>::quiet_NaN());
    require(last.phase != SpatialStaticPhase::handheld_or_moving,
            "non-finite coordinates must not create a movement exit");

    SpatialStaticConfig invalid_config;
    invalid_config.short_seconds = std::numeric_limits<double>::quiet_NaN();
    invalid_config.long_seconds = -1.0;
    invalid_config.pending_seconds = std::numeric_limits<double>::infinity();
    invalid_config.max_gap_seconds = 0.0;
    invalid_config.min_detection_ratio = 2.0f;
    invalid_config.radius_ratio = -1.0f;
    invalid_config.min_radius_px = std::numeric_limits<float>::quiet_NaN();
    invalid_config.min_bbox_iou = -1.0f;
    invalid_config.max_bbox_size_change = std::numeric_limits<float>::infinity();
    invalid_config.wrist_follow_cosine = 2.0f;
    SpatialStaticPolicy defensive_config(invalid_config);
    for (int i = 0; i <= 6; ++i) {
        last = defensive_config.observe(sample(i / 8.0, true, 0.0f, 0.0f, 0.0f), 0.0f);
    }
    require(last.phase == SpatialStaticPhase::pending && last.hold_candidate,
            "invalid configuration must fall back without corrupting state");

    const std::vector<RegressionSample> camera04_samples =
        load_camera04_regression_fixture();
    SpatialStaticPolicy camera04_policy(config);
    std::deque<int> formal_history;
    int maximum_formal_hits = 0;
    int pending_shadow_samples = 0;
    bool saw_pending = false;
    bool saw_suppressed = false;
    bool suppressed_by_motion_decoupling = false;
    bool discarded_pending_risk = false;
    for (const RegressionSample& regression : camera04_samples) {
        SpatialFrameObservation frame;
        frame.time_sec = regression.time_sec;
        frame.person = regression.person;
        frame.phones.push_back(SpatialPhoneCandidate{
            regression.phone, 1.0f, regression.risk_score, regression.accepted});
        // The fixture itself must supply the spatial evidence for suppression.
        const SpatialStaticDecision decision = camera04_policy.observe(frame, 0.0f);
        saw_pending = saw_pending || decision.phase == SpatialStaticPhase::pending;
        saw_suppressed = saw_suppressed || decision.phase == SpatialStaticPhase::suppressed;

        const bool risky = regression.accepted && regression.risk_score >= 0.65f;
        if (decision.hold_candidate && decision.primary_candidate_index.has_value()) {
            camera04_policy.append_shadow(
                ShadowRiskSample{regression.time_sec, regression.risk_score, risky});
            ++pending_shadow_samples;
        }

        if (decision.discard_shadow) {
            camera04_policy.discard_shadow();
            std::fill(formal_history.begin(), formal_history.end(), 0);
            discarded_pending_risk = pending_shadow_samples > 0 &&
                camera04_policy.take_replay().empty();
        } else if (decision.replay_shadow) {
            for (const ShadowRiskSample& replay : camera04_policy.take_replay()) {
                formal_history.push_back(replay.accepted ? 1 : 0);
            }
        }

        const bool block_primary = decision.hold_candidate || decision.discard_shadow ||
            decision.phase == SpatialStaticPhase::suppressed;
        formal_history.push_back(!block_primary && risky ? 1 : 0);
        while (formal_history.size() > 30) formal_history.pop_front();
        int formal_hits = 0;
        for (int hit : formal_history) formal_hits += hit;
        maximum_formal_hits = std::max(maximum_formal_hits, formal_hits);
        if (decision.phase == SpatialStaticPhase::suppressed) {
            suppressed_by_motion_decoupling = decision.metrics.motion_decoupled &&
                std::string(decision.reason) == "motion_decoupled";
            break;
        }
    }
    require(maximum_formal_hits < 16,
            "camera04 static phone must remain below the S4_ALARM threshold");
    require(saw_pending, "camera04 static phone must enter pending before suppression");
    require(saw_suppressed, "camera04 static phone must reach suppressed");
    require(suppressed_by_motion_decoupling,
            "camera04 suppression must come from fixture motion decoupling");
    require(discarded_pending_risk,
            "camera04 pending risk must be discarded after suppression");
    return 0;
}
