#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <deque>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

namespace jiankong::custom_pipeline {

struct SpatialRect { float x1, y1, x2, y2; };
struct SpatialPoint { float x, y; };
enum class FixedTemplateEvidence { none, matched, mismatch };

inline FixedTemplateEvidence classify_fixed_template_spatial_evidence(
    float center_distance,
    float reference_diagonal,
    float width_ratio,
    float height_ratio) {
    if (!std::isfinite(center_distance) || center_distance < 0.0f ||
        !std::isfinite(reference_diagonal) || reference_diagonal <= 0.0f ||
        !std::isfinite(width_ratio) || !std::isfinite(height_ratio)) {
        return FixedTemplateEvidence::none;
    }
    const float guard_radius = std::max(96.0f, reference_diagonal * 3.0f);
    if (center_distance > guard_radius) return FixedTemplateEvidence::none;

    const float close_radius = std::max(10.0f, reference_diagonal * 0.50f);
    const bool size_matches =
        width_ratio >= 0.65f && width_ratio <= 1.55f &&
        height_ratio >= 0.65f && height_ratio <= 1.55f;
    return center_distance <= close_radius && size_matches
        ? FixedTemplateEvidence::matched
        : FixedTemplateEvidence::mismatch;
}

struct SpatialPhoneCandidate {
    SpatialRect box;
    float confidence = 0.0f;
    float risk_score = 0.0f;
    bool accepted = false;
    FixedTemplateEvidence fixed_template_evidence = FixedTemplateEvidence::none;
    float fixed_template_score = 0.0f;
};
struct SpatialFrameObservation {
    double time_sec = 0.0;
    SpatialRect person{};
    bool wrist_valid = false;
    float wrist_x = 0.0f;
    float wrist_y = 0.0f;
    std::vector<SpatialPhoneCandidate> phones;
};
struct ShadowRiskSample {
    double time_sec = 0.0;
    float risk_score = 0.0f;
    bool accepted = false;
};
struct SpatialStaticConfig {
    double short_seconds = 3.0;
    double long_seconds = 6.0;
    double pending_seconds = 0.75;
    double max_gap_seconds = 0.75;
    float min_detection_ratio = 0.60f;
    float radius_ratio = 0.03f;
    float min_radius_px = 8.0f;
    float min_bbox_iou = 0.45f;
    float max_bbox_size_change = 0.35f;
    float wrist_follow_cosine = 0.65f;
    float hotspot_early_hold_score = 1.0f;
    int hotspot_early_hold_samples = 3;
    // Live inference may require a user-confirmed fixed-phone template before
    // withholding a stationary phone. Defaults to false for regression users.
    bool manual_templates_only = false;
};
enum class SpatialStaticPhase { observed, pending, suppressed, handheld_or_moving };
struct SpatialStaticMetrics {
    int sample_count = 0;
    float detection_ratio = 0.0f;
    float center_spread_px = 0.0f;
    float bbox_iou_median = 0.0f;
    float person_motion_px = 0.0f;
    float wrist_motion_px = 0.0f;
    bool motion_decoupled = false;
    bool follows_wrist = false;
};
struct SpatialStaticDecision {
    SpatialStaticPhase phase = SpatialStaticPhase::observed;
    SpatialStaticMetrics metrics{};
    std::optional<std::size_t> primary_candidate_index;
    std::vector<SpatialPoint> primary_cluster_centers;
    float primary_cluster_tolerance_px = 0.0f;
    bool hold_candidate = false;
    bool discard_shadow = false;
    bool replay_shadow = false;
    const char* reason = "observed";
};

class SpatialStaticPolicy {
public:
    explicit SpatialStaticPolicy(SpatialStaticConfig config = {}) : config_(normalize_config(config)) {}

    SpatialStaticDecision observe(const SpatialFrameObservation& frame, float hotspot_score) {
        if (!std::isfinite(frame.time_sec)) {
            SpatialStaticDecision decision;
            decision.phase = current_phase_;
            decision.hold_candidate = current_phase_ == SpatialStaticPhase::pending;
            decision.reason = "invalid_observation";
            return decision;
        }

        SpatialFrameObservation clean_frame = frame;
        if (!valid_rect(clean_frame.person)) clean_frame.person = SpatialRect{};
        clean_frame.wrist_valid = clean_frame.wrist_valid &&
            std::isfinite(clean_frame.wrist_x) && std::isfinite(clean_frame.wrist_y);
        for (SpatialPhoneCandidate& phone : clean_frame.phones) {
            if (!std::isfinite(phone.confidence)) phone.confidence = 0.0f;
            if (!std::isfinite(phone.risk_score)) phone.risk_score = 0.0f;
        }
        if (!std::isfinite(hotspot_score)) hotspot_score = 0.0f;

        if (has_last_time_ && clean_frame.time_sec < last_time_sec_) {
            reset_short_state();
        }
        has_last_time_ = true;
        last_time_sec_ = clean_frame.time_sec;

        observations_.push_back(std::move(clean_frame));
        const SpatialFrameObservation& current_frame = observations_.back();
        prune(current_frame.time_sec);
        if (!current_frame.phones.empty()) {
            last_detection_time_sec_ = current_frame.time_sec;
            has_detection_ = true;
        }

        SpatialStaticDecision decision;
        const Analysis short_analysis = analyze(current_frame.time_sec, config_.short_seconds);
        const Analysis long_analysis = analyze(current_frame.time_sec, config_.long_seconds);
        decision.metrics = short_analysis.metrics;
        decision.primary_candidate_index = short_analysis.primary_candidate_index;
        decision.primary_cluster_centers = short_analysis.primary_cluster_centers;
        decision.primary_cluster_tolerance_px = short_analysis.primary_cluster_tolerance_px;

        const bool gap_expired = has_detection_ && current_frame.phones.empty() &&
            current_frame.time_sec - last_detection_time_sec_ > config_.max_gap_seconds;
        if (gap_expired) {
            const bool replay = current_phase_ != SpatialStaticPhase::suppressed;
            clear_observations();
            decision.phase = SpatialStaticPhase::observed;
            decision.replay_shadow = replay;
            decision.reason = replay ? "preconfirm_disappearance" : "suppressed_disappearance";
            return decision;
        }

        const bool phone_moving = short_analysis.recent_movement;
        if (short_analysis.metrics.follows_wrist || phone_moving) {
            current_phase_ = SpatialStaticPhase::handheld_or_moving;
            decision.phase = current_phase_;
            decision.replay_shadow = true;
            decision.reason = short_analysis.metrics.follows_wrist ? "follows_wrist" : "phone_moving";
            return decision;
        }

        FixedTemplateEvidence fixed_template_evidence = FixedTemplateEvidence::none;
        if (short_analysis.primary_candidate_index.has_value() &&
            *short_analysis.primary_candidate_index < current_frame.phones.size()) {
            fixed_template_evidence =
                current_frame.phones[*short_analysis.primary_candidate_index]
                    .fixed_template_evidence;
        }
        if (fixed_template_evidence == FixedTemplateEvidence::mismatch) {
            clear_observations();
            current_phase_ = SpatialStaticPhase::handheld_or_moving;
            decision.phase = current_phase_;
            decision.replay_shadow = true;
            decision.reason = "fixed_template_mismatch";
            return decision;
        }
        if (config_.manual_templates_only &&
            fixed_template_evidence != FixedTemplateEvidence::matched) {
            clear_observations();
            current_phase_ = SpatialStaticPhase::handheld_or_moving;
            decision.phase = current_phase_;
            decision.replay_shadow = true;
            decision.reason = "manual_template_required";
            return decision;
        }
        if (fixed_template_evidence == FixedTemplateEvidence::matched) {
            hotspot_score = std::max(hotspot_score, 4.0f);
        }

        const bool known_hotspot_stable =
            hotspot_score >= config_.hotspot_early_hold_score &&
            short_analysis.consecutive_detection_count >= config_.hotspot_early_hold_samples &&
            short_analysis.primary_candidate_index.has_value() &&
            short_analysis.metrics.center_spread_px <= short_analysis.radius_px &&
            short_analysis.metrics.bbox_iou_median + 1e-6f >= config_.min_bbox_iou;
        if (known_hotspot_stable) {
            current_phase_ = SpatialStaticPhase::pending;
            decision.phase = current_phase_;
            decision.hold_candidate = true;
            decision.reason = "known_hotspot_pending";
        }

        const bool short_complete = short_analysis.coverage_seconds + kTimeEpsilon >= config_.short_seconds;
        const bool long_complete = long_analysis.coverage_seconds + kTimeEpsilon >= config_.long_seconds;
        const bool short_context = short_analysis.metrics.motion_decoupled || hotspot_score >= 1.0f;
        const bool long_context = long_analysis.metrics.motion_decoupled;
        const bool short_suppressed = short_complete && short_analysis.stable && short_context;
        const bool long_suppressed = long_complete && long_analysis.stable && long_context;
        if (short_suppressed || long_suppressed) {
            current_phase_ = SpatialStaticPhase::suppressed;
            decision.phase = current_phase_;
            decision.primary_candidate_index = short_suppressed
                ? short_analysis.primary_candidate_index
                : long_analysis.primary_candidate_index;
            decision.primary_cluster_centers = short_suppressed
                ? short_analysis.primary_cluster_centers
                : long_analysis.primary_cluster_centers;
            decision.primary_cluster_tolerance_px = short_suppressed
                ? short_analysis.primary_cluster_tolerance_px
                : long_analysis.primary_cluster_tolerance_px;
            decision.discard_shadow = true;
            decision.reason = hotspot_score >= 1.0f ? "hotspot" : "motion_decoupled";
            return decision;
        }

        if (known_hotspot_stable) return decision;

        if (short_complete) {
            current_phase_ = SpatialStaticPhase::handheld_or_moving;
            decision.phase = current_phase_;
            decision.replay_shadow = true;
            decision.reason = "static_context_timeout";
            return decision;
        }

        if (short_analysis.stable &&
            short_analysis.coverage_seconds + kTimeEpsilon >= config_.pending_seconds) {
            current_phase_ = SpatialStaticPhase::pending;
            decision.phase = current_phase_;
            decision.hold_candidate = true;
            decision.reason = "pending_static_context";
            return decision;
        }

        if (current_phase_ == SpatialStaticPhase::pending && has_detection_ &&
            current_frame.time_sec - last_detection_time_sec_ <= config_.max_gap_seconds) {
            decision.phase = current_phase_;
            decision.hold_candidate = true;
            decision.reason = "pending_dropout";
            return decision;
        }

        current_phase_ = SpatialStaticPhase::observed;
        decision.phase = current_phase_;
        decision.reason = "observed";
        return decision;
    }

    void append_shadow(ShadowRiskSample sample) {
        if (!std::isfinite(sample.time_sec) || !std::isfinite(sample.risk_score)) return;
        shadow_.push_back(sample);
    }

    std::vector<ShadowRiskSample> take_replay() {
        std::vector<ShadowRiskSample> replay;
        replay.swap(shadow_);
        return replay;
    }

    void discard_shadow() {
        shadow_.clear();
    }

    void reset_short_state() {
        clear_observations();
        shadow_.clear();
    }

private:
    struct CandidateSample {
        std::size_t frame_index = 0;
        std::size_t candidate_index = 0;
        double time_sec = 0.0;
        float center_x = 0.0f;
        float center_y = 0.0f;
        float width = 0.0f;
        float height = 0.0f;
        float confidence = 0.0f;
        const SpatialFrameObservation* frame = nullptr;
    };

    struct Cluster {
        std::vector<CandidateSample> samples;
    };

    struct Analysis {
        SpatialStaticMetrics metrics{};
        std::optional<std::size_t> primary_candidate_index;
        std::vector<SpatialPoint> primary_cluster_centers;
        float primary_cluster_tolerance_px = 0.0f;
        float radius_px = 0.0f;
        float phone_motion_px = 0.0f;
        double coverage_seconds = 0.0;
        int consecutive_detection_count = 0;
        bool stable = false;
        bool recent_movement = false;
    };

    static constexpr double kTimeEpsilon = 1e-6;

    static SpatialStaticConfig normalize_config(SpatialStaticConfig config) {
        const SpatialStaticConfig defaults;
        if (!std::isfinite(config.short_seconds) || config.short_seconds <= 0.0) {
            config.short_seconds = defaults.short_seconds;
        }
        if (!std::isfinite(config.long_seconds) || config.long_seconds < config.short_seconds) {
            config.long_seconds = std::max(defaults.long_seconds, config.short_seconds);
        }
        if (!std::isfinite(config.pending_seconds) || config.pending_seconds <= 0.0 ||
            config.pending_seconds > config.short_seconds) {
            config.pending_seconds = std::min(defaults.pending_seconds, config.short_seconds);
        }
        if (!std::isfinite(config.max_gap_seconds) || config.max_gap_seconds <= 0.0) {
            config.max_gap_seconds = defaults.max_gap_seconds;
        }
        if (!std::isfinite(config.min_detection_ratio) || config.min_detection_ratio < 0.0f ||
            config.min_detection_ratio > 1.0f) {
            config.min_detection_ratio = defaults.min_detection_ratio;
        }
        if (!std::isfinite(config.radius_ratio) || config.radius_ratio < 0.0f ||
            config.radius_ratio > 1.0f) {
            config.radius_ratio = defaults.radius_ratio;
        }
        if (!std::isfinite(config.min_radius_px) || config.min_radius_px <= 0.0f) {
            config.min_radius_px = defaults.min_radius_px;
        }
        if (!std::isfinite(config.min_bbox_iou) || config.min_bbox_iou < 0.0f ||
            config.min_bbox_iou > 1.0f) {
            config.min_bbox_iou = defaults.min_bbox_iou;
        }
        if (!std::isfinite(config.max_bbox_size_change) || config.max_bbox_size_change < 0.0f ||
            config.max_bbox_size_change > 1.0f) {
            config.max_bbox_size_change = defaults.max_bbox_size_change;
        }
        if (!std::isfinite(config.wrist_follow_cosine) || config.wrist_follow_cosine < -1.0f ||
            config.wrist_follow_cosine > 1.0f) {
            config.wrist_follow_cosine = defaults.wrist_follow_cosine;
        }
        if (!std::isfinite(config.hotspot_early_hold_score) ||
            config.hotspot_early_hold_score < 0.0f) {
            config.hotspot_early_hold_score = defaults.hotspot_early_hold_score;
        }
        if (config.hotspot_early_hold_samples < 2) {
            config.hotspot_early_hold_samples = defaults.hotspot_early_hold_samples;
        }
        return config;
    }

    static bool valid_rect(const SpatialRect& box) {
        return std::isfinite(box.x1) && std::isfinite(box.y1) &&
            std::isfinite(box.x2) && std::isfinite(box.y2) &&
            box.x2 > box.x1 && box.y2 > box.y1;
    }

    static float center_x(const SpatialRect& box) {
        return (box.x1 + box.x2) * 0.5f;
    }

    static float center_y(const SpatialRect& box) {
        return (box.y1 + box.y2) * 0.5f;
    }

    static float distance(float x1, float y1, float x2, float y2) {
        return std::hypot(x2 - x1, y2 - y1);
    }

    static float median(std::vector<float> values) {
        if (values.empty()) return 0.0f;
        const std::size_t middle = values.size() / 2;
        std::nth_element(values.begin(), values.begin() + middle, values.end());
        const float upper = values[middle];
        if (values.size() % 2 != 0) return upper;
        std::nth_element(values.begin(), values.begin() + middle - 1, values.begin() + middle);
        return (values[middle - 1] + upper) * 0.5f;
    }

    static float rect_iou(const CandidateSample& sample, float median_x, float median_y,
                          float median_width, float median_height) {
        const float ax1 = sample.center_x - sample.width * 0.5f;
        const float ay1 = sample.center_y - sample.height * 0.5f;
        const float ax2 = sample.center_x + sample.width * 0.5f;
        const float ay2 = sample.center_y + sample.height * 0.5f;
        const float bx1 = median_x - median_width * 0.5f;
        const float by1 = median_y - median_height * 0.5f;
        const float bx2 = median_x + median_width * 0.5f;
        const float by2 = median_y + median_height * 0.5f;
        const float intersection_width = std::max(0.0f, std::min(ax2, bx2) - std::max(ax1, bx1));
        const float intersection_height = std::max(0.0f, std::min(ay2, by2) - std::max(ay1, by1));
        const float intersection = intersection_width * intersection_height;
        const float union_area = sample.width * sample.height + median_width * median_height - intersection;
        return union_area > 0.0f ? intersection / union_area : 0.0f;
    }

    static float motion_extent(const std::vector<std::pair<float, float>>& points) {
        float maximum = 0.0f;
        for (std::size_t i = 0; i < points.size(); ++i) {
            for (std::size_t j = i + 1; j < points.size(); ++j) {
                maximum = std::max(maximum, distance(points[i].first, points[i].second,
                                                     points[j].first, points[j].second));
            }
        }
        return maximum;
    }

    Analysis analyze(double now_sec, double window_seconds) const {
        Analysis analysis;
        std::vector<const SpatialFrameObservation*> frames;
        const double cutoff = now_sec - window_seconds;
        for (const SpatialFrameObservation& observation : observations_) {
            if (observation.time_sec + kTimeEpsilon >= cutoff) frames.push_back(&observation);
        }
        if (frames.empty()) return analysis;

        std::vector<float> person_heights;
        std::vector<std::pair<float, float>> person_centers;
        std::vector<std::pair<float, float>> wrist_points;
        std::vector<CandidateSample> candidates;
        for (std::size_t frame_index = 0; frame_index < frames.size(); ++frame_index) {
            const SpatialFrameObservation& frame = *frames[frame_index];
            if (valid_rect(frame.person)) {
                person_heights.push_back(frame.person.y2 - frame.person.y1);
                person_centers.emplace_back(center_x(frame.person), center_y(frame.person));
            }
            if (frame.wrist_valid) wrist_points.emplace_back(frame.wrist_x, frame.wrist_y);
            for (std::size_t candidate_index = 0;
                 candidate_index < frame.phones.size(); ++candidate_index) {
                const SpatialPhoneCandidate& phone = frame.phones[candidate_index];
                if (!valid_rect(phone.box)) continue;
                CandidateSample sample;
                sample.frame_index = frame_index;
                sample.candidate_index = candidate_index;
                sample.time_sec = frame.time_sec;
                sample.center_x = center_x(phone.box);
                sample.center_y = center_y(phone.box);
                sample.width = std::max(0.0f, phone.box.x2 - phone.box.x1);
                sample.height = std::max(0.0f, phone.box.y2 - phone.box.y1);
                sample.confidence = phone.confidence;
                sample.frame = &frame;
                candidates.push_back(sample);
            }
        }

        analysis.radius_px = std::max(config_.min_radius_px,
            config_.radius_ratio * median(std::move(person_heights)));
        analysis.metrics.person_motion_px = motion_extent(person_centers);
        analysis.metrics.wrist_motion_px = motion_extent(wrist_points);
        if (candidates.empty()) return analysis;

        std::vector<Cluster> clusters;
        for (const CandidateSample& candidate : candidates) {
            Cluster* closest = nullptr;
            float closest_distance = std::numeric_limits<float>::max();
            for (Cluster& cluster : clusters) {
                float candidate_distance = std::numeric_limits<float>::max();
                for (const CandidateSample& member : cluster.samples) {
                    candidate_distance = std::min(candidate_distance,
                        distance(candidate.center_x, candidate.center_y,
                                 member.center_x, member.center_y));
                }
                if (candidate_distance <= analysis.radius_px * 2.0f &&
                    candidate_distance < closest_distance) {
                    closest = &cluster;
                    closest_distance = candidate_distance;
                }
            }
            if (closest == nullptr) {
                clusters.push_back(Cluster{});
                closest = &clusters.back();
            }
            closest->samples.push_back(candidate);
        }

        const auto frame_count = [](const Cluster& cluster) {
            std::vector<std::size_t> frame_indices;
            frame_indices.reserve(cluster.samples.size());
            for (const CandidateSample& sample : cluster.samples) {
                frame_indices.push_back(sample.frame_index);
            }
            std::sort(frame_indices.begin(), frame_indices.end());
            return static_cast<std::size_t>(
                std::unique(frame_indices.begin(), frame_indices.end()) - frame_indices.begin());
        };
        const auto frame_confidence_sum = [](const Cluster& cluster) {
            std::vector<std::pair<std::size_t, float>> confidences;
            for (const CandidateSample& sample : cluster.samples) {
                auto it = std::find_if(confidences.begin(), confidences.end(),
                    [&](const auto& value) { return value.first == sample.frame_index; });
                if (it == confidences.end()) {
                    confidences.emplace_back(sample.frame_index, sample.confidence);
                } else {
                    it->second = std::max(it->second, sample.confidence);
                }
            }
            float sum = 0.0f;
            for (const auto& value : confidences) sum += value.second;
            return sum;
        };

        const Cluster* primary = &clusters.front();
        for (const Cluster& cluster : clusters) {
            const std::size_t cluster_frames = frame_count(cluster);
            const std::size_t primary_frames = frame_count(*primary);
            if (cluster_frames > primary_frames ||
                (cluster_frames == primary_frames &&
                 frame_confidence_sum(cluster) > frame_confidence_sum(*primary))) {
                primary = &cluster;
            }
        }

        std::vector<const CandidateSample*> representatives(frames.size(), nullptr);
        for (const CandidateSample& sample : primary->samples) {
            const CandidateSample*& representative = representatives[sample.frame_index];
            if (representative == nullptr || sample.confidence > representative->confidence) {
                representative = &sample;
            }
        }
        const CandidateSample* current_representative = representatives.back();
        if (current_representative != nullptr) {
            analysis.primary_candidate_index = current_representative->candidate_index;
        }

        analysis.primary_cluster_tolerance_px = analysis.radius_px * 2.0f;
        analysis.primary_cluster_centers.reserve(primary->samples.size());
        for (const CandidateSample& sample : primary->samples) {
            analysis.primary_cluster_centers.push_back(
                SpatialPoint{sample.center_x, sample.center_y});
        }

        std::vector<bool> detected_frames(frames.size(), false);
        std::vector<float> xs;
        std::vector<float> ys;
        std::vector<float> widths;
        std::vector<float> heights;
        std::vector<double> detection_times;
        xs.reserve(primary->samples.size());
        ys.reserve(primary->samples.size());
        widths.reserve(primary->samples.size());
        heights.reserve(primary->samples.size());
        detection_times.reserve(primary->samples.size());
        for (const CandidateSample* representative : representatives) {
            if (representative == nullptr) continue;
            const CandidateSample& sample = *representative;
            detected_frames[sample.frame_index] = true;
            xs.push_back(sample.center_x);
            ys.push_back(sample.center_y);
            widths.push_back(sample.width);
            heights.push_back(sample.height);
            detection_times.push_back(sample.time_sec);
        }
        const std::size_t detected_count = static_cast<std::size_t>(
            std::count(detected_frames.begin(), detected_frames.end(), true));
        analysis.metrics.sample_count = static_cast<int>(detected_count);
        analysis.metrics.detection_ratio = static_cast<float>(detected_count) /
            static_cast<float>(frames.size());

        const float median_x = median(xs);
        const float median_y = median(ys);
        const float median_width = median(widths);
        const float median_height = median(heights);
        std::vector<float> ious;
        float maximum_size_change = 0.0f;
        for (const CandidateSample* representative : representatives) {
            if (representative == nullptr) continue;
            const CandidateSample& sample = *representative;
            analysis.metrics.center_spread_px = std::max(analysis.metrics.center_spread_px,
                distance(sample.center_x, sample.center_y, median_x, median_y));
            ious.push_back(rect_iou(sample, median_x, median_y, median_width, median_height));
            if (median_width > 0.0f) {
                maximum_size_change = std::max(maximum_size_change,
                    std::abs(sample.width - median_width) / median_width);
            }
            if (median_height > 0.0f) {
                maximum_size_change = std::max(maximum_size_change,
                    std::abs(sample.height - median_height) / median_height);
            }
        }
        analysis.metrics.bbox_iou_median = median(std::move(ious));
        analysis.phone_motion_px = motion_extent([&]() {
            std::vector<std::pair<float, float>> centers;
            centers.reserve(detected_count);
            for (const CandidateSample* representative : representatives) {
                if (representative == nullptr) continue;
                centers.emplace_back(representative->center_x, representative->center_y);
            }
            return centers;
        }());

        std::sort(detection_times.begin(), detection_times.end());
        detection_times.erase(std::unique(detection_times.begin(), detection_times.end()), detection_times.end());
        analysis.coverage_seconds = detection_times.back() - detection_times.front();
        double maximum_gap = 0.0;
        for (std::size_t i = 1; i < detection_times.size(); ++i) {
            maximum_gap = std::max(maximum_gap, detection_times[i] - detection_times[i - 1]);
        }
        maximum_gap = std::max(maximum_gap, now_sec - detection_times.back());

        analysis.stable = analysis.metrics.detection_ratio + 1e-6f >= config_.min_detection_ratio &&
            analysis.metrics.center_spread_px <= analysis.radius_px &&
            analysis.metrics.bbox_iou_median + 1e-6f >= config_.min_bbox_iou &&
            maximum_size_change <= config_.max_bbox_size_change &&
            maximum_gap <= config_.max_gap_seconds + kTimeEpsilon;

        std::vector<const CandidateSample*> consecutive_tail;
        for (std::size_t index = representatives.size(); index > 0; --index) {
            const CandidateSample* representative = representatives[index - 1];
            if (representative == nullptr) break;
            consecutive_tail.push_back(representative);
        }
        std::reverse(consecutive_tail.begin(), consecutive_tail.end());
        analysis.consecutive_detection_count = static_cast<int>(consecutive_tail.size());
        std::size_t motion_start = consecutive_tail.size();
        if (consecutive_tail.size() >= 3) {
            motion_start = consecutive_tail.size() - 1;
            float next_dx = 0.0f;
            float next_dy = 0.0f;
            float next_distance = 0.0f;
            for (std::size_t i = consecutive_tail.size() - 1; i > 0; --i) {
                const float dx = consecutive_tail[i]->center_x - consecutive_tail[i - 1]->center_x;
                const float dy = consecutive_tail[i]->center_y - consecutive_tail[i - 1]->center_y;
                const float step_distance = std::hypot(dx, dy);
                if (step_distance < analysis.radius_px * 0.1f) break;
                if (next_distance > 0.0f) {
                    const float cosine = (dx * next_dx + dy * next_dy) /
                        (step_distance * next_distance);
                    if (cosine < 0.5f) break;
                }
                motion_start = i - 1;
                next_dx = dx;
                next_dy = dy;
                next_distance = step_distance;
            }
            if (consecutive_tail.size() - motion_start >= 3) {
                const CandidateSample* first = consecutive_tail[motion_start];
                const CandidateSample* last = consecutive_tail.back();
                analysis.recent_movement = distance(first->center_x, first->center_y,
                                                     last->center_x, last->center_y) >=
                    analysis.radius_px;
            }
        }

        if (analysis.recent_movement) {
            const CandidateSample* first_wrist_sample = consecutive_tail[motion_start];
            const CandidateSample* last_wrist_sample = consecutive_tail.back();
            bool wrist_sequence_valid = true;
            for (std::size_t i = motion_start; i < consecutive_tail.size(); ++i) {
                wrist_sequence_valid = wrist_sequence_valid && consecutive_tail[i]->frame->wrist_valid;
            }
            const float phone_dx = last_wrist_sample->center_x - first_wrist_sample->center_x;
            const float phone_dy = last_wrist_sample->center_y - first_wrist_sample->center_y;
            const float wrist_dx = last_wrist_sample->frame->wrist_x - first_wrist_sample->frame->wrist_x;
            const float wrist_dy = last_wrist_sample->frame->wrist_y - first_wrist_sample->frame->wrist_y;
            const float phone_distance = std::hypot(phone_dx, phone_dy);
            const float wrist_distance = std::hypot(wrist_dx, wrist_dy);
            if (wrist_sequence_valid && phone_distance >= analysis.radius_px &&
                wrist_distance >= analysis.radius_px * 0.5f) {
                const float cosine = (phone_dx * wrist_dx + phone_dy * wrist_dy) /
                    (phone_distance * wrist_distance);
                const float motion_ratio = phone_distance / wrist_distance;
                analysis.metrics.follows_wrist = cosine >= config_.wrist_follow_cosine &&
                    motion_ratio >= 0.25f && motion_ratio <= 4.0f;
            }
        }

        const float context_motion = std::max(analysis.metrics.person_motion_px,
                                               analysis.metrics.wrist_motion_px);
        analysis.metrics.motion_decoupled = analysis.stable &&
            context_motion >= analysis.radius_px && analysis.phone_motion_px < analysis.radius_px;
        return analysis;
    }

    void prune(double now_sec) {
        const double cutoff = now_sec - config_.long_seconds;
        while (!observations_.empty() && observations_.front().time_sec + kTimeEpsilon < cutoff) {
            observations_.pop_front();
        }
    }

    void clear_observations() {
        observations_.clear();
        current_phase_ = SpatialStaticPhase::observed;
        has_detection_ = false;
        last_detection_time_sec_ = 0.0;
        has_last_time_ = false;
        last_time_sec_ = 0.0;
    }

    SpatialStaticConfig config_;
    std::deque<SpatialFrameObservation> observations_;
    std::vector<ShadowRiskSample> shadow_;
    SpatialStaticPhase current_phase_ = SpatialStaticPhase::observed;
    bool has_detection_ = false;
    double last_detection_time_sec_ = 0.0;
    bool has_last_time_ = false;
    double last_time_sec_ = 0.0;
};

}  // namespace jiankong::custom_pipeline
