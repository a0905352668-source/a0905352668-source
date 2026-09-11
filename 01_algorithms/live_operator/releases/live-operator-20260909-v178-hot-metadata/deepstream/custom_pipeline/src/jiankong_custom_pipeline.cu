#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include "deepstream_batch_reader.hpp"
#include "bounded_output_writer.hpp"
#include "gated_event_policy.hpp"
#include "live_calibration_policy.hpp"
#include "live_pipeline_policy.hpp"
#include "measurement_window.hpp"
#include "rgba_letterbox_cuda.hpp"
#include "rtsp_live_runtime.hpp"
#include "static_phone_hotspot.hpp"
#include "static_phone_spatial_policy.hpp"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <condition_variable>
#include <csignal>
#include <ctime>
#include <deque>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <mutex>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
using json = nlohmann::json;

struct Logger final : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cerr << "[TRT] " << msg << std::endl;
        }
    }
};

static Logger g_logger;
static volatile std::sig_atomic_t output_stop_requested = 0;
static void request_output_stop(int) { output_stop_requested = 1; }

static double now_sec() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

static double unix_now_sec() {
    using clock = std::chrono::system_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

struct LiveWallClockAnchor {
    bool initialized = false;
    std::uint64_t anchor_pts_ns = 0;
    std::uint64_t last_pts_ns = 0;
    double anchor_wall_unix_seconds = 0.0;
};

static constexpr double kLiveWallClockMaxDriftSeconds = 1.0;

static double live_frame_capture_unix_seconds(
    LiveWallClockAnchor& anchor,
    const jiankong::custom_pipeline::DeviceFrameView& frame) {
    if (!frame.pts_valid) {
        anchor.initialized = false;
        return frame.received_at_unix_seconds;
    }
    if (!anchor.initialized || frame.pts_ns < anchor.last_pts_ns) {
        anchor.initialized = true;
        anchor.anchor_pts_ns = frame.pts_ns;
        anchor.anchor_wall_unix_seconds = frame.received_at_unix_seconds;
    }
    double projected_unix_seconds = anchor.anchor_wall_unix_seconds +
        static_cast<double>(frame.pts_ns - anchor.anchor_pts_ns) / 1e9;
    if (std::abs(projected_unix_seconds - frame.received_at_unix_seconds) >
        kLiveWallClockMaxDriftSeconds) {
        anchor.anchor_pts_ns = frame.pts_ns;
        anchor.anchor_wall_unix_seconds = frame.received_at_unix_seconds;
        projected_unix_seconds = frame.received_at_unix_seconds;
    }
    anchor.last_pts_ns = frame.pts_ns;
    return projected_unix_seconds;
}

static std::string format_utc_timestamp(double unix_seconds) {
    const double whole_seconds = std::floor(unix_seconds);
    const std::time_t value = static_cast<std::time_t>(whole_seconds);
    std::tm utc{};
#ifdef _WIN32
    if (gmtime_s(&utc, &value) != 0) throw std::runtime_error("failed to format UTC timestamp");
#else
    if (gmtime_r(&value, &utc) == nullptr) throw std::runtime_error("failed to format UTC timestamp");
#endif
    const int milliseconds = static_cast<int>((unix_seconds - whole_seconds) * 1000.0);
    std::ostringstream formatted;
    formatted << std::put_time(&utc, "%Y-%m-%dT%H:%M:%S") << '.'
              << std::setfill('0') << std::setw(3) << milliseconds << 'Z';
    return formatted.str();
}

static void check_cuda(cudaError_t err, const std::string& where) {
    if (err != cudaSuccess) {
        throw std::runtime_error(where + ": " + cudaGetErrorString(err));
    }
}

static size_t volume(const nvinfer1::Dims& d) {
    size_t v = 1;
    for (int i = 0; i < d.nbDims; ++i) {
        if (d.d[i] <= 0) {
            throw std::runtime_error("non-concrete TensorRT dims");
        }
        v *= static_cast<size_t>(d.d[i]);
    }
    return v;
}

static bool has_dynamic_dim(const nvinfer1::Dims& d) {
    for (int i = 0; i < d.nbDims; ++i) {
        if (d.d[i] <= 0) return true;
    }
    return false;
}

struct Rect {
    float x1 = 0, y1 = 0, x2 = 0, y2 = 0;
    float w() const { return std::max(0.0f, x2 - x1); }
    float h() const { return std::max(0.0f, y2 - y1); }
    float area() const { return w() * h(); }
};

static Rect clamp_rect(Rect r, int width, int height) {
    r.x1 = std::max(0.0f, std::min(static_cast<float>(width - 1), r.x1));
    r.y1 = std::max(0.0f, std::min(static_cast<float>(height - 1), r.y1));
    r.x2 = std::max(0.0f, std::min(static_cast<float>(width - 1), r.x2));
    r.y2 = std::max(0.0f, std::min(static_cast<float>(height - 1), r.y2));
    return r;
}

static float iou(const Rect& a, const Rect& b) {
    const float x1 = std::max(a.x1, b.x1);
    const float y1 = std::max(a.y1, b.y1);
    const float x2 = std::min(a.x2, b.x2);
    const float y2 = std::min(a.y2, b.y2);
    const float iw = std::max(0.0f, x2 - x1);
    const float ih = std::max(0.0f, y2 - y1);
    const float inter = iw * ih;
    return inter / std::max(1.0f, a.area() + b.area() - inter);
}

static float intersection_area(const Rect& a, const Rect& b) {
    const float x1 = std::max(a.x1, b.x1);
    const float y1 = std::max(a.y1, b.y1);
    const float x2 = std::min(a.x2, b.x2);
    const float y2 = std::min(a.y2, b.y2);
    return std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
}

static float containment_ratio(const Rect& a, const Rect& b) {
    return intersection_area(a, b) / std::max(1.0f, std::min(a.area(), b.area()));
}

static float rect_diag(const Rect& r) {
    return std::sqrt(r.w() * r.w() + r.h() * r.h());
}

static float center_dist(const Rect& a, const Rect& b) {
    const float acx = (a.x1 + a.x2) * 0.5f;
    const float acy = (a.y1 + a.y2) * 0.5f;
    const float bcx = (b.x1 + b.x2) * 0.5f;
    const float bcy = (b.y1 + b.y2) * 0.5f;
    const float dx = acx - bcx;
    const float dy = acy - bcy;
    return std::sqrt(dx * dx + dy * dy);
}

static bool point_in_rect(float x, float y, const Rect& r) {
    return x >= r.x1 && x <= r.x2 && y >= r.y1 && y <= r.y2;
}

static float clampf(float v, float lo, float hi) {
    return std::max(lo, std::min(hi, v));
}

static float rect_cx(const Rect& r) { return (r.x1 + r.x2) * 0.5f; }
static float rect_cy(const Rect& r) { return (r.y1 + r.y2) * 0.5f; }

static float point_dist(const cv::Point2f& a, const cv::Point2f& b) {
    const float dx = a.x - b.x;
    const float dy = a.y - b.y;
    return std::sqrt(dx * dx + dy * dy);
}

static float point_to_rect_distance(const cv::Point2f& p, const Rect& r) {
    const float dx = std::max(std::max(r.x1 - p.x, 0.0f), p.x - r.x2);
    const float dy = std::max(std::max(r.y1 - p.y, 0.0f), p.y - r.y2);
    return std::sqrt(dx * dx + dy * dy);
}

static float point_segment_distance(const cv::Point2f& p, const cv::Point2f& a, const cv::Point2f& b) {
    const float vx = b.x - a.x;
    const float vy = b.y - a.y;
    const float len2 = vx * vx + vy * vy;
    if (len2 <= 1e-6f) return point_dist(p, a);
    const float t = clampf(((p.x - a.x) * vx + (p.y - a.y) * vy) / len2, 0.0f, 1.0f);
    return point_dist(p, cv::Point2f(a.x + t * vx, a.y + t * vy));
}

static bool point_in_polygon(const cv::Point2f& p, const std::vector<cv::Point2f>& poly) {
    if (poly.size() < 3) return false;
    return cv::pointPolygonTest(poly, p, false) >= 0.0;
}

static Rect bbox_from_polygon(const std::vector<cv::Point2f>& poly) {
    if (poly.empty()) return Rect{};
    Rect r{poly[0].x, poly[0].y, poly[0].x, poly[0].y};
    for (const auto& p : poly) {
        r.x1 = std::min(r.x1, p.x);
        r.y1 = std::min(r.y1, p.y);
        r.x2 = std::max(r.x2, p.x);
        r.y2 = std::max(r.y2, p.y);
    }
    return r;
}

static float box_to_poly_distance(const Rect& box, const std::vector<cv::Point2f>& poly) {
    if (poly.empty()) return std::numeric_limits<float>::max();
    const Rect pb = bbox_from_polygon(poly);
    if (intersection_area(box, pb) > 0.0f) return 0.0f;
    const std::vector<cv::Point2f> corners = {
        {box.x1, box.y1}, {box.x2, box.y1}, {box.x2, box.y2}, {box.x1, box.y2}
    };
    float best = std::numeric_limits<float>::max();
    for (const auto& c : corners) {
        if (point_in_polygon(c, poly)) return 0.0f;
        for (size_t i = 0; i < poly.size(); ++i) {
            best = std::min(best, point_segment_distance(c, poly[i], poly[(i + 1) % poly.size()]));
        }
    }
    return best;
}

static bool boxes_intersect(const Rect& a, const Rect& b) {
    return intersection_area(a, b) > 0.0f;
}

static float angle_deg(const cv::Point2f& v1, const cv::Point2f& v2) {
    const float n1 = std::sqrt(v1.x * v1.x + v1.y * v1.y);
    const float n2 = std::sqrt(v2.x * v2.x + v2.y * v2.y);
    if (n1 < 1e-6f || n2 < 1e-6f) return 180.0f;
    const float cosv = clampf((v1.x * v2.x + v1.y * v2.y) / (n1 * n2), -1.0f, 1.0f);
    return std::acos(cosv) * 180.0f / static_cast<float>(CV_PI);
}

static bool ray_hits_box(const cv::Point2f& start, const cv::Point2f& direction, const Rect& rect, float max_t = 4.0f) {
    if (std::sqrt(direction.x * direction.x + direction.y * direction.y) < 1e-6f) return false;
    float tmin = 0.0f;
    float tmax = max_t;
    const float s[2] = {start.x, start.y};
    const float d[2] = {direction.x, direction.y};
    const float lo[2] = {rect.x1, rect.y1};
    const float hi[2] = {rect.x2, rect.y2};
    for (int i = 0; i < 2; ++i) {
        if (std::abs(d[i]) < 1e-6f) {
            if (s[i] < lo[i] || s[i] > hi[i]) return false;
            continue;
        }
        float t1 = (lo[i] - s[i]) / d[i];
        float t2 = (hi[i] - s[i]) / d[i];
        if (t1 > t2) std::swap(t1, t2);
        tmin = std::max(tmin, t1);
        tmax = std::min(tmax, t2);
        if (tmax < tmin) return false;
    }
    return tmax >= std::max(0.0f, tmin);
}

struct LetterboxMeta {
    float scale = 1.0f;
    int pad_x = 0;
    int pad_y = 0;
    int in_w = 0;
    int in_h = 0;
};

static LetterboxMeta compute_letterbox_meta(int src_w, int src_h, int size) {
    LetterboxMeta meta;
    meta.in_w = src_w;
    meta.in_h = src_h;
    const float r = std::min(size / static_cast<float>(src_w), size / static_cast<float>(src_h));
    const int new_w = std::max(1, static_cast<int>(std::round(src_w * r)));
    const int new_h = std::max(1, static_cast<int>(std::round(src_h * r)));
    meta.scale = r;
    meta.pad_x = (size - new_w) / 2;
    meta.pad_y = (size - new_h) / 2;
    return meta;
}

struct Det {
    Rect box;
    float conf = 0;
    int stream = -1;
    int roi_index = -1;
    int person_index = -1;
    int track_id = -1;
    float kpts[17][3]{};
};

struct RoiJob {
    int stream = -1;
    Rect roi;
    int person_index = -1;
    std::string screen_id;
    std::string source = "person_roi";
};

struct Zone {
    std::string name;
    std::vector<cv::Point2f> polygon;
    float weight = 1.0f;
};

struct ScreenParams {
    float person_expand_x = 0.35f;
    float person_expand_y = 0.20f;
    int near_zone_detect_interval = 10;
    float phone_valid_conf_person_roi = 0.35f;
    float phone_valid_conf_near_zone = 0.50f;
    float phone_valid_conf_floor = 0.45f;
    float alarm_raw_phone_confidence = 0.75f;
    float phone_max_person_area_ratio = 0.05f;
    float phone_large_person_diag_ratio = 0.18f;
    float phone_large_confidence = 0.85f;
    float screen_distance_min_px = 70.0f;
    float screen_distance_max_px = 180.0f;
    float angle_thresh = 90.0f;
    float angle_relaxed_thresh = 110.0f;
    float hand_radius_ratio = 0.18f;
    float hand_radius_min = 40.0f;
    float corridor_width_ratio = 0.35f;
    float corridor_width_min = 80.0f;
    int person_state_window = 30;
    int person_state_min_hits = 16;
    int handheld_suspect_min_hits = 12;
    float person_state_risk_threshold = 0.65f;
    bool enable_static_phone_suppression = true;
    float static_phone_window_seconds = 1.5f;
    float static_phone_max_disp_ratio = 0.03f;
    float static_phone_risk_multiplier = 0.2f;
    float static_phone_min_abs_disp_px = 6.0f;
    float static_phone_min_bbox_iou = 0.45f;
    float static_phone_max_bbox_size_change_ratio = 0.35f;
    float static_phone_lower_person_start_ratio = 0.55f;
    float static_phone_wrist_follow_min_motion_px = 10.0f;
    float static_phone_wrist_follow_cosine = 0.65f;
};

struct ScreenConfig {
    std::string screen_id;
    std::vector<cv::Point2f> screen_poly;
    std::vector<cv::Point2f> near_zone;
    std::vector<Zone> danger_zones;
    std::vector<Zone> ignore_zones;
    std::vector<Zone> desk_static_zones;
    ScreenParams params;
};

struct CandidateEval {
    Det phone;
    int person_index = -1;
    int track_id = -1;
    std::string screen_id;
    std::string zone_reason = "outside";
    std::string reject_reason;
    std::string level = "ignore";
    Rect person_roi;
    Rect corridor_bbox;
    float static_zone_score = 0.0f;
    float person_match_score = 0.0f;
    float phone_score = 0.0f;
    float phone_person_area_ratio = 0.0f;
    float phone_person_diag_ratio = 0.0f;
    bool phone_geometry_valid = true;
    float phone_hand_score = 0.0f;
    float screen_relation_score = 0.0f;
    float pose_score = 0.0f;
    float temporal_score = 0.0f;
    float risk_score = 0.0f;
    float best_angle = 180.0f;
    bool best_ray_hit = false;
    bool phone_static = false;
    float phone_static_duration = 0.0f;
    float phone_motion_px = 0.0f;
    bool phone_in_desk_zone = false;
    bool static_suppressed = false;
    float static_risk_multiplier = 1.0f;
    bool phone_follow_wrist = false;
    bool static_suppression_enabled = true;
    float static_window_seconds = 1.5f;
    float static_max_disp_ratio = 0.03f;
    float static_config_risk_multiplier = 0.2f;
    float static_min_abs_disp_px = 6.0f;
    float static_min_bbox_iou = 0.45f;
    float static_max_bbox_size_change_ratio = 0.35f;
    float static_lower_person_start_ratio = 0.55f;
    float static_wrist_follow_min_motion_px = 10.0f;
    float static_wrist_follow_cosine = 0.65f;
    bool person_alarm = false;
    int person_window_hits = 0;
    std::string candidate_reason;
    int static_cluster_samples = 0;
    float static_detection_ratio = 0.0f;
    float static_center_spread_px = 0.0f;
    float static_bbox_iou_median = 0.0f;
    bool static_pending = false;
    double static_pending_duration = 0.0;
    float static_hotspot_score = 0.0f;
    std::string static_context_reason = "disabled";
    int static_shadow_hits = 0;
    std::string static_exit_reason = "disabled";
    bool fixed_template_near = false;
    bool fixed_template_match = false;
    float fixed_template_score = 0.0f;
    std::string fixed_template_id;
    bool gated_phone_valid = false;
    bool high_confidence_phone = false;
    jiankong::custom_pipeline::EvidenceState gated_person_association =
        jiankong::custom_pipeline::EvidenceState::unknown;
    jiankong::custom_pipeline::EvidenceState gated_hand_relation =
        jiankong::custom_pipeline::EvidenceState::unknown;
    jiankong::custom_pipeline::EvidenceState gated_screen_intent =
        jiankong::custom_pipeline::EvidenceState::unknown;
    bool screen_ray_hit = false;
    bool exact_screen_ray_hit = false;
    bool corridor_screen_ray_hit = false;
    bool accepted() const { return reject_reason.empty(); }
};

struct FixedTemplateMatchResult {
    jiankong::custom_pipeline::FixedTemplateEvidence evidence =
        jiankong::custom_pipeline::FixedTemplateEvidence::none;
    float appearance_score = 0.0f;
    std::string template_id;
};

struct FixedPhoneTemplate {
    std::string template_id;
    std::string camera;
    Rect representative_box;
    std::vector<cv::Mat> descriptors;
};

static cv::Mat fixed_template_descriptor(const cv::Mat& image) {
    if (image.empty()) return {};
    cv::Mat gray;
    if (image.channels() == 4) {
        cv::cvtColor(image, gray, cv::COLOR_RGBA2GRAY);
    } else if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else if (image.channels() == 1) {
        gray = image;
    } else {
        return {};
    }
    cv::Mat resized;
    cv::resize(gray, resized, cv::Size(48, 48), 0.0, 0.0, cv::INTER_AREA);
    cv::GaussianBlur(resized, resized, cv::Size(3, 3), 0.0);
    cv::Mat descriptor;
    resized.convertTo(descriptor, CV_32F);
    descriptor = descriptor.reshape(1, 1);
    const cv::Scalar mean = cv::mean(descriptor);
    descriptor -= mean[0];
    const double norm = cv::norm(descriptor, cv::NORM_L2);
    if (!std::isfinite(norm) || norm < 1e-6) return {};
    descriptor /= norm;
    return descriptor;
}

static float fixed_template_similarity(
    const cv::Mat& descriptor, const std::vector<cv::Mat>& references) {
    if (descriptor.empty() || references.empty()) return 0.0f;
    std::vector<float> scores;
    scores.reserve(references.size());
    for (const cv::Mat& reference : references) {
        if (reference.empty() || reference.total() != descriptor.total()) continue;
        const double score = descriptor.dot(reference);
        if (std::isfinite(score)) scores.push_back(clampf(
            static_cast<float>(score), -1.0f, 1.0f));
    }
    if (scores.empty()) return 0.0f;
    std::sort(scores.begin(), scores.end());
    return scores[scores.size() / 2];
}

static Rect fixed_template_context_roi(const Rect& box, int width, int height) {
    const float context_width = std::max(96.0f, box.w() * 5.0f);
    const float context_height = std::max(96.0f, box.h() * 5.0f);
    const cv::Point2f center{
        0.5f * (box.x1 + box.x2),
        0.5f * (box.y1 + box.y2),
    };
    return clamp_rect(
        Rect{
            center.x - context_width * 0.5f,
            center.y - context_height * 0.5f,
            center.x + context_width * 0.5f,
            center.y + context_height * 0.5f,
        },
        width,
        height);
}

static cv::Mat download_fixed_template_context(
    const jiankong::custom_pipeline::DeviceFrameView& frame, const Rect& phone_box) {
    if (frame.rgba == nullptr || frame.pitch_bytes == 0 ||
        frame.width <= 0 || frame.height <= 0) {
        return {};
    }
    const Rect roi = fixed_template_context_roi(phone_box, frame.width, frame.height);
    const int x = std::max(0, static_cast<int>(std::floor(roi.x1)));
    const int y = std::max(0, static_cast<int>(std::floor(roi.y1)));
    const int right = std::min(frame.width, static_cast<int>(std::ceil(roi.x2)));
    const int bottom = std::min(frame.height, static_cast<int>(std::ceil(roi.y2)));
    const int width = right - x;
    const int height = bottom - y;
    if (width < 2 || height < 2) return {};
    cv::Mat rgba(height, width, CV_8UC4);
    const auto* source = frame.rgba +
        static_cast<std::size_t>(y) * frame.pitch_bytes +
        static_cast<std::size_t>(x) * 4U;
    check_cuda(cudaMemcpy2D(
        rgba.data,
        rgba.step,
        source,
        frame.pitch_bytes,
        static_cast<std::size_t>(width) * 4U,
        static_cast<std::size_t>(height),
        cudaMemcpyDeviceToHost),
        "fixed template ROI download");
    return rgba;
}

class FixedPhoneTemplateRegistry {
public:
    explicit FixedPhoneTemplateRegistry(fs::path root = {})
        : root_(std::move(root)) {}

    void maybe_reload(double now) {
        if (root_.empty() || !std::isfinite(now) ||
            now - last_reload_at_ < 5.0) {
            return;
        }
        last_reload_at_ = now;
        std::vector<FixedPhoneTemplate> loaded;
        std::error_code error;
        if (!fs::is_directory(root_, error) || error) {
            templates_.clear();
            return;
        }
        for (fs::directory_iterator it(root_, error), end;
             !error && it != end; it.increment(error)) {
            if (!it->is_directory(error) || error || it->is_symlink(error)) continue;
            const fs::path metadata_path = it->path() / "template.json";
            std::ifstream input(metadata_path);
            if (!input) continue;
            try {
                json payload;
                input >> payload;
                if (!payload.value("active", true)) continue;
                FixedPhoneTemplate item;
                item.template_id = payload.value("template_id", "");
                item.camera = payload.value("camera", "");
                if (item.template_id.empty() || item.camera.empty() ||
                    !payload.contains("samples") || !payload["samples"].is_array()) {
                    continue;
                }
                std::vector<float> centers_x;
                std::vector<float> centers_y;
                std::vector<float> widths;
                std::vector<float> heights;
                for (const auto& sample : payload["samples"]) {
                    if (!sample.is_object() || !sample.contains("phone_box") ||
                        !sample["phone_box"].is_array() ||
                        sample["phone_box"].size() != 4) {
                        continue;
                    }
                    Rect box{
                        sample["phone_box"][0].get<float>(),
                        sample["phone_box"][1].get<float>(),
                        sample["phone_box"][2].get<float>(),
                        sample["phone_box"][3].get<float>(),
                    };
                    if (box.w() < 2.0f || box.h() < 2.0f) continue;
                    centers_x.push_back((box.x1 + box.x2) * 0.5f);
                    centers_y.push_back((box.y1 + box.y2) * 0.5f);
                    widths.push_back(box.w());
                    heights.push_back(box.h());
                    const std::string image_name = sample.value("image", "");
                    const fs::path image_component(image_name);
                    if (image_component.empty() ||
                        image_component.has_parent_path()) {
                        continue;
                    }
                    const cv::Mat image = cv::imread(
                        (it->path() / image_component).string(), cv::IMREAD_COLOR);
                    cv::Mat descriptor = fixed_template_descriptor(image);
                    if (!descriptor.empty()) {
                        item.descriptors.push_back(std::move(descriptor));
                    }
                }
                if (centers_x.size() < 3 || item.descriptors.size() < 3) continue;
                const auto median = [](std::vector<float> values) {
                    std::sort(values.begin(), values.end());
                    return values[values.size() / 2];
                };
                const float center_x = median(centers_x);
                const float center_y = median(centers_y);
                const float box_width = median(widths);
                const float box_height = median(heights);
                item.representative_box = Rect{
                    center_x - box_width * 0.5f,
                    center_y - box_height * 0.5f,
                    center_x + box_width * 0.5f,
                    center_y + box_height * 0.5f,
                };
                loaded.push_back(std::move(item));
            } catch (const std::exception& error_value) {
                std::cerr << "[FIXED_TEMPLATE_SKIP] path=" << metadata_path
                          << " error=" << error_value.what() << std::endl;
            }
        }
        templates_.swap(loaded);
        std::cout << "[FIXED_TEMPLATE_RELOAD] count=" << templates_.size()
                  << " root=" << root_.string() << std::endl;
    }

    FixedTemplateMatchResult match(
        const std::string& camera,
        const Rect& phone_box,
        const jiankong::custom_pipeline::DeviceFrameView& frame) const {
        FixedTemplateMatchResult result;
        const float detected_width = phone_box.w();
        const float detected_height = phone_box.h();
        if (detected_width < 2.0f || detected_height < 2.0f) return result;
        const float detected_center_x = (phone_box.x1 + phone_box.x2) * 0.5f;
        const float detected_center_y = (phone_box.y1 + phone_box.y2) * 0.5f;
        cv::Mat current_descriptor;
        for (const auto& item : templates_) {
            if (item.camera != camera) continue;
            const Rect& reference = item.representative_box;
            const float reference_center_x = (reference.x1 + reference.x2) * 0.5f;
            const float reference_center_y = (reference.y1 + reference.y2) * 0.5f;
            const float reference_diag = std::hypot(reference.w(), reference.h());
            const float center_distance = std::hypot(
                detected_center_x - reference_center_x,
                detected_center_y - reference_center_y);
            const float width_ratio = detected_width / std::max(1.0f, reference.w());
            const float height_ratio = detected_height / std::max(1.0f, reference.h());
            const auto spatial_evidence =
                jiankong::custom_pipeline::classify_fixed_template_spatial_evidence(
                    center_distance, reference_diag, width_ratio, height_ratio);
            if (spatial_evidence ==
                jiankong::custom_pipeline::FixedTemplateEvidence::none) {
                continue;
            }
            if (spatial_evidence ==
                jiankong::custom_pipeline::FixedTemplateEvidence::mismatch) {
                if (result.evidence !=
                    jiankong::custom_pipeline::FixedTemplateEvidence::matched) {
                    result.evidence = spatial_evidence;
                    result.template_id = item.template_id;
                }
                continue;
            }

            // A user-confirmed fixed-phone template represents a physical
            // location.  Appearance is retained as diagnostics, but must not
            // veto a stable location match when a nearby person changes the
            // 96px context crop.  Movement and wrist-follow checks run before
            // this evidence reaches the static policy and still release a
            // phone that is picked up.
            if (result.evidence !=
                jiankong::custom_pipeline::FixedTemplateEvidence::matched) {
                result.template_id = item.template_id;
            }
            result.evidence =
                jiankong::custom_pipeline::FixedTemplateEvidence::matched;
            if (current_descriptor.empty()) {
                current_descriptor = fixed_template_descriptor(
                    download_fixed_template_context(frame, phone_box));
            }
            const float score =
                fixed_template_similarity(current_descriptor, item.descriptors);
            if (score > result.appearance_score) {
                result.appearance_score = score;
                result.template_id = item.template_id;
            }
        }
        return result;
    }

private:
    fs::path root_;
    std::vector<FixedPhoneTemplate> templates_;
    double last_reload_at_ = -std::numeric_limits<double>::infinity();
};

struct FormalSpatialSample {
    float center_x = 0.0f;
    float center_y = 0.0f;
    bool valid = false;
};

struct StaticPhoneObs {
    int frame_id = 0;
    float timestamp = 0.0f;
    cv::Point2f center;
    Rect box;
    Rect person_box;
    bool has_wrist = false;
    cv::Point2f wrist;
    float risk_score = 0.0f;
    bool candidate = false;
};

struct PersonTrackState {
    explicit PersonTrackState(
        jiankong::custom_pipeline::SpatialStaticConfig config = {})
        : spatial_static_policy(config), spatial_static_shadow_seconds(config.short_seconds) {}

    int track_id = -1;
    std::deque<Rect> bbox_history;
    std::deque<int> phone_history;
    std::deque<float> risk_history;
    std::deque<int> candidate_history;
    std::deque<int> handheld_history;
    std::deque<StaticPhoneObs> static_phone_history;
    std::deque<FormalSpatialSample> formal_spatial_history;
    jiankong::custom_pipeline::SpatialStaticPolicy spatial_static_policy;
    std::deque<jiankong::custom_pipeline::ShadowRiskSample> static_shadow_history;
    jiankong::custom_pipeline::SpatialStaticDecision last_static_decision{};
    jiankong::custom_pipeline::GatedEventPolicy gated_event_policy;
    jiankong::custom_pipeline::GatedEventDecision gated_event_decision{};
    bool legacy_alarm_triggered = false;
    bool spatial_static_enabled = true;
    double static_pending_started_at = std::numeric_limits<double>::quiet_NaN();
    double static_pending_duration = 0.0;
    double spatial_static_shadow_seconds = 3.0;
    float static_hotspot_score = 0.0f;
    std::string state = "S0_CLEAR";
    int stable_count = 0;
    int window_hits = 0;
    int legacy_window_hits = 0;
    int handheld_phone_hits = 0;
    int handheld_phone_stable_count = 0;
    int static_phone_suppressed_hits = 0;
    bool alarm_triggered = false;
    int last_seen = 0;
    Rect last_bbox;
    bool has_last_bbox = false;
    Rect smoothed_bbox;
    bool has_smoothed_bbox = false;
    bool suspect_active = false;
    int last_suspect_frame = 0;
    float last_risk_score = 0.0f;
    std::string last_screen_id;
};

static jiankong::custom_pipeline::StaticEvidenceState gated_static_state(
    const PersonTrackState& state, const CandidateEval* ev);
static jiankong::custom_pipeline::GatedFrameEvidence gated_frame_evidence(
    const PersonTrackState& state, const CandidateEval* ev);

struct StreamState {
    std::string path;
    std::string output_path;
    std::string view;
    cv::VideoCapture cap;
    int width = 0;
    int height = 0;
    double native_fps = 25.0;
    int total_frames = 0;
    std::vector<int> sample_indices;
    size_t sample_cursor = 0;
    int raw_cursor = 0;
    bool done = false;
    long long frames = 0;
    long long persons = 0;
    long long rois = 0;
    long long phones_raw = 0;
    long long phones_nms = 0;
    long long accepted_candidates = 0;
    long long static_phone_suppressed_frames = 0;
    long long static_phone_suppressed_candidates = 0;
    long long desk_zone_phone_frames = 0;
    long long alarm_frames = 0;
    bool screen_active = false;
    std::vector<ScreenConfig> screens;
    std::map<int, PersonTrackState> person_states;
    std::unique_ptr<jiankong::custom_pipeline::CameraStaticHotspotMap> static_hotspots;
    fs::path static_hotspot_path;
    double last_static_hotspot_persisted_at = -std::numeric_limits<double>::infinity();
    bool static_hotspot_dirty = false;
    int next_track_id = 1;
    int frame_id = 0;
    float alert_counter = 0.0f;
    std::set<int> previous_alarm_tracks;
    cv::VideoWriter writer;
};

struct FrameQueue {
    std::mutex mutex;
    std::condition_variable not_empty;
    std::condition_variable not_full;
    std::deque<cv::Mat> frames;
    bool done = false;
};

static std::vector<char> read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("failed to open " + path);
    in.seekg(0, std::ios::end);
    size_t size = static_cast<size_t>(in.tellg());
    in.seekg(0, std::ios::beg);
    std::vector<char> data(size);
    in.read(data.data(), static_cast<std::streamsize>(data.size()));
    return data;
}

class PinnedFloatBuffer {
public:
    PinnedFloatBuffer() = default;
    ~PinnedFloatBuffer() { reset(); }

    PinnedFloatBuffer(const PinnedFloatBuffer&) = delete;
    PinnedFloatBuffer& operator=(const PinnedFloatBuffer&) = delete;

    void resize(size_t n) {
        if (n <= size_) return;
        reset();
        size_ = n;
        check_cuda(cudaMallocHost(reinterpret_cast<void**>(&ptr_), size_ * sizeof(float)), "cudaMallocHost");
    }

    void reset() {
        if (ptr_) {
            cudaFreeHost(ptr_);
            ptr_ = nullptr;
        }
        size_ = 0;
    }

    float* data() { return ptr_; }
    const float* data() const { return ptr_; }
    size_t size() const { return size_; }

private:
    float* ptr_ = nullptr;
    size_t size_ = 0;
};

class PinnedIntBuffer {
public:
    PinnedIntBuffer() = default;
    ~PinnedIntBuffer() { reset(); }

    PinnedIntBuffer(const PinnedIntBuffer&) = delete;
    PinnedIntBuffer& operator=(const PinnedIntBuffer&) = delete;

    void resize(size_t n) {
        if (n <= size_) return;
        reset();
        size_ = n;
        check_cuda(cudaMallocHost(reinterpret_cast<void**>(&ptr_), size_ * sizeof(int)), "cudaMallocHost");
    }

    void reset() {
        if (ptr_) {
            cudaFreeHost(ptr_);
            ptr_ = nullptr;
        }
        size_ = 0;
    }

    int* data() { return ptr_; }
    const int* data() const { return ptr_; }

private:
    int* ptr_ = nullptr;
    size_t size_ = 0;
};

__global__ void compact_trt_candidates_kernel(const float* output,
                                              int logical_batch,
                                              int channels,
                                              int num_boxes,
                                              int confidence_channel,
                                              float confidence_threshold,
                                              int* counts,
                                              int* indices,
                                              float* values) {
    const int linear = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = logical_batch * num_boxes;
    if (linear >= total) return;

    const int batch_index = linear / num_boxes;
    const int box_index = linear - batch_index * num_boxes;
    const float* base = output + static_cast<size_t>(batch_index) * channels * num_boxes;
    // Match the former CPU condition exactly, including its NaN behaviour.
    if (base[confidence_channel * num_boxes + box_index] < confidence_threshold) return;

    const int slot = atomicAdd(counts + batch_index, 1);
    const size_t compact_index = static_cast<size_t>(batch_index) * num_boxes + slot;
    indices[compact_index] = box_index;
    float* compact = values + compact_index * channels;
    for (int channel = 0; channel < channels; ++channel) {
        compact[channel] = base[channel * num_boxes + box_index];
    }
}

class TrtRunner {
public:
    explicit TrtRunner(const std::string& plan_path) {
        auto plan = read_file(plan_path);
        runtime_ = nvinfer1::createInferRuntime(g_logger);
        if (!runtime_) throw std::runtime_error("createInferRuntime failed");
        engine_ = runtime_->deserializeCudaEngine(plan.data(), plan.size());
        if (!engine_) throw std::runtime_error("deserializeCudaEngine failed: " + plan_path);
        context_ = engine_->createExecutionContext();
        if (!context_) throw std::runtime_error("createExecutionContext failed");

        const int nb = engine_->getNbBindings();
        buffers_.assign(nb, nullptr);
        for (int i = 0; i < nb; ++i) {
            const bool is_input = engine_->bindingIsInput(i);
            const auto dims = engine_->getBindingDimensions(i);
            if (is_input) {
                input_index_ = i;
                engine_input_dims_ = dims;
                dynamic_input_ = has_dynamic_dim(dims);
                input_dims_ = dynamic_input_
                                  ? engine_->getProfileDimensions(i, 0, nvinfer1::OptProfileSelector::kMAX)
                                  : dims;
                input_capacity_elems_ = volume(input_dims_);
                check_cuda(cudaMalloc(&buffers_[i], input_capacity_elems_ * sizeof(float)), "cudaMalloc input");
                host_input_.resize(input_capacity_elems_);
            } else {
                output_index_ = i;
                engine_output_dims_ = dims;
                output_dims_ = dims;
            }
            std::cerr << "[ENGINE] " << plan_path << " binding=" << i
                      << " name=" << engine_->getBindingName(i)
                      << " input=" << is_input
                      << " dims=" << dims_to_string(dims)
                      << std::endl;
        }
        if (input_index_ < 0 || output_index_ < 0) throw std::runtime_error("missing bindings");
        check_cuda(cudaStreamCreate(&stream_), "cudaStreamCreate");
        configure_shape(input_dims_.d[0]);
    }

    ~TrtRunner() {
        if (compact_values_device_) cudaFree(compact_values_device_);
        if (compact_indices_device_) cudaFree(compact_indices_device_);
        if (compact_counts_device_) cudaFree(compact_counts_device_);
        for (void* p : buffers_) {
            if (p) cudaFree(p);
        }
        if (stream_) cudaStreamDestroy(stream_);
        if (context_) context_->destroy();
        if (engine_) engine_->destroy();
        if (runtime_) runtime_->destroy();
    }

    float* input() { return host_input_.data(); }
    const float* output() const { return host_output_.data(); }
    nvinfer1::Dims input_dims() const { return input_dims_; }
    nvinfer1::Dims output_dims() const { return output_dims_; }
    bool dynamic_input() const { return dynamic_input_; }
    int preprocessing_batch_size(int actual_batch) const {
        if (!dynamic_input_) return input_dims_.d[0];
        return std::max(1, std::min(actual_batch, input_dims_.d[0]));
    }

    void infer(bool input_already_on_device = false, int actual_batch = -1) {
        configure_shape(actual_batch);
        if (!input_already_on_device) {
            check_cuda(cudaMemcpyAsync(buffers_[input_index_], host_input_.data(), current_input_elems_ * sizeof(float),
                                       cudaMemcpyHostToDevice, stream_),
                       "cudaMemcpyAsync H2D");
        }
        if (!context_->enqueueV2(buffers_.data(), stream_, nullptr)) {
            throw std::runtime_error("enqueueV2 failed");
        }
        check_cuda(cudaMemcpyAsync(host_output_.data(), buffers_[output_index_], current_output_elems_ * sizeof(float),
                                   cudaMemcpyDeviceToHost, stream_),
                   "cudaMemcpyAsync D2H");
        check_cuda(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
    }

    void infer_compacted(bool input_already_on_device,
                         int actual_batch,
                         int channels,
                         int num_boxes,
                         int confidence_channel,
                         float confidence_threshold) {
        configure_shape(actual_batch);
        if (output_dims_.nbDims != 3 || output_dims_.d[0] < actual_batch ||
            output_dims_.d[1] != channels || output_dims_.d[2] != num_boxes) {
            throw std::runtime_error("unexpected TensorRT output shape for GPU candidate compaction");
        }
        if (actual_batch <= 0 || confidence_channel < 0 || confidence_channel >= channels) {
            throw std::runtime_error("invalid GPU candidate compaction parameters");
        }
        ensure_compaction_capacity(actual_batch, channels, num_boxes);
        if (!input_already_on_device) {
            check_cuda(cudaMemcpyAsync(buffers_[input_index_], host_input_.data(), current_input_elems_ * sizeof(float),
                                       cudaMemcpyHostToDevice, stream_),
                       "cudaMemcpyAsync H2D");
        }
        if (!context_->enqueueV2(buffers_.data(), stream_, nullptr)) {
            throw std::runtime_error("enqueueV2 failed");
        }

        check_cuda(cudaMemsetAsync(compact_counts_device_, 0, static_cast<size_t>(actual_batch) * sizeof(int), stream_),
                   "cudaMemsetAsync compact counts");
        const int total = actual_batch * num_boxes;
        const int threads = 256;
        compact_trt_candidates_kernel<<<(total + threads - 1) / threads, threads, 0, stream_>>>(
            static_cast<const float*>(buffers_[output_index_]),
            actual_batch,
            channels,
            num_boxes,
            confidence_channel,
            confidence_threshold,
            compact_counts_device_,
            compact_indices_device_,
            compact_values_device_);
        check_cuda(cudaGetLastError(), "compact_trt_candidates_kernel");
        check_cuda(cudaMemcpyAsync(compact_counts_host_.data(), compact_counts_device_,
                                   static_cast<size_t>(actual_batch) * sizeof(int),
                                   cudaMemcpyDeviceToHost, stream_),
                   "cudaMemcpyAsync compact counts D2H");
        check_cuda(cudaStreamSynchronize(stream_), "cudaStreamSynchronize compact counts");

        for (int batch_index = 0; batch_index < actual_batch; ++batch_index) {
            const int count = compact_counts_host_.data()[batch_index];
            if (count < 0 || count > num_boxes) {
                throw std::runtime_error("invalid compacted TensorRT candidate count");
            }
            if (count == 0) continue;
            const size_t candidate_offset = static_cast<size_t>(batch_index) * num_boxes;
            check_cuda(cudaMemcpyAsync(compact_indices_host_.data() + candidate_offset,
                                       compact_indices_device_ + candidate_offset,
                                       static_cast<size_t>(count) * sizeof(int),
                                       cudaMemcpyDeviceToHost, stream_),
                       "cudaMemcpyAsync compact indices D2H");
            check_cuda(cudaMemcpyAsync(compact_values_host_.data() + candidate_offset * channels,
                                       compact_values_device_ + candidate_offset * channels,
                                       static_cast<size_t>(count) * channels * sizeof(float),
                                       cudaMemcpyDeviceToHost, stream_),
                       "cudaMemcpyAsync compact values D2H");
        }
        check_cuda(cudaStreamSynchronize(stream_), "cudaStreamSynchronize compact values");
        compact_batch_ = actual_batch;
        compact_channels_ = channels;
        compact_num_boxes_ = num_boxes;
    }

    int compacted_count(int batch_index) const {
        if (batch_index < 0 || batch_index >= compact_batch_) throw std::out_of_range("compacted batch index");
        return compact_counts_host_.data()[batch_index];
    }

    int compacted_original_index(int batch_index, int slot) const {
        validate_compacted_slot(batch_index, slot);
        return compact_indices_host_.data()[static_cast<size_t>(batch_index) * compact_num_boxes_ + slot];
    }

    const float* compacted_values(int batch_index, int slot) const {
        validate_compacted_slot(batch_index, slot);
        const size_t candidate_index = static_cast<size_t>(batch_index) * compact_num_boxes_ + slot;
        return compact_values_host_.data() + candidate_index * compact_channels_;
    }

    void* device_input() { return buffers_[input_index_]; }
    cudaStream_t stream() const { return stream_; }

private:
    static std::string dims_to_string(const nvinfer1::Dims& d) {
        std::ostringstream oss;
        for (int i = 0; i < d.nbDims; ++i) {
            if (i) oss << "x";
            oss << d.d[i];
        }
        return oss.str();
    }

    void ensure_device_capacity(int binding_index, size_t elems, size_t& capacity, const char* name) {
        if (elems <= capacity) return;
        if (buffers_[binding_index]) {
            cudaFree(buffers_[binding_index]);
            buffers_[binding_index] = nullptr;
        }
        check_cuda(cudaMalloc(&buffers_[binding_index], elems * sizeof(float)), name);
        capacity = elems;
    }

    void ensure_compaction_capacity(int batch, int channels, int num_boxes) {
        const size_t candidates = static_cast<size_t>(batch) * num_boxes;
        const size_t values = candidates * channels;
        if (static_cast<size_t>(batch) > compact_count_capacity_) {
            if (compact_counts_device_) cudaFree(compact_counts_device_);
            compact_counts_device_ = nullptr;
            check_cuda(cudaMalloc(reinterpret_cast<void**>(&compact_counts_device_),
                                  static_cast<size_t>(batch) * sizeof(int)),
                       "cudaMalloc compact counts");
            compact_counts_host_.resize(batch);
            compact_count_capacity_ = batch;
        }
        if (candidates > compact_candidate_capacity_) {
            if (compact_indices_device_) cudaFree(compact_indices_device_);
            compact_indices_device_ = nullptr;
            check_cuda(cudaMalloc(reinterpret_cast<void**>(&compact_indices_device_), candidates * sizeof(int)),
                       "cudaMalloc compact indices");
            compact_indices_host_.resize(candidates);
            compact_candidate_capacity_ = candidates;
        }
        if (values > compact_value_capacity_) {
            if (compact_values_device_) cudaFree(compact_values_device_);
            compact_values_device_ = nullptr;
            check_cuda(cudaMalloc(reinterpret_cast<void**>(&compact_values_device_), values * sizeof(float)),
                       "cudaMalloc compact values");
            compact_values_host_.resize(values);
            compact_value_capacity_ = values;
        }
    }

    void validate_compacted_slot(int batch_index, int slot) const {
        if (batch_index < 0 || batch_index >= compact_batch_ ||
            slot < 0 || slot >= compact_counts_host_.data()[batch_index]) {
            throw std::out_of_range("compacted candidate slot");
        }
    }

    void configure_shape(int requested_batch) {
        nvinfer1::Dims actual_input = input_dims_;
        if (requested_batch > 0 && actual_input.nbDims > 0) {
            actual_input.d[0] = std::min(requested_batch, input_dims_.d[0]);
        }

        if (dynamic_input_) {
            if (!context_->setBindingDimensions(input_index_, actual_input)) {
                throw std::runtime_error("setBindingDimensions failed");
            }
            if (!context_->allInputDimensionsSpecified()) {
                throw std::runtime_error("TensorRT input dimensions not fully specified");
            }
            actual_input = context_->getBindingDimensions(input_index_);
            output_dims_ = context_->getBindingDimensions(output_index_);
        } else {
            actual_input = input_dims_;
            output_dims_ = engine_output_dims_;
        }

        current_input_elems_ = volume(actual_input);
        current_output_elems_ = volume(output_dims_);
        ensure_device_capacity(input_index_, current_input_elems_, input_capacity_elems_, "cudaMalloc input");
        ensure_device_capacity(output_index_, current_output_elems_, output_capacity_elems_, "cudaMalloc output");
        // Keep the host buffer at the engine's maximum batch capacity.  The
        // live preprocessor pads the inactive slots of a fixed-batch engine;
        // shrinking this buffer for a dynamic profile would make those CUDA
        // writes overflow as soon as fewer than the maximum cameras are live.
        host_output_.resize(current_output_elems_);
    }

    nvinfer1::IRuntime* runtime_ = nullptr;
    nvinfer1::ICudaEngine* engine_ = nullptr;
    nvinfer1::IExecutionContext* context_ = nullptr;
    cudaStream_t stream_ = nullptr;
    std::vector<void*> buffers_;
    int input_index_ = -1;
    int output_index_ = -1;
    nvinfer1::Dims engine_input_dims_{};
    nvinfer1::Dims engine_output_dims_{};
    nvinfer1::Dims input_dims_{};
    nvinfer1::Dims output_dims_{};
    size_t input_capacity_elems_ = 0;
    size_t output_capacity_elems_ = 0;
    size_t current_input_elems_ = 0;
    size_t current_output_elems_ = 0;
    bool dynamic_input_ = false;
    PinnedFloatBuffer host_input_;
    PinnedFloatBuffer host_output_;
    int* compact_counts_device_ = nullptr;
    int* compact_indices_device_ = nullptr;
    float* compact_values_device_ = nullptr;
    size_t compact_count_capacity_ = 0;
    size_t compact_candidate_capacity_ = 0;
    size_t compact_value_capacity_ = 0;
    int compact_batch_ = 0;
    int compact_channels_ = 0;
    int compact_num_boxes_ = 0;
    PinnedIntBuffer compact_counts_host_;
    PinnedIntBuffer compact_indices_host_;
    PinnedFloatBuffer compact_values_host_;
};

__global__ void bgr_letterbox_to_rgb_chw_kernel(const unsigned char* src,
                                                int src_w,
                                                int src_h,
                                                int crop_x,
                                                int crop_y,
                                                int crop_w,
                                                int crop_h,
                                                float scale,
                                                int pad_x,
                                                int pad_y,
                                                int out_size,
                                                float* dst,
                                                int batch_index) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= out_size || y >= out_size) return;

    const int new_w = max(1, __float2int_rn(crop_w * scale));
    const int new_h = max(1, __float2int_rn(crop_h * scale));
    const int area = out_size * out_size;
    float* out = dst + static_cast<size_t>(batch_index) * 3 * area;
    const int off = y * out_size + x;

    float r = 114.0f / 255.0f;
    float g = 114.0f / 255.0f;
    float b = 114.0f / 255.0f;

    if (x >= pad_x && x < pad_x + new_w && y >= pad_y && y < pad_y + new_h) {
        const float fx = (static_cast<float>(x - pad_x) + 0.5f) / scale - 0.5f;
        const float fy = (static_cast<float>(y - pad_y) + 0.5f) / scale - 0.5f;
        int x0 = static_cast<int>(floorf(fx));
        int y0 = static_cast<int>(floorf(fy));
        const float tx = fx - x0;
        const float ty = fy - y0;
        x0 = max(0, min(crop_w - 1, x0));
        y0 = max(0, min(crop_h - 1, y0));
        const int x1 = max(0, min(crop_w - 1, x0 + 1));
        const int y1 = max(0, min(crop_h - 1, y0 + 1));
        const int gx0 = max(0, min(src_w - 1, crop_x + x0));
        const int gx1 = max(0, min(src_w - 1, crop_x + x1));
        const int gy0 = max(0, min(src_h - 1, crop_y + y0));
        const int gy1 = max(0, min(src_h - 1, crop_y + y1));

        const unsigned char* p00 = src + (static_cast<size_t>(gy0) * src_w + gx0) * 3;
        const unsigned char* p01 = src + (static_cast<size_t>(gy0) * src_w + gx1) * 3;
        const unsigned char* p10 = src + (static_cast<size_t>(gy1) * src_w + gx0) * 3;
        const unsigned char* p11 = src + (static_cast<size_t>(gy1) * src_w + gx1) * 3;
        const float w00 = (1.0f - tx) * (1.0f - ty);
        const float w01 = tx * (1.0f - ty);
        const float w10 = (1.0f - tx) * ty;
        const float w11 = tx * ty;
        b = (w00 * p00[0] + w01 * p01[0] + w10 * p10[0] + w11 * p11[0]) / 255.0f;
        g = (w00 * p00[1] + w01 * p01[1] + w10 * p10[1] + w11 * p11[1]) / 255.0f;
        r = (w00 * p00[2] + w01 * p01[2] + w10 * p10[2] + w11 * p11[2]) / 255.0f;
    }

    out[off] = r;
    out[area + off] = g;
    out[2 * area + off] = b;
}

class GpuPreprocessor {
public:
    explicit GpuPreprocessor(size_t streams) : frame_ptrs_(streams, nullptr), capacities_(streams, 0) {}

    ~GpuPreprocessor() {
        for (auto* p : frame_ptrs_) {
            if (p) cudaFree(p);
        }
    }

    GpuPreprocessor(const GpuPreprocessor&) = delete;
    GpuPreprocessor& operator=(const GpuPreprocessor&) = delete;

    void upload_frame(int stream_idx, const cv::Mat& frame, cudaStream_t stream) {
        const size_t bytes = static_cast<size_t>(frame.cols) * frame.rows * 3;
        ensure_capacity(stream_idx, bytes);
        if (frame.isContinuous()) {
            check_cuda(cudaMemcpyAsync(frame_ptrs_[stream_idx], frame.data, bytes, cudaMemcpyHostToDevice, stream),
                       "cudaMemcpyAsync frame H2D");
        } else {
            cv::Mat contiguous = frame.clone();
            check_cuda(cudaMemcpyAsync(frame_ptrs_[stream_idx], contiguous.data, bytes, cudaMemcpyHostToDevice, stream),
                       "cudaMemcpyAsync frame H2D clone");
        }
    }

    LetterboxMeta preprocess(int stream_idx,
                             int src_w,
                             int src_h,
                             int crop_x,
                             int crop_y,
                             int crop_w,
                             int crop_h,
                             int out_size,
                             float* device_dst,
                             int batch_index,
                             cudaStream_t stream) {
        crop_x = std::max(0, std::min(src_w - 1, crop_x));
        crop_y = std::max(0, std::min(src_h - 1, crop_y));
        crop_w = std::max(1, std::min(src_w - crop_x, crop_w));
        crop_h = std::max(1, std::min(src_h - crop_y, crop_h));
        LetterboxMeta meta = compute_letterbox_meta(crop_w, crop_h, out_size);
        const dim3 block(16, 16);
        const dim3 grid((out_size + block.x - 1) / block.x, (out_size + block.y - 1) / block.y);
        bgr_letterbox_to_rgb_chw_kernel<<<grid, block, 0, stream>>>(
            frame_ptrs_[stream_idx],
            src_w,
            src_h,
            crop_x,
            crop_y,
            crop_w,
            crop_h,
            meta.scale,
            meta.pad_x,
            meta.pad_y,
            out_size,
            device_dst,
            batch_index);
        check_cuda(cudaGetLastError(), "launch bgr_letterbox_to_rgb_chw_kernel");
        return meta;
    }

private:
    void ensure_capacity(int stream_idx, size_t bytes) {
        if (stream_idx < 0 || stream_idx >= static_cast<int>(frame_ptrs_.size())) {
            throw std::runtime_error("invalid stream index for GPU frame buffer");
        }
        if (capacities_[stream_idx] >= bytes) return;
        if (frame_ptrs_[stream_idx]) cudaFree(frame_ptrs_[stream_idx]);
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&frame_ptrs_[stream_idx]), bytes), "cudaMalloc frame");
        capacities_[stream_idx] = bytes;
    }

    std::vector<unsigned char*> frame_ptrs_;
    std::vector<size_t> capacities_;
};

static LetterboxMeta preprocess_rgba_device_frame(
    const jiankong::custom_pipeline::DeviceFrameView& frame,
    int crop_x,
    int crop_y,
    int crop_w,
    int crop_h,
    int out_size,
    float* device_dst,
    int batch_index,
    cudaStream_t stream) {
    const auto meta = jiankong::custom_pipeline::launch_rgba_pitch_letterbox(
        frame.rgba,
        frame.pitch_bytes,
        frame.width,
        frame.height,
        crop_x,
        crop_y,
        crop_w,
        crop_h,
        out_size,
        device_dst,
        batch_index,
        stream);
    return LetterboxMeta{
        meta.scale,
        meta.pad_x,
        meta.pad_y,
        meta.source_width,
        meta.source_height,
    };
}

static std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
    return s;
}

static int view_order(const std::string& p) {
    std::string s = lower(p);
    if (s.find("dianqi1") != std::string::npos) return 0;
    if (s.find("dianqi2") != std::string::npos) return 1;
    if (s.find("jixie1") != std::string::npos) return 2;
    if (s.find("jixie2") != std::string::npos) return 3;
    if (s.find("ruanjian1") != std::string::npos || s.find("software1") != std::string::npos) return 4;
    if (s.find("ruanjian2") != std::string::npos || s.find("software2") != std::string::npos) return 5;
    if (s.find("zoulang") != std::string::npos || s.find("corridor") != std::string::npos) return 6;
    return 99;
}

static std::vector<std::string> find_videos_by_rank_from_end(const std::string& root, int rank_from_end) {
    std::vector<std::string> out;
    rank_from_end = std::max(1, rank_from_end);
    std::vector<fs::path> direct_videos;
    for (const auto& ent : fs::directory_iterator(root)) {
        if (!ent.is_regular_file()) continue;
        std::string ext = lower(ent.path().extension().string());
        if (ext != ".mp4" && ext != ".avi" && ext != ".mkv" && ext != ".mov") continue;
        direct_videos.push_back(ent.path());
    }
    if (!direct_videos.empty()) {
        std::sort(direct_videos.begin(), direct_videos.end(), [](const fs::path& a, const fs::path& b) {
            int oa = view_order(a.string()), ob = view_order(b.string());
            if (oa != ob) return oa < ob;
            return a.filename() < b.filename();
        });
        for (const auto& p : direct_videos) out.push_back(p.string());
        return out;
    }
    for (const auto& ent : fs::directory_iterator(root)) {
        if (!ent.is_directory()) continue;
        std::vector<fs::path> candidates;
        for (const auto& f : fs::directory_iterator(ent.path())) {
            if (!f.is_regular_file()) continue;
            std::string ext = lower(f.path().extension().string());
            if (ext != ".mp4" && ext != ".avi" && ext != ".mkv" && ext != ".mov") continue;
            candidates.push_back(f.path());
        }
        if (candidates.empty()) continue;
        std::sort(candidates.begin(), candidates.end(), [](const fs::path& a, const fs::path& b) {
            auto ta = fs::last_write_time(a);
            auto tb = fs::last_write_time(b);
            if (ta != tb) return ta < tb;
            return a.filename() < b.filename();
        });
        int idx = static_cast<int>(candidates.size()) - rank_from_end;
        if (idx < 0) {
            idx = 0;
        }
        out.push_back(candidates[idx].string());
    }
    std::sort(out.begin(), out.end(), [](const std::string& a, const std::string& b) {
        int oa = view_order(a), ob = view_order(b);
        if (oa != ob) return oa < ob;
        return a < b;
    });
    return out;
}

static std::string safe_stem(const std::string& path) {
    std::string s = fs::path(path).stem().string();
    for (char& c : s) {
        const bool ok = std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '-' || c == '.';
        if (!ok) c = '_';
    }
    return s;
}

static void load_static_hotspots(StreamState& stream, double now) {
    if (!stream.static_hotspots || stream.static_hotspot_path.empty()) return;
    std::error_code exists_error;
    const bool exists = fs::exists(stream.static_hotspot_path, exists_error);
    if (!exists && !exists_error) return;

    try {
        if (exists_error) {
            throw std::runtime_error("state existence check failed: " + exists_error.message());
        }
        std::ifstream input(stream.static_hotspot_path);
        if (!input) throw std::runtime_error("state file could not be opened");
        const json state = json::parse(input);
        if (!state.is_object() || !state.contains("cells") || !state["cells"].is_array()) {
            throw std::runtime_error("state root must contain a cells array");
        }

        std::map<std::pair<int, int>, jiankong::custom_pipeline::StaticHotspotCell> cells;
        for (const auto& item : state["cells"]) {
            if (!item.is_object() || !item.contains("x") || !item["x"].is_number_integer() ||
                !item.contains("y") || !item["y"].is_number_integer() ||
                !item.contains("heat") || !item["heat"].is_number() ||
                !item.contains("last_confirmed_at") ||
                !item["last_confirmed_at"].is_number()) {
                throw std::runtime_error("state contains an invalid cell");
            }
            cells[{item["x"].get<int>(), item["y"].get<int>()}] = {
                item["heat"].get<float>(), item["last_confirmed_at"].get<double>()};
        }
        stream.static_hotspots->restore(cells, now);
        stream.static_hotspots->decay(now);
    } catch (const std::exception& error) {
        stream.static_hotspots->restore({}, now);
        std::cerr << "[STATIC_HOTSPOT_WARNING] camera=" << stream.view
                  << " path=" << stream.static_hotspot_path.string()
                  << " reason=" << error.what() << std::endl;
    }
}

static bool write_static_hotspots_atomic(const StreamState& stream) {
    if (!stream.static_hotspots || stream.static_hotspot_path.empty()) return true;
    const fs::path temporary = stream.static_hotspot_path.string() + ".tmp";
    try {
        fs::create_directories(stream.static_hotspot_path.parent_path());
        json state;
        state["version"] = 1;
        state["camera_name"] = stream.view;
        state["frame_width"] = stream.width;
        state["frame_height"] = stream.height;
        state["cells"] = json::array();
        for (const auto& [key, cell] : stream.static_hotspots->cells()) {
            state["cells"].push_back({
                {"x", key.first},
                {"y", key.second},
                {"heat", cell.heat},
                {"last_confirmed_at", cell.last_confirmed_at},
            });
        }

        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        if (!output) throw std::runtime_error("temporary state file could not be opened");
        output << state.dump(2) << '\n';
        output.flush();
        if (!output) throw std::runtime_error("temporary state file flush failed");
        output.close();
        if (!output) throw std::runtime_error("temporary state file close failed");

        std::error_code rename_error;
        fs::rename(temporary, stream.static_hotspot_path, rename_error);
        if (rename_error) {
            throw std::runtime_error("atomic state rename failed: " + rename_error.message());
        }
        return true;
    } catch (const std::exception& error) {
        std::error_code cleanup_error;
        fs::remove(temporary, cleanup_error);
        std::cerr << "[STATIC_HOTSPOT_WARNING] camera=" << stream.view
                  << " path=" << stream.static_hotspot_path.string()
                  << " reason=" << error.what() << std::endl;
        return false;
    }
}

static void persist_static_hotspots(StreamState& stream, double now, bool force) {
    if (!stream.static_hotspots || !std::isfinite(now)) return;
    if (!force && (!stream.static_hotspot_dirty ||
                   now - stream.last_static_hotspot_persisted_at < 30.0)) {
        return;
    }
    stream.last_static_hotspot_persisted_at = now;
    stream.static_hotspots->decay(now);
    if (write_static_hotspots_atomic(stream)) stream.static_hotspot_dirty = false;
}

static std::string calibration_name_for_path(const std::string& path) {
    const std::string s = lower(path);
    if (s.find("dianqi1") != std::string::npos || s.find("camera_01") != std::string::npos) return "camera_01_screen_calibration_v21.json";
    if (s.find("dianqi2") != std::string::npos || s.find("camera_02") != std::string::npos) return "camera_02_screen_calibration_v21.json";
    if (s.find("jixie1") != std::string::npos || s.find("mechanical1") != std::string::npos) return "camera_mechanical_01_screen_calibration_v21.json";
    if (s.find("jixie2") != std::string::npos || s.find("mechanical2") != std::string::npos) return "camera_mechanical_02_screen_calibration_v21.json";
    if (s.find("ruanjian1") != std::string::npos || s.find("software1") != std::string::npos) return "camera_software_01_screen_calibration_v21.json";
    if (s.find("ruanjian2") != std::string::npos || s.find("software2") != std::string::npos) return "camera_software_02_screen_calibration_v21.json";
    if (s.find("zoulang") != std::string::npos || s.find("corridor") != std::string::npos) return "camera_corridor_screen_calibration_v21.json";
    return "";
}

struct ManagedCameraSpec {
    std::string relay;
    std::string view;
    std::string calibration;
    bool has_screen = true;
};

static bool safe_managed_camera_token(const std::string& value) {
    if (value.empty() || value.size() > 80) return false;
    for (const unsigned char character : value) {
        if (!(std::isalnum(character) || character == '_' || character == '-')) return false;
    }
    return true;
}

static std::vector<ManagedCameraSpec> load_camera_manifest(const std::string& path) {
    if (path.empty() || !fs::exists(path)) {
        throw std::runtime_error("camera manifest is missing: " + path);
    }
    std::ifstream input(path);
    if (!input) throw std::runtime_error("failed to open camera manifest: " + path);
    const json document = json::parse(input);
    if (!document.is_object() || document.value("schema_version", 0) != 1 ||
        !document.contains("cameras") || !document["cameras"].is_array()) {
        throw std::runtime_error("invalid camera manifest: " + path);
    }
    std::vector<ManagedCameraSpec> result;
    std::set<std::string> relays;
    std::set<std::string> views;
    std::set<std::string> calibrations;
    for (const auto& item : document["cameras"]) {
        if (!item.is_object()) throw std::runtime_error("invalid camera entry in manifest");
        const std::string relay = item.value("relay", "");
        const std::string view = item.value("view", "");
        const std::string calibration = item.value("calibration", "");
        const bool has_screen = item.value("has_screen", true);
        const fs::path calibration_path(calibration);
        if (!safe_managed_camera_token(relay) || view.empty() || view.size() > 80 ||
            calibration_path.filename() != calibration_path ||
            calibration_path.extension() != ".json" || calibration.size() > 128 ||
            !relays.insert(relay).second || !views.insert(view).second ||
            !calibrations.insert(calibration).second) {
            throw std::runtime_error("unsafe or duplicate camera manifest entry");
        }
        result.push_back({relay, view, calibration, has_screen});
    }
    if (result.empty()) throw std::runtime_error("camera manifest has no enabled cameras");
    return result;
}

static std::vector<cv::Point2f> parse_polygon(const json& arr, float sx, float sy) {
    std::vector<cv::Point2f> poly;
    if (!arr.is_array()) return poly;
    for (const auto& p : arr) {
        if (!p.is_array() || p.size() < 2) continue;
        poly.emplace_back(p[0].get<float>() * sx, p[1].get<float>() * sy);
    }
    return poly;
}

static bool looks_like_polygon_points(const json& arr) {
    return arr.is_array() && !arr.empty()
           && arr[0].is_array() && arr[0].size() >= 2
           && arr[0][0].is_number() && arr[0][1].is_number();
}

static void append_zone_from_json(std::vector<Zone>& zones, const json& value,
                                  const std::string& default_name, float sx, float sy) {
    Zone z;
    z.name = default_name;
    z.weight = 1.0f;
    if (value.is_object()) {
        z.name = value.value("name", default_name);
        z.weight = value.value("weight", 1.0f);
        if (value.contains("polygon")) {
            z.polygon = parse_polygon(value["polygon"], sx, sy);
        } else if (value.contains("points")) {
            z.polygon = parse_polygon(value["points"], sx, sy);
        }
    } else if (looks_like_polygon_points(value)) {
        z.polygon = parse_polygon(value, sx, sy);
    }
    if (!z.polygon.empty()) zones.push_back(std::move(z));
}

static std::vector<Zone> parse_zone_field(const json& data, const char* plural_key, const char* singular_key,
                                          const std::string& default_name, float sx, float sy) {
    std::vector<Zone> zones;
    if (data.contains(plural_key)) {
        const auto& v = data[plural_key];
        if (looks_like_polygon_points(v) || v.is_object()) {
            append_zone_from_json(zones, v, default_name, sx, sy);
        } else if (v.is_array()) {
            int idx = 1;
            for (const auto& item : v) {
                append_zone_from_json(zones, item, default_name + "_" + std::to_string(idx++), sx, sy);
            }
        }
    }
    if (data.contains(singular_key)) {
        append_zone_from_json(zones, data[singular_key], default_name, sx, sy);
    }
    return zones;
}

static void update_params_from_json(ScreenParams& p, const json& params) {
    if (!params.is_object()) return;
    auto getf = [&](const char* key, float& dst) {
        if (params.contains(key) && params[key].is_number()) dst = params[key].get<float>();
    };
    auto geti = [&](const char* key, int& dst) {
        if (params.contains(key) && params[key].is_number_integer()) dst = params[key].get<int>();
    };
    auto getb = [&](const char* key, bool& dst) {
        if (params.contains(key) && params[key].is_boolean()) dst = params[key].get<bool>();
    };
    getf("person_expand_x", p.person_expand_x);
    getf("person_expand_y", p.person_expand_y);
    geti("near_zone_detect_interval", p.near_zone_detect_interval);
    getf("phone_valid_conf_person_roi", p.phone_valid_conf_person_roi);
    getf("phone_valid_conf_near_zone", p.phone_valid_conf_near_zone);
    getf("phone_valid_conf_floor", p.phone_valid_conf_floor);
    getf("alarm_raw_phone_confidence", p.alarm_raw_phone_confidence);
    getf("phone_max_person_area_ratio", p.phone_max_person_area_ratio);
    getf("phone_large_person_diag_ratio", p.phone_large_person_diag_ratio);
    getf("phone_large_confidence", p.phone_large_confidence);
    getf("screen_distance_min_px", p.screen_distance_min_px);
    getf("screen_distance_max_px", p.screen_distance_max_px);
    getf("angle_thresh", p.angle_thresh);
    getf("angle_relaxed_thresh", p.angle_relaxed_thresh);
    getf("hand_radius_ratio", p.hand_radius_ratio);
    getf("hand_radius_min", p.hand_radius_min);
    getf("corridor_width_ratio", p.corridor_width_ratio);
    getf("corridor_width_min", p.corridor_width_min);
    geti("person_state_window", p.person_state_window);
    geti("person_state_min_hits", p.person_state_min_hits);
    geti("handheld_suspect_min_hits", p.handheld_suspect_min_hits);
    getf("person_state_risk_threshold", p.person_state_risk_threshold);
    getb("enable_static_phone_suppression", p.enable_static_phone_suppression);
    getf("static_phone_window_seconds", p.static_phone_window_seconds);
    getf("static_window_seconds", p.static_phone_window_seconds);
    getf("static_phone_max_disp_ratio", p.static_phone_max_disp_ratio);
    getf("static_max_disp_ratio", p.static_phone_max_disp_ratio);
    getf("static_phone_risk_multiplier", p.static_phone_risk_multiplier);
    getf("static_risk_multiplier", p.static_phone_risk_multiplier);
    getf("static_phone_min_abs_disp_px", p.static_phone_min_abs_disp_px);
    getf("static_phone_min_bbox_iou", p.static_phone_min_bbox_iou);
    getf("static_phone_max_bbox_size_change_ratio", p.static_phone_max_bbox_size_change_ratio);
    getf("static_phone_lower_person_start_ratio", p.static_phone_lower_person_start_ratio);
    getf("static_phone_wrist_follow_min_motion_px", p.static_phone_wrist_follow_min_motion_px);
    getf("static_phone_wrist_follow_cosine", p.static_phone_wrist_follow_cosine);
}

static std::vector<ScreenConfig> load_calibration(const std::string& path,
                                                  int width,
                                                  int height,
                                                  bool preserve_invalid_screens = false) {
    if (path.empty() || !fs::exists(path)) return {};
    std::ifstream in(path);
    if (!in) throw std::runtime_error("failed to open calibration: " + path);
    json data = json::parse(in);
    float ref_w = static_cast<float>(width);
    float ref_h = static_cast<float>(height);
    if (data.contains("frame_size") && data["frame_size"].is_array() && data["frame_size"].size() >= 2) {
        ref_w = data["frame_size"][0].get<float>();
        ref_h = data["frame_size"][1].get<float>();
    } else {
        if (data.contains("image_width")) ref_w = data["image_width"].get<float>();
        if (data.contains("image_height")) ref_h = data["image_height"].get<float>();
    }
    const float sx = width / std::max(1.0f, ref_w);
    const float sy = height / std::max(1.0f, ref_h);
    std::vector<ScreenConfig> screens;
    ScreenParams base_params;
    update_params_from_json(base_params, data.value("params", json::object()));
    const std::vector<Zone> global_desk_zones =
        parse_zone_field(data, "desk_static_zones", "desk_static_zone", "desk_static_zone", sx, sy);
    if (!data.contains("screens") || !data["screens"].is_array()) return screens;
    int idx = 1;
    for (const auto& item : data["screens"]) {
        ScreenConfig sc;
        sc.params = base_params;
        sc.desk_static_zones = global_desk_zones;
        sc.screen_id = item.value("screen_id", item.value("id", "screen_" + std::to_string(idx)));
        update_params_from_json(sc.params, item.value("params", json::object()));
        if (item.contains("screen_poly")) {
            sc.screen_poly = parse_polygon(item["screen_poly"], sx, sy);
        } else if (item.contains("bbox_xyxy") && item["bbox_xyxy"].is_array() && item["bbox_xyxy"].size() >= 4) {
            const auto& b = item["bbox_xyxy"];
            const float x1 = b[0].get<float>() * sx, y1 = b[1].get<float>() * sy;
            const float x2 = b[2].get<float>() * sx, y2 = b[3].get<float>() * sy;
            sc.screen_poly = {{x1, y1}, {x2, y1}, {x2, y2}, {x1, y2}};
        } else if (item.contains("points")) {
            sc.screen_poly = parse_polygon(item["points"], sx, sy);
        }
        sc.near_zone = item.contains("near_zone") ? parse_polygon(item["near_zone"], sx, sy) : std::vector<cv::Point2f>{};
        for (const auto& z : item.value("danger_zones", json::array())) {
            Zone dz;
            dz.name = z.value("name", "danger");
            dz.weight = z.value("weight", 1.0f);
            dz.polygon = parse_polygon(z.value("polygon", json::array()), sx, sy);
            if (!dz.polygon.empty()) sc.danger_zones.push_back(std::move(dz));
        }
        for (const auto& z : item.value("ignore_zones", json::array())) {
            Zone iz;
            iz.name = z.value("name", "ignore");
            iz.weight = 0.0f;
            iz.polygon = parse_polygon(z.value("polygon", json::array()), sx, sy);
            if (!iz.polygon.empty()) sc.ignore_zones.push_back(std::move(iz));
        }
        auto local_desk_zones = parse_zone_field(item, "desk_static_zones", "desk_static_zone", "desk_static_zone", sx, sy);
        sc.desk_static_zones.insert(sc.desk_static_zones.end(), local_desk_zones.begin(), local_desk_zones.end());
        if (sc.near_zone.empty() && !sc.screen_poly.empty()) {
            const Rect sb = bbox_from_polygon(sc.screen_poly);
            const Rect near = clamp_rect(Rect{sb.x1 - sb.w() * 1.2f, sb.y1 - sb.h() * 0.9f,
                                              sb.x2 + sb.w() * 1.2f, sb.y2 + sb.h() * 0.9f}, width, height);
            sc.near_zone = {{near.x1, near.y1}, {near.x2, near.y1}, {near.x2, near.y2}, {near.x1, near.y2}};
        }
        if (preserve_invalid_screens || !sc.screen_poly.empty()) screens.push_back(std::move(sc));
        ++idx;
    }
    return screens;
}

static void validate_live_calibration(const std::string& view,
                                      const std::string& calibration_name,
                                      const fs::path& calibration_path,
                                      const std::vector<ScreenConfig>& screens) {
    if (calibration_name.empty() || calibration_path.empty() || !fs::exists(calibration_path)) {
        throw std::runtime_error("live calibration missing for view " + view + ": " +
                                 calibration_path.string());
    }
    std::ifstream input(calibration_path);
    if (!input) throw std::runtime_error("failed to open calibration: " + calibration_path.string());
    const json data = json::parse(input);
    const bool explicit_screens_array = data.contains("screens") && data["screens"].is_array();
    if (screens.empty()) {
        throw std::runtime_error("live calibration has no screens for view " + view + ": " +
                                 calibration_path.string());
    }
    for (const auto& screen : screens) {
        bool finite = screen.screen_poly.size() >= 3;
        for (const auto& point : screen.screen_poly) {
            finite = finite && std::isfinite(point.x) && std::isfinite(point.y);
        }
        const double area = finite ? std::abs(cv::contourArea(screen.screen_poly)) : 0.0;
        if (!finite || area <= 1.0) {
            throw std::runtime_error("live calibration has invalid screen polygon for view " + view +
                                     " screen " + screen.screen_id + ": " +
                                     calibration_path.string());
        }
    }
    const auto activation = jiankong::custom_pipeline::classify_live_calibration(
        calibration_name, explicit_screens_array,
        explicit_screens_array ? data["screens"].size() : 0, true);
    if (activation != jiankong::custom_pipeline::ScreenActivation::active) {
        throw std::runtime_error("live calibration policy rejected active view " + view + ": " +
                                 calibration_path.string());
    }
}

static void validate_explicit_zero_screen_calibration(const std::string& view,
                                                      const fs::path& calibration_path) {
    if (calibration_path.empty() || !fs::exists(calibration_path)) {
        throw std::runtime_error("live calibration missing for screen-inactive view " + view + ": " +
                                 calibration_path.string());
    }
    std::ifstream input(calibration_path);
    if (!input) {
        throw std::runtime_error("failed to open screen-inactive calibration: " +
                                 calibration_path.string());
    }
    const json data = json::parse(input);
    const bool explicit_screens_array = data.contains("screens") && data["screens"].is_array();
    const auto activation = jiankong::custom_pipeline::classify_live_calibration(
        "camera_01_screen_calibration_v21.json", explicit_screens_array,
        explicit_screens_array ? data["screens"].size() : 0, true);
    if (activation != jiankong::custom_pipeline::ScreenActivation::inactive) {
        throw std::runtime_error("screen-inactive calibration must explicitly contain zero screens for view " +
                                 view + ": " + calibration_path.string());
    }
}

static jiankong::custom_pipeline::SourceReaderMetrics source_reader_metrics_for(
    const std::vector<jiankong::custom_pipeline::SourceReaderMetrics>& metrics,
    std::size_t source_id) {
    const auto it = std::find_if(metrics.begin(), metrics.end(), [source_id](const auto& value) {
        return value.source_id == source_id;
    });
    return it == metrics.end() ? jiankong::custom_pipeline::SourceReaderMetrics{} : *it;
}

static void draw_rect(cv::Mat& frame,
                      const Rect& r,
                      const cv::Scalar& color,
                      int thickness,
                      const std::string& label = "",
                      double font_scale = 0.45,
                      bool label_inside = false) {
    cv::Rect rr(cv::Point(static_cast<int>(std::round(r.x1)), static_cast<int>(std::round(r.y1))),
                cv::Point(static_cast<int>(std::round(r.x2)), static_cast<int>(std::round(r.y2))));
    rr &= cv::Rect(0, 0, frame.cols, frame.rows);
    if (rr.width <= 0 || rr.height <= 0) return;
    cv::rectangle(frame, rr, color, thickness, cv::LINE_AA);
    if (!label.empty()) {
        int baseline = 0;
        cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, font_scale, 1, &baseline);
        int x = std::max(0, rr.x);
        int y = label_inside ? rr.y + text_size.height + 3 : std::max(text_size.height + 2, rr.y - 3);
        if (y + baseline + 2 > frame.rows) y = frame.rows - baseline - 2;
        cv::rectangle(frame,
                      cv::Rect(x, y - text_size.height - 2, std::min(text_size.width + 4, frame.cols - x), text_size.height + baseline + 3),
                      color,
                      cv::FILLED);
        cv::putText(frame, label, cv::Point(x + 2, y), cv::FONT_HERSHEY_SIMPLEX, font_scale, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    }
}

static void draw_poly(cv::Mat& frame,
                      const std::vector<cv::Point2f>& poly,
                      const cv::Scalar& color,
                      int thickness,
                      const std::string& label = "",
                      double font_scale = 0.34) {
    if (poly.size() < 2) return;
    std::vector<cv::Point> pts;
    pts.reserve(poly.size());
    for (const auto& p : poly) {
        pts.emplace_back(static_cast<int>(std::round(p.x)), static_cast<int>(std::round(p.y)));
    }
    const std::vector<std::vector<cv::Point>> contours{pts};
    cv::polylines(frame, contours, true, color, thickness, cv::LINE_AA);
    if (!label.empty()) {
        const Rect b = bbox_from_polygon(poly);
        draw_rect(frame, Rect{b.x1, b.y1, b.x1 + 1.0f, b.y1 + 1.0f}, color, 1, label, font_scale, false);
    }
}

static json rect_to_json(const Rect& r) {
    return json::array({r.x1, r.y1, r.x2, r.y2});
}

static json polygon_to_json(const std::vector<cv::Point2f>& poly) {
    json out = json::array();
    for (const auto& p : poly) {
        out.push_back(json::array({p.x, p.y}));
    }
    return out;
}

static void make_sample_indices(StreamState& s, double infer_fps, int max_samples) {
    const double duration = s.total_frames > 0 && s.native_fps > 0 ? s.total_frames / s.native_fps : 0.0;
    const int sampled_total = duration > 0 ? static_cast<int>(std::floor(duration * infer_fps)) + 1 : s.total_frames;
    s.sample_indices.clear();
    int last = -1;
    for (int i = 0; i < sampled_total; ++i) {
        int idx = static_cast<int>(std::llround(i * s.native_fps / infer_fps));
        if (s.total_frames > 0) idx = std::min(s.total_frames - 1, idx);
        if (idx < 0 || idx == last) continue;
        s.sample_indices.push_back(idx);
        last = idx;
        if (max_samples > 0 && static_cast<int>(s.sample_indices.size()) >= max_samples) break;
    }
}

static bool read_next_sample(StreamState& s, cv::Mat& frame) {
    if (s.done || s.sample_cursor >= s.sample_indices.size()) {
        s.done = true;
        return false;
    }
    const int target = s.sample_indices[s.sample_cursor];
    while (s.raw_cursor < target) {
        if (!s.cap.grab()) {
            s.done = true;
            return false;
        }
        ++s.raw_cursor;
    }
    if (!s.cap.read(frame)) {
        s.done = true;
        return false;
    }
    ++s.raw_cursor;
    ++s.sample_cursor;
    return true;
}

static LetterboxMeta preprocess_into(const cv::Mat& src, int size, float* dst, int batch_index, int batch_size) {
    (void)batch_size;
    LetterboxMeta meta = compute_letterbox_meta(src.cols, src.rows, size);
    const int new_w = std::max(1, static_cast<int>(std::round(src.cols * meta.scale)));
    const int new_h = std::max(1, static_cast<int>(std::round(src.rows * meta.scale)));

    cv::Mat canvas(size, size, CV_8UC3, cv::Scalar(114, 114, 114));
    cv::Mat resized;
    cv::resize(src, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
    resized.copyTo(canvas(cv::Rect(meta.pad_x, meta.pad_y, new_w, new_h)));

    const size_t image_area = static_cast<size_t>(size) * size;
    const size_t base = static_cast<size_t>(batch_index) * 3 * image_area;
    float* c0 = dst + base;
    float* c1 = c0 + image_area;
    float* c2 = c1 + image_area;
    for (int y = 0; y < size; ++y) {
        const cv::Vec3b* row = canvas.ptr<cv::Vec3b>(y);
        for (int x = 0; x < size; ++x) {
            const cv::Vec3b& bgr = row[x];
            const size_t off = static_cast<size_t>(y) * size + x;
            c0[off] = bgr[2] / 255.0f;
            c1[off] = bgr[1] / 255.0f;
            c2[off] = bgr[0] / 255.0f;
        }
    }
    return meta;
}

static Rect map_box(float cx, float cy, float w, float h, const LetterboxMeta& m, int out_w, int out_h) {
    Rect r;
    r.x1 = (cx - w * 0.5f - m.pad_x) / m.scale;
    r.y1 = (cy - h * 0.5f - m.pad_y) / m.scale;
    r.x2 = (cx + w * 0.5f - m.pad_x) / m.scale;
    r.y2 = (cy + h * 0.5f - m.pad_y) / m.scale;
    return clamp_rect(r, out_w, out_h);
}

static std::vector<Det> nms(std::vector<Det> dets, float iou_thr, int max_det) {
    std::sort(dets.begin(), dets.end(), [](const Det& a, const Det& b) { return a.conf > b.conf; });
    std::vector<Det> keep;
    keep.reserve(std::min<int>(max_det, dets.size()));
    for (const auto& d : dets) {
        bool ok = true;
        for (const auto& k : keep) {
            if (iou(d.box, k.box) > iou_thr) {
                ok = false;
                break;
            }
        }
        if (ok) keep.push_back(d);
        if (max_det > 0 && static_cast<int>(keep.size()) >= max_det) break;
    }
    return keep;
}

static int visible_pose_points(const Det& d, float kp_thr) {
    int visible = 0;
    for (int idx : {0, 5, 6, 7, 8, 9, 10}) {
        if (d.kpts[idx][2] >= kp_thr) ++visible;
    }
    return visible;
}

static float person_quality(const Det& d, float kp_thr) {
    return d.conf + visible_pose_points(d, kp_thr) * 0.03f + std::log1p(std::max(1.0f, d.box.area())) * 0.01f;
}

static bool duplicate_person_box(const Det& a, const Det& b) {
    const float ov = iou(a.box, b.box);
    const float contain = containment_ratio(a.box, b.box);
    const float dist = center_dist(a.box, b.box);
    const float center_limit = std::max(35.0f, std::min(rect_diag(a.box), rect_diag(b.box)) * 0.35f);
    if (ov >= 0.55f) return true;
    if (contain >= 0.82f && dist <= center_limit) return true;
    if (ov >= 0.35f && dist <= center_limit * 0.70f) return true;
    return false;
}

static std::vector<Det> dedupe_people(std::vector<Det> people, float kp_thr) {
    std::sort(people.begin(), people.end(), [kp_thr](const Det& a, const Det& b) {
        return person_quality(a, kp_thr) > person_quality(b, kp_thr);
    });
    std::vector<Det> keep;
    keep.reserve(people.size());
    for (const auto& person : people) {
        bool dup = false;
        for (const auto& kept : keep) {
            if (duplicate_person_box(person, kept)) {
                dup = true;
                break;
            }
        }
        if (!dup) keep.push_back(person);
    }
    std::sort(keep.begin(), keep.end(), [](const Det& a, const Det& b) {
        if (std::abs(a.box.x1 - b.box.x1) > 4.0f) return a.box.x1 < b.box.x1;
        return a.box.y1 < b.box.y1;
    });
    return keep;
}

static bool phone_hits_roi(const Det& phone, const Rect& roi) {
    const float cx = (phone.box.x1 + phone.box.x2) * 0.5f;
    const float cy = (phone.box.y1 + phone.box.y2) * 0.5f;
    return point_in_rect(cx, cy, roi) || iou(phone.box, roi) > 0.01f;
}

static bool has_kpt(const Det& person, int idx, float kp_thr) {
    return idx >= 0 && idx < 17 && person.kpts[idx][2] >= kp_thr;
}

static cv::Point2f kpt_point(const Det& person, int idx) {
    return cv::Point2f(person.kpts[idx][0], person.kpts[idx][1]);
}

static float person_scale(const Det& person, float kp_thr) {
    float shoulder_width = 0.0f;
    if (has_kpt(person, 5, kp_thr) && has_kpt(person, 6, kp_thr)) {
        shoulder_width = point_dist(kpt_point(person, 5), kpt_point(person, 6));
    }
    return std::max({person.box.w(), person.box.h(), shoulder_width * 3.0f, 80.0f});
}

static cv::Point2f phone_center(const Det& phone) {
    return cv::Point2f(rect_cx(phone.box), rect_cy(phone.box));
}

static bool nearest_wrist_point(const Det& phone, const Det& person, float kp_thr, cv::Point2f& wrist_out) {
    const cv::Point2f pc = phone_center(phone);
    bool found = false;
    float best = std::numeric_limits<float>::max();
    for (int idx : {9, 10}) {
        if (!has_kpt(person, idx, kp_thr)) continue;
        const cv::Point2f w = kpt_point(person, idx);
        const float d = point_dist(pc, w);
        if (d < best) {
            best = d;
            wrist_out = w;
            found = true;
        }
    }
    return found;
}

static bool point_in_zones(const cv::Point2f& p, const std::vector<Zone>& zones) {
    for (const auto& z : zones) {
        if (point_in_polygon(p, z.polygon)) return true;
    }
    return false;
}

static cv::Point2f person_anchor(const Det& person, float kp_thr) {
    if (has_kpt(person, 5, kp_thr) && has_kpt(person, 6, kp_thr)) {
        const auto l = kpt_point(person, 5);
        const auto r = kpt_point(person, 6);
        return cv::Point2f((l.x + r.x) * 0.5f, (l.y + r.y) * 0.5f);
    }
    return cv::Point2f(rect_cx(person.box), rect_cy(person.box));
}

static Rect expand_rect_rule(const Rect& r, float expand_x, float expand_y, int width, int height) {
    const float dx = r.w() * expand_x;
    const float dy = r.h() * expand_y;
    return clamp_rect(Rect{r.x1 - dx, r.y1 - dy, r.x2 + dx, r.y2 + dy}, width, height);
}

static Rect corridor_bbox(const cv::Point2f& start, const cv::Point2f& end, float corridor_width, int width, int height) {
    return clamp_rect(Rect{std::min(start.x, end.x) - corridor_width * 0.5f,
                           std::min(start.y, end.y) - corridor_width * 0.5f,
                           std::max(start.x, end.x) + corridor_width * 0.5f,
                           std::max(start.y, end.y) + corridor_width * 0.5f},
                      width, height);
}

static bool person_related_to_screen(const Det& person, const ScreenConfig& screen, int width, int height, float kp_thr) {
    if (screen.screen_poly.empty()) return false;
    const float scale = person_scale(person, kp_thr);
    const Rect person_roi = expand_rect_rule(person.box, screen.params.person_expand_x, screen.params.person_expand_y, width, height);
    const Rect near_bbox = bbox_from_polygon(screen.near_zone);
    if (boxes_intersect(person_roi, near_bbox)) return true;
    return box_to_poly_distance(person.box, screen.screen_poly) <= scale * 0.50f;
}

static std::pair<float, std::string> static_zone_score(const Det& phone, const ScreenConfig& screen) {
    const cv::Point2f c = phone_center(phone);
    for (const auto& z : screen.ignore_zones) {
        if (point_in_polygon(c, z.polygon)) return {0.0f, "ignore:" + z.name};
    }
    for (const auto& z : screen.danger_zones) {
        if (point_in_polygon(c, z.polygon)) return {z.weight, "danger:" + z.name};
    }
    if (point_in_polygon(c, screen.near_zone)) return {0.5f, "near_zone"};
    const Rect screen_box = bbox_from_polygon(screen.screen_poly);
    const float screen_dist = point_to_rect_distance(c, screen_box);
    const float screen_distance_min_px = std::max(0.0f, screen.params.screen_distance_min_px);
    const float screen_distance_max_px = std::max(screen_distance_min_px, screen.params.screen_distance_max_px);
    const float screen_distance_base = std::max(rect_diag(phone.box) * 3.0f, screen_distance_min_px);
    const float screen_distance_threshold = clampf(screen_distance_base, screen_distance_min_px, screen_distance_max_px);
    if (screen_dist <= screen_distance_threshold) {
        return {0.25f, "screen_distance"};
    }
    return {0.0f, "outside"};
}

static float normalize_conf(float conf, float low, float high) {
    return clampf((conf - low) / std::max(1e-6f, high - low), 0.0f, 1.0f);
}

static float phone_reliability_score(const Det& phone, float required_conf) {
    const float conf_score = normalize_conf(phone.conf, required_conf, std::max(required_conf + 0.25f, 0.75f));
    const float aspect = phone.box.w() / std::max(1.0f, phone.box.h());
    const float aspect_score = (aspect >= 0.25f && aspect <= 4.0f) ? 1.0f : 0.5f;
    const float size_score = phone.box.area() >= 80.0f ? 1.0f : clampf(phone.box.area() / 80.0f, 0.0f, 0.7f);
    return clampf(0.70f * conf_score + 0.20f * size_score + 0.10f * aspect_score, 0.0f, 1.0f);
}

struct PhonePersonGeometryDecision {
    bool valid = false;
    float area_ratio = 0.0f;
    float diag_ratio = 0.0f;
};

static PhonePersonGeometryDecision phone_person_geometry_gate(
    const Det& phone, const Det& person, const ScreenParams& params) {
    PhonePersonGeometryDecision decision;
    const float person_width = std::max(1.0f, person.box.w());
    const float person_height = std::max(1.0f, person.box.h());
    decision.area_ratio = phone.box.area() / (person_width * person_height);
    decision.diag_ratio = rect_diag(phone.box) / person_height;
    const bool plausible_area = decision.area_ratio <= params.phone_max_person_area_ratio;
    const bool plausible_large_box =
        decision.diag_ratio <= params.phone_large_person_diag_ratio ||
        phone.conf >= params.phone_large_confidence;
    decision.valid = plausible_area && plausible_large_box;
    return decision;
}

static float hand_link_score(const Det& phone, const Det& person, const ScreenConfig&, float kp_thr) {
    std::vector<cv::Point2f> wrists;
    if (has_kpt(person, 9, kp_thr)) wrists.push_back(kpt_point(person, 9));
    if (has_kpt(person, 10, kp_thr)) wrists.push_back(kpt_point(person, 10));
    if (wrists.empty()) return 0.0f;
    const float hand_thresh = std::max(rect_diag(phone.box) * 0.85f, person_scale(person, kp_thr) * 0.10f);
    const cv::Point2f pc = phone_center(phone);
    float best = std::numeric_limits<float>::max();
    for (const auto& w : wrists) best = std::min(best, point_dist(pc, w));
    if (best <= hand_thresh) return 1.0f;
    if (best <= hand_thresh * 1.5f) return 0.5f;
    return 0.0f;
}

static std::tuple<float, Rect, Rect, bool> person_zone_score(const Det& phone, const Det& person,
                                                             const ScreenConfig& screen, int width, int height,
                                                             float kp_thr) {
    const float scale = person_scale(person, kp_thr);
    const Rect person_roi = expand_rect_rule(person.box, screen.params.person_expand_x, screen.params.person_expand_y, width, height);
    const Rect screen_box = bbox_from_polygon(screen.screen_poly);
    const cv::Point2f screen_center(rect_cx(screen_box), rect_cy(screen_box));
    const float cw = std::max(scale * screen.params.corridor_width_ratio, screen.params.corridor_width_min);
    const Rect corridor = corridor_bbox(person_anchor(person, kp_thr), screen_center, cw, width, height);
    const cv::Point2f c = phone_center(phone);
    const float person_box_score = point_in_rect(c.x, c.y, person_roi) ? 0.8f : 0.0f;
    const float wrist_score = hand_link_score(phone, person, screen, kp_thr);
    const bool corridor_hit = point_in_rect(c.x, c.y, corridor);
    const float corridor_score = corridor_hit ? 0.7f : 0.0f;
    return {std::max({person_box_score, wrist_score, corridor_score}), person_roi, corridor, corridor_hit};
}

static std::tuple<float, float, bool, std::string> aim_score(const Det& phone, const Det& person,
                                                             const ScreenConfig& screen, float static_score,
                                                             float hand_score, float kp_thr) {
    const Rect target_box = bbox_from_polygon(screen.screen_poly);
    const cv::Point2f screen_center(rect_cx(target_box), rect_cy(target_box));
    const cv::Point2f pc = phone_center(phone);
    float best_angle = 180.0f;
    bool best_ray_hit = false;
    std::string best_reason = "none";
    auto consider = [&](const std::string& reason, const cv::Point2f& start, const cv::Point2f& vec) {
        const float a = angle_deg(vec, cv::Point2f(screen_center.x - start.x, screen_center.y - start.y));
        const bool hit = ray_hits_box(start, vec, target_box, 4.0f);
        if (hit || a < best_angle) {
            best_angle = a;
            best_ray_hit = hit;
            best_reason = reason;
        }
    };
    for (auto side : {0, 1}) {
        const int shoulder_idx = side == 0 ? 5 : 6;
        const int elbow_idx = side == 0 ? 7 : 8;
        const int wrist_idx = side == 0 ? 9 : 10;
        const std::string prefix = side == 0 ? "L" : "R";
        if (!has_kpt(person, wrist_idx, kp_thr)) continue;
        const auto wrist = kpt_point(person, wrist_idx);
        if (has_kpt(person, elbow_idx, kp_thr)) {
            const auto elbow = kpt_point(person, elbow_idx);
            consider(prefix + "_forearm", elbow, cv::Point2f(wrist.x - elbow.x, wrist.y - elbow.y));
        }
        if (has_kpt(person, shoulder_idx, kp_thr)) {
            const auto shoulder = kpt_point(person, shoulder_idx);
            consider(prefix + "_upper_to_wrist", shoulder, cv::Point2f(wrist.x - shoulder.x, wrist.y - shoulder.y));
        }
        consider(prefix + "_wrist_phone", wrist, cv::Point2f(pc.x - wrist.x, pc.y - wrist.y));
    }
    if (best_ray_hit || best_angle <= screen.params.angle_thresh) return {1.0f, best_angle, best_ray_hit, best_reason};
    if (best_angle <= screen.params.angle_relaxed_thresh && phone.conf >= 0.55f && hand_score >= 0.8f && static_score >= 0.8f) {
        return {0.7f, best_angle, best_ray_hit, "relaxed:" + best_reason};
    }
    return {0.0f, best_angle, best_ray_hit, best_reason};
}

static float pose_support_score(const Det& phone, const Det& person, float aim, float hand, float kp_thr) {
    (void)kp_thr;
    const float rel_y = (rect_cy(phone.box) - person.box.y1) / std::max(1.0f, person.box.h());
    const float upper_body_score = clampf((0.90f - rel_y) / 0.65f, 0.0f, 1.0f);
    std::vector<cv::Point2f> wrists;
    if (has_kpt(person, 9, 0.0f)) wrists.push_back(kpt_point(person, 9));
    if (has_kpt(person, 10, 0.0f)) wrists.push_back(kpt_point(person, 10));
    float raised_score = 0.0f;
    if (!wrists.empty()) {
        float min_y = wrists[0].y;
        for (const auto& w : wrists) min_y = std::min(min_y, w.y);
        raised_score = min_y <= person.box.y1 + person.box.h() * 0.78f ? 1.0f : 0.35f;
    }
    return clampf(0.45f * upper_body_score + 0.35f * raised_score + 0.15f * hand + 0.05f * aim, 0.0f, 1.0f);
}

static float risk_score(float phone_score, float hand_score, float screen_score, float pose_score, float temporal_score) {
    return clampf(0.35f * phone_score + 0.25f * screen_score + 0.20f * hand_score
                  + 0.10f * pose_score + 0.10f * temporal_score, 0.0f, 1.0f);
}

static std::string candidate_level(float risk) {
    if (risk < 0.40f) return "ignore";
    if (risk < 0.65f) return "weak";
    if (risk < 0.80f) return "normal";
    return "strong";
}

static bool is_handheld_phone_suspect_candidate(const CandidateEval& ev) {
    if (ev.track_id < 0 || ev.person_index < 0) return false;
    if (!ev.phone_geometry_valid) return false;
    if (ev.static_suppressed || ev.phone_static) return false;
    if (ev.reject_reason == "ignore_zone" || ev.reject_reason == "low_conf_for_source") return false;
    if (ev.phone.conf < 0.50f || ev.phone_score <= 0.0f) return false;
    const bool person_context = ev.person_match_score >= 0.28f || ev.phone_hand_score >= 0.50f;
    const bool hand_or_upper_roi = ev.phone_hand_score >= 0.50f || (ev.person_match_score >= 0.28f && ev.pose_score >= 0.45f);
    return person_context && hand_or_upper_roi && ev.risk_score >= 0.45f;
}

static float compute_temporal_score(const PersonTrackState* state, float risk_threshold) {
    if (!state || state->risk_history.empty()) return 0.0f;
    int hits = 0;
    float sum_risk = 0.0f;
    for (float r : state->risk_history) {
        if (r >= risk_threshold) ++hits;
        sum_risk += r;
    }
    int candidates = 0;
    for (int c : state->candidate_history) candidates += c ? 1 : 0;
    const float n = static_cast<float>(std::max<size_t>(1, state->risk_history.size()));
    return clampf(0.50f * (hits / n) + 0.35f * (sum_risk / n) + 0.15f * (candidates / n), 0.0f, 1.0f);
}

static int trailing_candidate_count(const std::deque<int>& values) {
    int count = 0;
    for (auto it = values.rbegin(); it != values.rend(); ++it) {
        if (!*it) break;
        ++count;
    }
    return count;
}

static void push_limited(std::deque<int>& q, int v, int max_len) {
    q.push_back(v);
    while (static_cast<int>(q.size()) > max_len) q.pop_front();
}

static void push_limited(std::deque<float>& q, float v, int max_len) {
    q.push_back(v);
    while (static_cast<int>(q.size()) > max_len) q.pop_front();
}

static void push_limited(std::deque<Rect>& q, const Rect& v, int max_len) {
    q.push_back(v);
    while (static_cast<int>(q.size()) > max_len) q.pop_front();
}

static jiankong::custom_pipeline::EvidenceState gated_hand_relation(
    const Det& phone, const Det& person) {
    using jiankong::custom_pipeline::EvidenceState;
    const cv::Point2f center = phone_center(phone);
    const float person_h = std::max(1.0f, person.box.h());
    const float true_distance = std::max(rect_diag(phone.box), person_h * 0.10f);
    const float false_distance = person_h * 0.18f;
    float nearest_distance = std::numeric_limits<float>::max();
    bool any_reliable_wrist = false;
    bool both_reliable_wrists = true;
    bool arm_overlap = false;

    for (int side = 0; side < 2; ++side) {
        const int elbow_idx = side == 0 ? 7 : 8;
        const int wrist_idx = side == 0 ? 9 : 10;
        const bool wrist_reliable = has_kpt(person, wrist_idx, 0.45f);
        both_reliable_wrists = both_reliable_wrists && wrist_reliable;
        if (!wrist_reliable) continue;
        any_reliable_wrist = true;
        const cv::Point2f wrist = kpt_point(person, wrist_idx);
        nearest_distance = std::min(nearest_distance, point_to_rect_distance(wrist, phone.box));
        if (has_kpt(person, elbow_idx, 0.35f)) {
            const float arm_distance = point_segment_distance(
                center, kpt_point(person, elbow_idx), wrist);
            arm_overlap = arm_overlap || arm_distance <= std::max(rect_diag(phone.box), person_h * 0.06f);
        }
    }

    if (!any_reliable_wrist) return EvidenceState::unknown;
    if (nearest_distance <= true_distance && (arm_overlap || nearest_distance <= person_h * 0.06f)) {
        return EvidenceState::true_value;
    }
    if (both_reliable_wrists && nearest_distance > false_distance && !arm_overlap) {
        return EvidenceState::false_value;
    }
    return EvidenceState::unknown;
}

static float cross_2d(const cv::Point2f& a, const cv::Point2f& b) {
    return a.x * b.y - a.y * b.x;
}

static bool ray_hits_polygon(const cv::Point2f& start, const cv::Point2f& direction,
                             const std::vector<cv::Point2f>& polygon) {
    if (polygon.size() < 3 || std::hypot(direction.x, direction.y) < 1e-3f) return false;
    for (size_t i = 0; i < polygon.size(); ++i) {
        const cv::Point2f a = polygon[i];
        const cv::Point2f edge = polygon[(i + 1) % polygon.size()] - a;
        const float denominator = cross_2d(direction, edge);
        if (std::abs(denominator) < 1e-6f) continue;
        const cv::Point2f delta = a - start;
        const float ray_t = cross_2d(delta, edge) / denominator;
        const float edge_t = cross_2d(delta, direction) / denominator;
        if (ray_t >= 0.0f && edge_t >= 0.0f && edge_t <= 1.0f) return true;
    }
    return false;
}

static jiankong::custom_pipeline::EvidenceState gated_screen_intent(
    const Det& phone, const Det& person, const ScreenConfig& screen,
    float& angle_out, bool& ray_hit_out,
    bool& exact_ray_hit_out, bool& corridor_ray_hit_out) {
    using jiankong::custom_pipeline::EvidenceState;
    angle_out = 180.0f;
    ray_hit_out = false;
    exact_ray_hit_out = false;
    corridor_ray_hit_out = false;
    const cv::Point2f center = phone_center(phone);
    float nearest_wrist = std::numeric_limits<float>::max();
    for (int side = 0; side < 2; ++side) {
        const int wrist_idx = side == 0 ? 9 : 10;
        if (!has_kpt(person, wrist_idx, 0.45f)) continue;
        const float distance = point_dist(center, kpt_point(person, wrist_idx));
        nearest_wrist = std::min(nearest_wrist, distance);
    }
    if (!std::isfinite(nearest_wrist)) return EvidenceState::unknown;

    const Rect screen_box = bbox_from_polygon(screen.screen_poly);
    const cv::Point2f screen_center(rect_cx(screen_box), rect_cy(screen_box));
    const float angle_thresh = std::min(
        clampf(screen.params.angle_thresh, 0.0f, 180.0f), 70.0f);
    const float angle_relaxed_thresh = clampf(
        std::min(std::max(angle_thresh, screen.params.angle_relaxed_thresh), 90.0f),
        angle_thresh, 90.0f);
    const float corridor_width = std::max(
        std::max(1.0f, person.box.h()) * std::max(0.0f, screen.params.corridor_width_ratio),
        std::max(0.0f, screen.params.corridor_width_min));
    const float corridor_half_width = corridor_width * 0.5f;
    const Rect corridor_target{
        screen_box.x1 - corridor_half_width,
        screen_box.y1 - corridor_half_width,
        screen_box.x2 + corridor_half_width,
        screen_box.y2 + corridor_half_width};

    // When the phone is held between both hands, choosing only the nearest wrist
    // can alternate left/right on adjacent frames and flip the direction by more
    // than 90 degrees. Evaluate both arms that are genuinely close to the phone,
    // then prefer direct/corridor screen evidence before the smaller angle.
    const float side_slack = std::max(rect_diag(phone.box), person.box.h() * 0.08f);
    int best_hit_rank = -1;
    bool any_arm_pair = false;
    for (int side = 0; side < 2; ++side) {
        const int elbow_idx = side == 0 ? 7 : 8;
        const int wrist_idx = side == 0 ? 9 : 10;
        if (!has_kpt(person, wrist_idx, 0.45f) ||
            !has_kpt(person, elbow_idx, 0.35f)) {
            continue;
        }
        const cv::Point2f wrist = kpt_point(person, wrist_idx);
        if (point_dist(center, wrist) > nearest_wrist + side_slack) continue;
        const cv::Point2f elbow = kpt_point(person, elbow_idx);
        const cv::Point2f forearm_direction = wrist - elbow;
        const cv::Point2f phone_direction = center - elbow;
        const float forearm_norm = std::hypot(forearm_direction.x, forearm_direction.y);
        const float phone_norm = std::hypot(phone_direction.x, phone_direction.y);
        if (phone_norm < 1e-3f) continue;
        any_arm_pair = true;

        // Fuse the forearm direction with the elbow-to-phone direction. A pose
        // keypoint can move several pixels on a small/occluded person, while the
        // detected phone can sit just beside the wrist.
        cv::Point2f direction = phone_direction;
        if (forearm_norm >= 1e-3f) {
            const cv::Point2f blended(
                forearm_direction.x / forearm_norm + phone_direction.x / phone_norm,
                forearm_direction.y / forearm_norm + phone_direction.y / phone_norm);
            const float blended_norm = std::hypot(blended.x, blended.y);
            if (blended_norm >= 1e-3f) {
                const float direction_scale = std::max(forearm_norm, phone_norm);
                direction = cv::Point2f(
                    blended.x / blended_norm * direction_scale,
                    blended.y / blended_norm * direction_scale);
            }
        }

        const float candidate_angle = angle_deg(direction, screen_center - elbow);
        const bool candidate_exact_hit =
            ray_hits_polygon(elbow, direction, screen.screen_poly);
        const bool candidate_corridor_hit = candidate_angle <= angle_thresh &&
            ray_hits_box(elbow, direction, corridor_target, 4.0f);
        const int hit_rank = candidate_exact_hit ? 2 : (candidate_corridor_hit ? 1 : 0);
        if (hit_rank > best_hit_rank ||
            (hit_rank == best_hit_rank && candidate_angle < angle_out)) {
            best_hit_rank = hit_rank;
            angle_out = candidate_angle;
            exact_ray_hit_out = candidate_exact_hit;
            corridor_ray_hit_out = candidate_corridor_hit;
        }
    }
    if (!any_arm_pair) return EvidenceState::unknown;
    ray_hit_out = exact_ray_hit_out || corridor_ray_hit_out;
    if (ray_hit_out || angle_out <= angle_thresh) return EvidenceState::true_value;
    if (angle_out <= angle_relaxed_thresh) return EvidenceState::unknown;
    return EvidenceState::false_value;
}

static jiankong::custom_pipeline::EvidenceState gated_person_association(
    const Det& phone, const Det& person,
    jiankong::custom_pipeline::EvidenceState hand_relation,
    int width, int height) {
    using jiankong::custom_pipeline::EvidenceState;
    if (phone.person_index < 0 || phone.person_index != person.person_index) {
        return EvidenceState::false_value;
    }
    if (hand_relation == EvidenceState::true_value) return EvidenceState::true_value;
    const Rect strict_roi = expand_rect_rule(person.box, 0.18f, 0.10f, width, height);
    const cv::Point2f center = phone_center(phone);
    return point_in_rect(center.x, center.y, strict_roi)
        ? EvidenceState::true_value : EvidenceState::false_value;
}

static void push_limited(std::deque<StaticPhoneObs>& q, const StaticPhoneObs& v,
                         int max_len) {
    q.push_back(v);
    while (static_cast<int>(q.size()) > max_len) q.pop_front();
}

static bool phone_in_lower_person_area(const CandidateEval& ev, const Det& person) {
    const cv::Point2f c = phone_center(ev.phone);
    const float margin_x = person.box.w() * 0.18f;
    const float lower_y = person.box.y1 + person.box.h() *
        clampf(ev.static_lower_person_start_ratio, 0.10f, 0.95f);
    return c.y >= lower_y && c.x >= person.box.x1 - margin_x &&
        c.x <= person.box.x2 + margin_x;
}

static float rect_size_change_ratio(const Rect& a, const Rect& b) {
    const float wa = std::max(1.0f, a.w());
    const float ha = std::max(1.0f, a.h());
    const float wb = std::max(1.0f, b.w());
    const float hb = std::max(1.0f, b.h());
    const float wr = std::abs(wa - wb) / std::max(wa, wb);
    const float hr = std::abs(ha - hb) / std::max(ha, hb);
    return std::max(wr, hr);
}

static void apply_legacy_static_phone_suppression(
    CandidateEval& ev, PersonTrackState& state, const Det& person,
    int frame_id, float infer_fps) {
    if (!ev.static_suppression_enabled || ev.track_id < 0 ||
        ev.person_index < 0 || ev.phone_score <= 0.0f) {
        return;
    }
    const float fps = std::max(1.0f, infer_fps);
    StaticPhoneObs obs;
    obs.frame_id = frame_id;
    obs.timestamp = frame_id / fps;
    obs.center = phone_center(ev.phone);
    obs.box = ev.phone.box;
    obs.person_box = person.box;
    obs.risk_score = ev.risk_score;
    obs.candidate = ev.accepted();
    obs.has_wrist = nearest_wrist_point(ev.phone, person, 0.0f, obs.wrist);
    const int max_history = std::max(
        8, static_cast<int>(std::ceil(
            std::max(2.0f, ev.static_window_seconds + 1.0f) * fps)) + 2);
    push_limited(state.static_phone_history, obs, max_history);

    std::vector<StaticPhoneObs> recent;
    const float window = std::max(0.1f, ev.static_window_seconds);
    const float cutoff = obs.timestamp - std::max(window, ev.static_window_seconds);
    for (const auto& h : state.static_phone_history) {
        if (h.timestamp >= cutoff) recent.push_back(h);
    }
    if (recent.size() < 2) return;

    float min_x = recent[0].center.x, max_x = recent[0].center.x;
    float min_y = recent[0].center.y, max_y = recent[0].center.y;
    float min_iou = 1.0f;
    float max_size_change = 0.0f;
    float max_gap = 0.0f;
    for (size_t i = 0; i < recent.size(); ++i) {
        min_x = std::min(min_x, recent[i].center.x);
        max_x = std::max(max_x, recent[i].center.x);
        min_y = std::min(min_y, recent[i].center.y);
        max_y = std::max(max_y, recent[i].center.y);
        min_iou = std::min(min_iou, iou(obs.box, recent[i].box));
        max_size_change = std::max(
            max_size_change, rect_size_change_ratio(obs.box, recent[i].box));
        if (i > 0) {
            max_gap = std::max(max_gap,
                recent[i].timestamp - recent[i - 1].timestamp);
        }
    }
    const float duration = recent.back().timestamp - recent.front().timestamp;
    const float motion = std::sqrt(
        (max_x - min_x) * (max_x - min_x) +
        (max_y - min_y) * (max_y - min_y));
    const float max_disp = std::max(
        ev.static_min_abs_disp_px,
        ev.static_max_disp_ratio * std::max(1.0f, person.box.h()));
    const bool continuous = duration >= window &&
        max_gap <= std::max(0.35f, window * 0.60f);
    const bool center_stable = motion <= max_disp;
    const bool bbox_stable = min_iou >= ev.static_min_bbox_iou &&
        max_size_change <= ev.static_max_bbox_size_change_ratio;
    const bool explicit_static_hint = ev.phone_in_desk_zone;
    const bool lower_location_hint = phone_in_lower_person_area(ev, person);

    const cv::Point2f phone_vec(
        recent.back().center.x - recent.front().center.x,
        recent.back().center.y - recent.front().center.y);
    const float phone_endpoint_motion = std::sqrt(
        phone_vec.x * phone_vec.x + phone_vec.y * phone_vec.y);
    const cv::Point2f first_person_center(
        (recent.front().person_box.x1 + recent.front().person_box.x2) * 0.5f,
        (recent.front().person_box.y1 + recent.front().person_box.y2) * 0.5f);
    const cv::Point2f last_person_center(
        (recent.back().person_box.x1 + recent.back().person_box.x2) * 0.5f,
        (recent.back().person_box.y1 + recent.back().person_box.y2) * 0.5f);
    const cv::Point2f person_vec(
        last_person_center.x - first_person_center.x,
        last_person_center.y - first_person_center.y);
    const float person_motion = std::sqrt(
        person_vec.x * person_vec.x + person_vec.y * person_vec.y);

    bool follows_wrist = false;
    float wrist_motion = 0.0f;
    if (recent.front().has_wrist && recent.back().has_wrist) {
        const cv::Point2f wrist_vec(
            recent.back().wrist.x - recent.front().wrist.x,
            recent.back().wrist.y - recent.front().wrist.y);
        wrist_motion = std::sqrt(
            wrist_vec.x * wrist_vec.x + wrist_vec.y * wrist_vec.y);
        const float denom = std::max(
            1e-6f, phone_endpoint_motion * wrist_motion);
        const float cosine =
            (phone_vec.x * wrist_vec.x + phone_vec.y * wrist_vec.y) / denom;
        const float wrist_min = std::max(
            ev.static_wrist_follow_min_motion_px, person.box.h() * 0.04f);
        follows_wrist = wrist_motion >= wrist_min &&
            phone_endpoint_motion >= ev.static_min_abs_disp_px &&
            phone_endpoint_motion >= wrist_motion * 0.35f &&
            cosine >= ev.static_wrist_follow_cosine;
    }
    const float independent_motion_min = std::max(
        ev.static_wrist_follow_min_motion_px, person.box.h() * 0.04f);
    const bool wrist_decoupled = recent.front().has_wrist &&
        recent.back().has_wrist && wrist_motion >= independent_motion_min &&
        phone_endpoint_motion <= max_disp;
    const bool person_decoupled = person_motion >= independent_motion_min &&
        phone_endpoint_motion <= max_disp;
    const bool motion_decoupled = wrist_decoupled || person_decoupled;
    const bool prolonged_lower_hint = lower_location_hint &&
        duration >= window * 2.0f;

    ev.phone_static_duration = duration;
    ev.phone_motion_px = motion;
    ev.phone_follow_wrist = follows_wrist;
    ev.phone_static = continuous && center_stable && bbox_stable &&
        !follows_wrist &&
        (motion_decoupled || explicit_static_hint || prolonged_lower_hint);
    if (!ev.phone_static) return;

    ev.static_suppressed = true;
    ev.static_risk_multiplier = clampf(
        ev.static_config_risk_multiplier, 0.0f, 1.0f);
    ev.risk_score = clampf(
        ev.risk_score * ev.static_risk_multiplier, 0.0f, 1.0f);
    ev.level = candidate_level(ev.risk_score);
    ev.reject_reason = "static_phone_suppressed";
    std::ostringstream oss;
    oss << "STATIC_PHONE_SUPPRESSED"
        << "|duration=" << std::fixed << std::setprecision(2)
        << ev.phone_static_duration
        << "|motion=" << ev.phone_motion_px
        << "|desk=" << (ev.phone_in_desk_zone ? 1 : 0)
        << "|lower_hint=" << (lower_location_hint ? 1 : 0)
        << "|follow_wrist=" << (ev.phone_follow_wrist ? 1 : 0)
        << "|motion_decoupled=" << (motion_decoupled ? 1 : 0)
        << "|person_motion=" << person_motion
        << "|wrist_motion=" << wrist_motion
        << "|risk=" << ev.risk_score;
    ev.candidate_reason = oss.str();
    state.static_phone_suppressed_hits += 1;
}

static void push_limited(std::deque<FormalSpatialSample>& q,
                         const FormalSpatialSample& v, int max_len) {
    q.push_back(v);
    while (static_cast<int>(q.size()) > max_len) q.pop_front();
}

static jiankong::custom_pipeline::SpatialRect spatial_rect(const Rect& box) {
    return jiankong::custom_pipeline::SpatialRect{box.x1, box.y1, box.x2, box.y2};
}

static jiankong::custom_pipeline::SpatialFrameObservation build_spatial_frame_observation(
    const Det& person, const std::vector<CandidateEval*>& person_evals, double time_sec) {
    jiankong::custom_pipeline::SpatialFrameObservation frame;
    frame.time_sec = time_sec;
    frame.person = spatial_rect(person.box);
    frame.phones.reserve(person_evals.size());

    const CandidateEval* wrist_reference = nullptr;
    for (const CandidateEval* ev : person_evals) {
        frame.phones.push_back(jiankong::custom_pipeline::SpatialPhoneCandidate{
            spatial_rect(ev->phone.box),
            ev->phone.conf,
            ev->risk_score,
            ev->accepted(),
            ev->fixed_template_match
                ? jiankong::custom_pipeline::FixedTemplateEvidence::matched
                : (ev->fixed_template_near
                    ? jiankong::custom_pipeline::FixedTemplateEvidence::mismatch
                    : jiankong::custom_pipeline::FixedTemplateEvidence::none),
            ev->fixed_template_score,
        });
        if (wrist_reference == nullptr || ev->phone.conf > wrist_reference->phone.conf) {
            wrist_reference = ev;
        }
    }
    if (wrist_reference != nullptr) {
        cv::Point2f wrist;
        frame.wrist_valid = nearest_wrist_point(wrist_reference->phone, person, 0.0f, wrist);
        frame.wrist_x = wrist.x;
        frame.wrist_y = wrist.y;
    }
    return frame;
}

static jiankong::custom_pipeline::SpatialStaticDecision observe_spatial_static_phone(
    PersonTrackState& state,
    const jiankong::custom_pipeline::SpatialFrameObservation& frame,
    bool enabled,
    jiankong::custom_pipeline::CameraStaticHotspotMap* static_hotspots,
    double hotspot_now,
    bool* hotspot_dirty) {
    if (!enabled) {
        jiankong::custom_pipeline::SpatialStaticDecision decision;
        decision.phase = jiankong::custom_pipeline::SpatialStaticPhase::handheld_or_moving;
        decision.replay_shadow = !state.static_shadow_history.empty();
        decision.reason = "disabled";
        state.spatial_static_policy.reset_short_state();
        state.last_static_decision = decision;
        return decision;
    }
    const float static_hotspot_score = 0.0f;
    float supporting_hotspot_score = static_hotspot_score;
    if (static_hotspots != nullptr && std::isfinite(hotspot_now)) {
        for (const auto& phone : frame.phones) {
            const float center_x = (phone.box.x1 + phone.box.x2) * 0.5f;
            const float center_y = (phone.box.y1 + phone.box.y2) * 0.5f;
            supporting_hotspot_score = std::max(
                supporting_hotspot_score,
                static_hotspots->score(center_x, center_y, hotspot_now));
        }
    }
    state.static_hotspot_score = std::isfinite(supporting_hotspot_score)
        ? std::max(0.0f, supporting_hotspot_score) : 0.0f;

    const auto previous_phase = state.last_static_decision.phase;
    state.last_static_decision = state.spatial_static_policy.observe(
        frame, supporting_hotspot_score);
    const auto& decision = state.last_static_decision;
    if (static_hotspots == nullptr || !std::isfinite(hotspot_now) ||
        !decision.primary_candidate_index.has_value() ||
        *decision.primary_candidate_index >= frame.phones.size() ||
        decision.primary_cluster_centers.empty()) {
        return decision;
    }

    const auto& current_primary = frame.phones[*decision.primary_candidate_index];
    const float primary_center_x = (current_primary.box.x1 + current_primary.box.x2) * 0.5f;
    const float primary_center_y = (current_primary.box.y1 + current_primary.box.y2) * 0.5f;
    if (decision.phase == jiankong::custom_pipeline::SpatialStaticPhase::suppressed &&
        previous_phase != jiankong::custom_pipeline::SpatialStaticPhase::suppressed) {
        static_hotspots->confirm(primary_center_x, primary_center_y, hotspot_now);
        if (hotspot_dirty != nullptr) *hotspot_dirty = true;
    } else if (decision.phase ==
                   jiankong::custom_pipeline::SpatialStaticPhase::handheld_or_moving &&
               (std::strcmp(decision.reason, "phone_moving") == 0 ||
                std::strcmp(decision.reason, "follows_wrist") == 0 ||
                std::strcmp(decision.reason, "fixed_template_mismatch") == 0)) {
        static_hotspots->penalize(primary_center_x, primary_center_y, hotspot_now);
        if (hotspot_dirty != nullptr) *hotspot_dirty = true;
    }
    return state.last_static_decision;
}

static const char* spatial_static_context_reason(const char* reason) {
    if (reason != nullptr &&
        (std::strcmp(reason, "pending_static_context") == 0 ||
         std::strcmp(reason, "pending_dropout") == 0)) {
        return "pending";
    }
    static constexpr const char* allowed[] = {
        "disabled", "not_primary", "observed", "pending",
        "motion_decoupled", "hotspot"
    };
    if (reason != nullptr) {
        for (const char* value : allowed) {
            if (std::strcmp(reason, value) == 0) return value;
        }
    }
    return "observed";
}

static const char* spatial_static_exit_reason(const char* reason) {
    static constexpr const char* allowed[] = {
        "phone_moving", "follows_wrist", "static_context_timeout",
        "preconfirm_disappearance", "suppressed_disappearance",
        "invalid_observation", "fixed_template_mismatch"
    };
    if (reason != nullptr) {
        for (const char* value : allowed) {
            if (std::strcmp(reason, value) == 0) return value;
        }
    }
    return "none";
}

static void update_spatial_static_diagnostics(
    PersonTrackState& state, const std::vector<CandidateEval*>& person_evals,
    const jiankong::custom_pipeline::SpatialStaticDecision& decision,
    double frame_time_sec) {
    const bool pending =
        decision.phase == jiankong::custom_pipeline::SpatialStaticPhase::pending;
    if (pending) {
        if (!std::isfinite(state.static_pending_started_at)) {
            state.static_pending_started_at = frame_time_sec;
        }
        state.static_pending_duration = std::max(
            0.0, frame_time_sec - state.static_pending_started_at);
    } else {
        state.static_pending_started_at = std::numeric_limits<double>::quiet_NaN();
        state.static_pending_duration = 0.0;
    }
    const char* bounded_context_reason =
        spatial_static_context_reason(decision.reason);
    const char* bounded_exit_reason = spatial_static_exit_reason(decision.reason);
    for (size_t candidate_index = 0; candidate_index < person_evals.size();
         ++candidate_index) {
        CandidateEval* ev = person_evals[candidate_index];
        if (ev == nullptr) continue;
        const bool is_current_primary =
            decision.primary_candidate_index.has_value() &&
            candidate_index == *decision.primary_candidate_index;
        if (!is_current_primary) {
            ev->static_cluster_samples = 0;
            ev->static_detection_ratio = 0.0f;
            ev->static_center_spread_px = 0.0f;
            ev->static_bbox_iou_median = 0.0f;
            ev->static_pending = false;
            ev->static_pending_duration = 0.0;
            ev->static_hotspot_score = 0.0f;
            ev->static_context_reason = "not_primary";
            ev->static_shadow_hits = 0;
            ev->static_suppressed = false;
            ev->static_exit_reason = "none";
            continue;
        }
        ev->static_cluster_samples = std::max(0, decision.metrics.sample_count);
        ev->static_detection_ratio = std::isfinite(decision.metrics.detection_ratio)
            ? clampf(decision.metrics.detection_ratio, 0.0f, 1.0f) : 0.0f;
        ev->static_center_spread_px = std::isfinite(decision.metrics.center_spread_px)
            ? std::max(0.0f, decision.metrics.center_spread_px) : 0.0f;
        ev->static_bbox_iou_median = std::isfinite(decision.metrics.bbox_iou_median)
            ? clampf(decision.metrics.bbox_iou_median, 0.0f, 1.0f) : 0.0f;
        ev->static_pending = pending;
        ev->static_pending_duration = state.static_pending_duration;
        ev->static_context_reason = bounded_context_reason;
        ev->static_shadow_hits = static_cast<int>(state.static_shadow_history.size());
        ev->static_suppressed = decision.phase ==
            jiankong::custom_pipeline::SpatialStaticPhase::suppressed;
        ev->static_exit_reason = bounded_exit_reason;
    }
}

static CandidateEval* best_formal_candidate(const std::vector<CandidateEval*>& person_evals,
                                            const CandidateEval* held_candidate) {
    CandidateEval* best = nullptr;
    for (CandidateEval* ev : person_evals) {
        if (ev == nullptr || ev == held_candidate) continue;
        if (best == nullptr || ev->risk_score > best->risk_score) best = ev;
    }
    return best;
}

static void append_formal_history(PersonTrackState& state, float risk, bool candidate,
                                  bool handheld_candidate,
                                  const std::optional<jiankong::custom_pipeline::SpatialRect>& spatial_box,
                                  int window_size) {
    FormalSpatialSample spatial_sample;
    if (spatial_box.has_value() && spatial_box->x2 > spatial_box->x1 &&
        spatial_box->y2 > spatial_box->y1) {
        spatial_sample.center_x = (spatial_box->x1 + spatial_box->x2) * 0.5f;
        spatial_sample.center_y = (spatial_box->y1 + spatial_box->y2) * 0.5f;
        spatial_sample.valid = std::isfinite(spatial_sample.center_x) &&
            std::isfinite(spatial_sample.center_y);
    }
    push_limited(state.risk_history, risk, window_size);
    push_limited(state.candidate_history, candidate ? 1 : 0, window_size);
    push_limited(state.handheld_history, handheld_candidate ? 1 : 0, window_size);
    push_limited(state.formal_spatial_history, spatial_sample, window_size);
}

static void replay_static_shadow(PersonTrackState& state, int window_size) {
    (void)window_size;
    std::vector<jiankong::custom_pipeline::ShadowRiskSample> replay(
        state.static_shadow_history.begin(), state.static_shadow_history.end());
    std::stable_sort(replay.begin(), replay.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.time_sec < rhs.time_sec;
    });
    state.static_shadow_history.clear();
    const std::size_t aligned_size = std::min({
        state.risk_history.size(), state.candidate_history.size(),
        state.handheld_history.size(), state.formal_spatial_history.size()});
    const std::size_t merge_count = std::min(aligned_size, replay.size());
    const std::size_t history_offset = aligned_size - merge_count;
    const std::size_t replay_offset = replay.size() - merge_count;
    for (std::size_t index = 0; index < merge_count; ++index) {
        const auto& sample = replay[replay_offset + index];
        const std::size_t history_index = history_offset + index;
        state.risk_history[history_index] = std::max(
            state.risk_history[history_index], sample.risk_score);
        state.candidate_history[history_index] =
            (state.candidate_history[history_index] || sample.accepted) ? 1 : 0;
    }
}

static float spatial_point_distance(
    const FormalSpatialSample& sample,
    const jiankong::custom_pipeline::SpatialPoint& point) {
    return std::hypot(sample.center_x - point.x, sample.center_y - point.y);
}

static void discard_static_shadow(
    PersonTrackState& state,
    const jiankong::custom_pipeline::SpatialStaticDecision& decision) {
    state.static_shadow_history.clear();
    if (decision.primary_cluster_centers.empty() ||
        !std::isfinite(decision.primary_cluster_tolerance_px) ||
        decision.primary_cluster_tolerance_px < 0.0f) {
        return;
    }
    const std::size_t count = state.formal_spatial_history.size();
    if (state.risk_history.size() != count || state.candidate_history.size() != count ||
        state.handheld_history.size() != count) {
        return;
    }
    for (std::size_t index = 0; index < count; ++index) {
        const FormalSpatialSample& sample = state.formal_spatial_history[index];
        if (!sample.valid) continue;
        bool matches_primary_cluster = false;
        for (const auto& center : decision.primary_cluster_centers) {
            if (!std::isfinite(center.x) || !std::isfinite(center.y)) continue;
            if (spatial_point_distance(sample, center) <=
                decision.primary_cluster_tolerance_px) {
                matches_primary_cluster = true;
                break;
            }
        }
        if (!matches_primary_cluster) continue;
        state.risk_history[index] = 0.0f;
        state.candidate_history[index] = 0;
        state.handheld_history[index] = 0;
    }
}

static void mark_spatial_static_suppressed(
    CandidateEval& ev, PersonTrackState& state,
    const jiankong::custom_pipeline::SpatialStaticDecision& decision) {
    ev.phone_static = true;
    ev.phone_static_duration = 3.0f;
    ev.phone_motion_px = decision.metrics.center_spread_px;
    ev.phone_follow_wrist = decision.metrics.follows_wrist;
    ev.static_suppressed = true;
    ev.static_risk_multiplier = clampf(ev.static_config_risk_multiplier, 0.0f, 1.0f);
    ev.risk_score = clampf(ev.risk_score * ev.static_risk_multiplier, 0.0f, 1.0f);
    ev.level = candidate_level(ev.risk_score);
    ev.reject_reason = "static_phone_suppressed";
    std::ostringstream reason;
    reason << "STATIC_PHONE_SUPPRESSED"
           << "|samples=" << decision.metrics.sample_count
           << "|detection_ratio=" << std::fixed << std::setprecision(2)
           << decision.metrics.detection_ratio
           << "|motion=" << decision.metrics.center_spread_px
           << "|motion_decoupled=" << (decision.metrics.motion_decoupled ? 1 : 0)
           << "|person_motion=" << decision.metrics.person_motion_px
           << "|wrist_motion=" << decision.metrics.wrist_motion_px
           << "|risk=" << ev.risk_score;
    ev.candidate_reason = reason.str();
    state.static_phone_suppressed_hits += 1;
}

static constexpr int SUSPECT_HOLD_FRAMES = 20;

static Rect lerp_rect(const Rect& a, const Rect& b, float alpha) {
    return Rect{
        a.x1 * (1.0f - alpha) + b.x1 * alpha,
        a.y1 * (1.0f - alpha) + b.y1 * alpha,
        a.x2 * (1.0f - alpha) + b.x2 * alpha,
        a.y2 * (1.0f - alpha) + b.y2 * alpha,
    };
}

static bool is_suspect_track(const PersonTrackState& state, int frame_id, int hold_frames = SUSPECT_HOLD_FRAMES) {
    if (state.alarm_triggered || state.window_hits > 0) return true;
    return state.suspect_active && frame_id - state.last_suspect_frame <= hold_frames;
}

static Rect display_bbox_for_track(const PersonTrackState& state, const Det& person) {
    return state.has_smoothed_bbox ? state.smoothed_bbox : person.box;
}

static Rect display_bbox_for_state(const PersonTrackState& state) {
    if (state.has_smoothed_bbox) return state.smoothed_bbox;
    return state.last_bbox;
}

static int assign_person_track_ids(std::vector<Det>& people, std::map<int, PersonTrackState>& states,
                                   int next_track_id, int frame_id, int window_size,
                                   const jiankong::custom_pipeline::SpatialStaticConfig& spatial_static_config =
                                       jiankong::custom_pipeline::SpatialStaticConfig(),
                                   bool spatial_static_enabled = true) {
    std::set<int> assigned;
    const int stale_after = std::max(window_size * 2, 60);
    for (auto& person : people) {
        int best_id = -1;
        float best_score = 0.0f;
        for (auto& kv : states) {
            const int track_id = kv.first;
            auto& state = kv.second;
            if (assigned.count(track_id) || !state.has_last_bbox || frame_id - state.last_seen > stale_after) continue;
            const float ov = iou(person.box, state.last_bbox);
            const float cd = center_dist(person.box, state.last_bbox);
            const float center_score = clampf(1.0f - cd / std::max({person.box.w(), person.box.h(), rect_diag(state.last_bbox), 1.0f}), 0.0f, 1.0f);
            const float score = std::max(ov, center_score * 0.5f);
            if (score > best_score) {
                best_score = score;
                best_id = track_id;
            }
        }
        if (best_id < 0 || best_score < 0.20f) {
            best_id = next_track_id++;
            auto [state_it, inserted] = states.try_emplace(best_id, spatial_static_config);
            state_it->second.track_id = best_id;
            if (inserted) state_it->second.spatial_static_enabled = spatial_static_enabled;
        }
        auto& state = states[best_id];
        state.track_id = best_id;
        person.track_id = best_id;
        state.last_seen = frame_id;
        state.last_bbox = person.box;
        state.has_last_bbox = true;
        assigned.insert(best_id);
    }
    return next_track_id;
}

static void prune_person_tracks(std::map<int, PersonTrackState>& states, int frame_id, int window_size) {
    jiankong::custom_pipeline::prune_stale_track_states(states, frame_id, window_size);
}

static void update_eval_risk(CandidateEval& ev, float risk_threshold) {
    const float candidate_threshold = std::max(0.65f, risk_threshold);
    ev.risk_score = risk_score(ev.phone_score, ev.phone_hand_score, ev.screen_relation_score, ev.pose_score, ev.temporal_score);
    ev.level = candidate_level(ev.risk_score);
    if (ev.person_index >= 0 && (ev.reject_reason.empty() || ev.reject_reason == "low_risk_score")) {
        if (ev.risk_score > candidate_threshold && ev.static_zone_score > 0.0f && ev.person_match_score >= 0.35f) ev.reject_reason.clear();
        else ev.reject_reason = "low_risk_score";
    }
    std::ostringstream oss;
    oss << ev.level << "|" << ev.zone_reason
        << "|phone=" << std::fixed << std::setprecision(2) << ev.phone_score
        << "|hand=" << ev.phone_hand_score
        << "|screen=" << ev.screen_relation_score
        << "|pose=" << ev.pose_score
        << "|temporal=" << ev.temporal_score
        << "|risk=" << ev.risk_score;
    if (is_handheld_phone_suspect_candidate(ev)) oss << "|handheld_suspect=1";
    if (!ev.reject_reason.empty()) oss << "|reject=" << ev.reject_reason;
    ev.candidate_reason = oss.str();
}

static CandidateEval evaluate_phone(const Det& phone, const ScreenConfig& screen, const std::vector<Det>& people,
                                     const std::map<int, PersonTrackState>& states, int width, int height, float kp_thr) {
    CandidateEval ev;
    ev.phone = phone;
    ev.screen_id = screen.screen_id;
    ev.static_suppression_enabled = screen.params.enable_static_phone_suppression;
    ev.static_window_seconds = screen.params.static_phone_window_seconds;
    ev.static_max_disp_ratio = screen.params.static_phone_max_disp_ratio;
    ev.static_config_risk_multiplier = screen.params.static_phone_risk_multiplier;
    ev.static_min_abs_disp_px = screen.params.static_phone_min_abs_disp_px;
    ev.static_min_bbox_iou = screen.params.static_phone_min_bbox_iou;
    ev.static_max_bbox_size_change_ratio = screen.params.static_phone_max_bbox_size_change_ratio;
    ev.static_lower_person_start_ratio = screen.params.static_phone_lower_person_start_ratio;
    ev.static_wrist_follow_min_motion_px = screen.params.static_phone_wrist_follow_min_motion_px;
    ev.static_wrist_follow_cosine = screen.params.static_phone_wrist_follow_cosine;
    ev.phone_in_desk_zone = point_in_zones(phone_center(phone), screen.desk_static_zones);
    const float required_conf = std::max(
        screen.params.phone_valid_conf_person_roi,
        screen.params.phone_valid_conf_floor);
    auto [static_score, zone_reason] = static_zone_score(phone, screen);
    ev.static_zone_score = static_score;
    ev.zone_reason = zone_reason;
    ev.screen_relation_score = clampf(static_score, 0.0f, 1.0f);
    if (zone_reason.rfind("ignore", 0) == 0) ev.reject_reason = "ignore_zone";
    else if (phone.conf < required_conf) ev.reject_reason = "low_conf_for_source";
    else if (static_score <= 0.0f) ev.reject_reason = "outside_zone";

    float best_score = 0.0f;
    Rect best_roi{};
    Rect best_corridor{};
    bool best_corridor_hit = false;
    int preferred_person_index = phone.person_index;
    for (int i = 0; i < static_cast<int>(people.size()); ++i) {
        if (preferred_person_index >= 0 && people[i].person_index != preferred_person_index) continue;
        if (!person_related_to_screen(people[i], screen, width, height, kp_thr)) continue;
        auto [person_zone, person_roi, corridor, corridor_hit] = person_zone_score(phone, people[i], screen, width, height, kp_thr);
        const float hand = hand_link_score(phone, people[i], screen, kp_thr);
        const float corridor_score = corridor_hit ? 0.7f : 0.0f;
        const float score = 0.45f * hand + 0.35f * person_zone + 0.20f * corridor_score;
        if (score > best_score) {
            best_score = score;
            ev.person_index = people[i].person_index;
            ev.track_id = people[i].track_id;
            ev.phone_hand_score = hand;
            best_roi = person_roi;
            best_corridor = corridor;
            best_corridor_hit = corridor_hit;
        }
    }
    ev.person_match_score = best_score;
    ev.person_roi = best_roi;
    ev.corridor_bbox = best_corridor;
    if (ev.reject_reason.empty() && ev.person_index < 0) ev.reject_reason = "no_matched_person";
    if (ev.reject_reason.empty() && ev.person_match_score < 0.35f) ev.reject_reason = "no_matched_person";
    if (ev.person_index >= 0) {
        const Det* person = nullptr;
        for (const auto& p : people) {
            if (p.person_index == ev.person_index) {
                person = &p;
                break;
            }
        }
        if (person) {
            const auto geometry = phone_person_geometry_gate(phone, *person, screen.params);
            ev.phone_person_area_ratio = geometry.area_ratio;
            ev.phone_person_diag_ratio = geometry.diag_ratio;
            ev.phone_geometry_valid = geometry.valid;
            if (!geometry.valid && ev.reject_reason.empty()) {
                ev.reject_reason = "implausible_phone_geometry";
            }
            auto [aim, best_angle, ray_hit, reason] = aim_score(phone, *person, screen, static_score, ev.phone_hand_score, kp_thr);
            (void)reason;
            ev.best_angle = best_angle;
            ev.best_ray_hit = ray_hit;
            ev.phone_score = phone_reliability_score(phone, required_conf);
            ev.gated_phone_valid = phone.conf >= required_conf && ev.phone_score > 0.0f &&
                ev.phone_geometry_valid && ev.zone_reason.rfind("ignore", 0) != 0 &&
                ev.static_zone_score > 0.0f;
            ev.high_confidence_phone =
                phone.conf >= screen.params.alarm_raw_phone_confidence;
            ev.gated_hand_relation = gated_hand_relation(phone, *person);
            ev.gated_person_association = gated_person_association(
                phone, *person, ev.gated_hand_relation, width, height);
            float gated_angle = 180.0f;
            bool gated_ray_hit = false;
            bool gated_exact_ray_hit = false;
            bool gated_corridor_ray_hit = false;
            ev.gated_screen_intent = gated_screen_intent(
                phone, *person, screen, gated_angle, gated_ray_hit,
                gated_exact_ray_hit, gated_corridor_ray_hit);
            ev.best_angle = gated_angle;
            ev.best_ray_hit = gated_ray_hit;
            ev.screen_ray_hit = gated_ray_hit;
            ev.exact_screen_ray_hit = gated_exact_ray_hit;
            ev.corridor_screen_ray_hit = gated_corridor_ray_hit;
            ev.pose_score = pose_support_score(phone, *person, aim, ev.phone_hand_score, kp_thr);
            auto sit = states.find(ev.track_id);
            ev.temporal_score = compute_temporal_score(sit == states.end() ? nullptr : &sit->second, screen.params.person_state_risk_threshold);
            update_eval_risk(ev, screen.params.person_state_risk_threshold);
            if (best_corridor_hit && ev.zone_reason == "outside") ev.zone_reason = "corridor";
        }
    }
    update_eval_risk(ev, screen.params.person_state_risk_threshold);
    return ev;
}

static float candidate_delta(const std::string& level) {
    if (level == "strong") return 2.0f;
    if (level == "normal") return 1.0f;
    if (level == "weak") return 0.3f;
    return 0.0f;
}

static void update_person_alarm_state(
    PersonTrackState& state, const Det& person,
    const std::vector<CandidateEval*>& person_evals, CandidateEval* ev,
    int frame_id, int window_size, float risk_threshold, int min_hits,
    int handheld_suspect_min_hits, double infer_fps,
    bool record_spatial_history = true) {
    const bool has_eval = ev != nullptr;
    const float risk = ev != nullptr ? ev->risk_score : 0.0f;
    const bool phone_seen = std::any_of(person_evals.begin(), person_evals.end(),
        [](const CandidateEval* candidate) {
            return candidate != nullptr && candidate->phone_score > 0.0f;
        });
    const bool candidate = ev != nullptr && ev->accepted() && risk >= risk_threshold;
    const bool handheld_candidate = ev != nullptr && is_handheld_phone_suspect_candidate(*ev);
    const std::optional<jiankong::custom_pipeline::SpatialRect> formal_spatial_box =
        ev != nullptr
            ? std::optional<jiankong::custom_pipeline::SpatialRect>(spatial_rect(ev->phone.box))
            : std::nullopt;
    state.smoothed_bbox = state.has_smoothed_bbox
        ? lerp_rect(state.smoothed_bbox, person.box, 0.35f) : person.box;
    state.has_smoothed_bbox = true;
    push_limited(state.bbox_history, person.box, window_size);
    push_limited(state.phone_history, phone_seen ? 1 : 0, window_size);
    if (record_spatial_history) {
        append_formal_history(
            state, risk, candidate, handheld_candidate, formal_spatial_box, window_size);
    } else {
        push_limited(state.risk_history, risk, window_size);
        push_limited(state.candidate_history, candidate ? 1 : 0, window_size);
        push_limited(state.handheld_history, handheld_candidate ? 1 : 0, window_size);
    }
    state.window_hits = std::accumulate(
        state.candidate_history.begin(), state.candidate_history.end(), 0);
    state.stable_count = trailing_candidate_count(state.candidate_history);
    state.handheld_phone_hits = std::accumulate(
        state.handheld_history.begin(), state.handheld_history.end(), 0);
    state.handheld_phone_stable_count = trailing_candidate_count(state.handheld_history);
    const int handheld_stable_min = std::max(6, (handheld_suspect_min_hits + 1) / 2);
    const bool handheld_suspect = state.handheld_phone_hits >= handheld_suspect_min_hits
                                  || state.handheld_phone_stable_count >= handheld_stable_min;
    state.legacy_window_hits = state.window_hits;
    state.legacy_alarm_triggered = state.window_hits >= min_hits;
    state.gated_event_policy.set_infer_fps(infer_fps);
    state.gated_event_decision = state.gated_event_policy.update(
        gated_frame_evidence(state, ev));
    state.alarm_triggered = state.gated_event_decision.alarm;
    state.window_hits = state.gated_event_decision.core_hits;
    state.state = jiankong::custom_pipeline::gated_event_state_name(
        state.gated_event_decision.state);
    if (has_eval) {
        state.last_risk_score = std::max(state.last_risk_score * 0.90f, risk);
        if (!ev->screen_id.empty()) state.last_screen_id = ev->screen_id;
    } else {
        state.last_risk_score *= 0.90f;
    }
    if (candidate || state.alarm_triggered || state.window_hits > 0 || handheld_suspect) {
        state.suspect_active = true;
        state.last_suspect_frame = frame_id;
    } else if (frame_id - state.last_suspect_frame > SUSPECT_HOLD_FRAMES) {
        state.suspect_active = false;
    }
    state.last_seen = frame_id;
    state.last_bbox = person.box;
    state.has_last_bbox = true;
}

static void update_person_states_legacy(
    std::vector<Det>& people, std::vector<CandidateEval>& evals,
    std::map<int, PersonTrackState>& states, int frame_id,
    int window_size, float risk_threshold, int min_hits,
    int handheld_suspect_min_hits, double infer_fps) {
    std::map<int, Det*> people_by_track;
    for (auto& person : people) {
        if (person.track_id >= 0) people_by_track[person.track_id] = &person;
    }
    for (auto& ev : evals) {
        if (ev.track_id < 0) continue;
        auto pit = people_by_track.find(ev.track_id);
        if (pit == people_by_track.end()) continue;
        auto& state = states[ev.track_id];
        state.track_id = ev.track_id;
        apply_legacy_static_phone_suppression(
            ev, state, *pit->second, frame_id, static_cast<float>(infer_fps));
    }

    std::map<int, int> best_by_track;
    for (int i = 0; i < static_cast<int>(evals.size()); ++i) {
        const auto& ev = evals[i];
        if (ev.track_id < 0) continue;
        auto it = best_by_track.find(ev.track_id);
        if (it == best_by_track.end() ||
            ev.risk_score > evals[it->second].risk_score) {
            best_by_track[ev.track_id] = i;
        }
    }
    for (const auto& person : people) {
        if (person.track_id < 0) continue;
        auto& state = states[person.track_id];
        state.track_id = person.track_id;
        auto it = best_by_track.find(person.track_id);
        const bool has_eval = it != best_by_track.end();
        CandidateEval* ev = has_eval ? &evals[it->second] : nullptr;
        const bool static_blocked = ev != nullptr && ev->static_suppressed;
        if (static_blocked) {
            state.risk_history.clear();
            state.candidate_history.clear();
            state.handheld_history.clear();
        }
        const float risk = ev != nullptr ? ev->risk_score : 0.0f;
        const bool phone_seen = ev != nullptr && ev->phone_score > 0.0f;
        const bool candidate = ev != nullptr && !static_blocked &&
            ev->accepted() && risk >= risk_threshold;
        const bool handheld_candidate = ev != nullptr && !static_blocked &&
            is_handheld_phone_suspect_candidate(*ev);
        state.smoothed_bbox = state.has_smoothed_bbox
            ? lerp_rect(state.smoothed_bbox, person.box, 0.35f) : person.box;
        state.has_smoothed_bbox = true;
        push_limited(state.bbox_history, person.box, window_size);
        push_limited(state.phone_history, phone_seen ? 1 : 0, window_size);
        push_limited(state.risk_history, risk, window_size);
        push_limited(state.candidate_history, candidate ? 1 : 0, window_size);
        push_limited(state.handheld_history, handheld_candidate ? 1 : 0, window_size);
        state.window_hits = std::accumulate(
            state.candidate_history.begin(), state.candidate_history.end(), 0);
        state.stable_count = trailing_candidate_count(state.candidate_history);
        state.handheld_phone_hits = std::accumulate(
            state.handheld_history.begin(), state.handheld_history.end(), 0);
        state.handheld_phone_stable_count =
            trailing_candidate_count(state.handheld_history);
        const int handheld_stable_min = std::max(
            6, (handheld_suspect_min_hits + 1) / 2);
        const bool handheld_suspect =
            state.handheld_phone_hits >= handheld_suspect_min_hits ||
            state.handheld_phone_stable_count >= handheld_stable_min;
        state.legacy_window_hits = state.window_hits;
        state.legacy_alarm_triggered = state.window_hits >= min_hits;
        state.gated_event_policy.set_infer_fps(infer_fps);
        state.gated_event_decision = state.gated_event_policy.update(
            gated_frame_evidence(state, ev));
        state.alarm_triggered = state.gated_event_decision.alarm;
        state.window_hits = state.gated_event_decision.core_hits;
        state.state = jiankong::custom_pipeline::gated_event_state_name(
            state.gated_event_decision.state);
        if (has_eval) {
            state.last_risk_score = std::max(
                state.last_risk_score * 0.90f, risk);
            if (!ev->screen_id.empty()) state.last_screen_id = ev->screen_id;
        } else {
            state.last_risk_score *= 0.90f;
        }
        if (candidate || state.alarm_triggered || state.window_hits > 0 ||
            handheld_suspect) {
            state.suspect_active = true;
            state.last_suspect_frame = frame_id;
        } else if (frame_id - state.last_suspect_frame > SUSPECT_HOLD_FRAMES) {
            state.suspect_active = false;
        }
        state.last_seen = frame_id;
        state.last_bbox = person.box;
        state.has_last_bbox = true;
    }
}

static void update_person_states(std::vector<Det>& people, std::vector<CandidateEval>& evals,
                                  std::map<int, PersonTrackState>& states, int frame_id,
                                  int window_size, float risk_threshold, int min_hits,
                                  int handheld_suspect_min_hits, double infer_fps,
                                  jiankong::custom_pipeline::CameraStaticHotspotMap* static_hotspots = nullptr,
                                  double hotspot_now = 0.0,
                                  bool* hotspot_dirty = nullptr,
                                  bool spatial_static_enabled = true) {
    if (!spatial_static_enabled) {
        update_person_states_legacy(
            people, evals, states, frame_id, window_size, risk_threshold,
            min_hits, handheld_suspect_min_hits, infer_fps);
        return;
    }
    std::map<int, std::vector<CandidateEval*>> evals_by_track;
    for (auto& ev : evals) {
        if (ev.track_id < 0) continue;
        evals_by_track[ev.track_id].push_back(&ev);
    }
    for (const auto& person : people) {
        if (person.track_id < 0) continue;
        auto& state = states[person.track_id];
        state.track_id = person.track_id;
        std::vector<CandidateEval*> person_evals;
        const auto eval_it = evals_by_track.find(person.track_id);
        if (eval_it != evals_by_track.end()) person_evals = eval_it->second;
        for (CandidateEval* candidate : person_evals) {
            if (candidate == nullptr) continue;
            candidate->static_hotspot_score = 0.0f;
            if (static_hotspots != nullptr && std::isfinite(hotspot_now)) {
                const cv::Point2f center = phone_center(candidate->phone);
                const float score = static_hotspots->score(
                    center.x, center.y, hotspot_now);
                if (std::isfinite(score)) {
                    candidate->static_hotspot_score = std::max(0.0f, score);
                }
            }
        }
        const double frame_time_sec = frame_id / std::max(1.0, infer_fps);
        const auto spatial_frame = build_spatial_frame_observation(person, person_evals, frame_time_sec);
        const auto spatial_decision = observe_spatial_static_phone(
            state, spatial_frame, state.spatial_static_enabled,
            static_hotspots, hotspot_now, hotspot_dirty);
        CandidateEval* spatial_primary = nullptr;
        if (spatial_decision.primary_candidate_index.has_value() &&
            *spatial_decision.primary_candidate_index < person_evals.size()) {
            spatial_primary = person_evals[*spatial_decision.primary_candidate_index];
        }
        const bool block_spatial_primary = spatial_decision.hold_candidate ||
            spatial_decision.discard_shadow ||
            spatial_decision.phase == jiankong::custom_pipeline::SpatialStaticPhase::suppressed;

        if (spatial_decision.hold_candidate && spatial_primary != nullptr) {
            const bool primary_candidate = spatial_primary->accepted() &&
                spatial_primary->risk_score >= risk_threshold;
            state.static_shadow_history.push_back(jiankong::custom_pipeline::ShadowRiskSample{
                frame_time_sec, spatial_primary->risk_score, primary_candidate});
            const int shadow_limit = std::max(window_size,
                static_cast<int>(std::ceil(
                    state.spatial_static_shadow_seconds * std::max(1.0, infer_fps))) + 2);
            while (static_cast<int>(state.static_shadow_history.size()) > shadow_limit) {
                state.static_shadow_history.pop_front();
            }
        }
        if (spatial_decision.discard_shadow) {
            discard_static_shadow(state, spatial_decision);
            if (spatial_primary != nullptr) {
                mark_spatial_static_suppressed(*spatial_primary, state, spatial_decision);
            }
        } else if (spatial_decision.replay_shadow) {
            replay_static_shadow(state, window_size);
        }
        update_spatial_static_diagnostics(
            state, person_evals, spatial_decision, frame_time_sec);

        CandidateEval* ev = best_formal_candidate(
            person_evals, block_spatial_primary ? spatial_primary : nullptr);
        update_person_alarm_state(
            state, person, person_evals, ev, frame_id, window_size, risk_threshold,
            min_hits, handheld_suspect_min_hits, infer_fps);
    }
}

static int stream_state_window(const StreamState& s) {
    int v = 30;
    for (const auto& screen : s.screens) v = std::max(v, screen.params.person_state_window);
    return std::max(1, v);
}

static int stream_state_min_hits(const StreamState& s, int window_size) {
    int v = 16;
    for (const auto& screen : s.screens) v = std::max(v, screen.params.person_state_min_hits);
    return std::max(1, std::min(v, window_size));
}

static int stream_handheld_suspect_min_hits(const StreamState& s, int min_hits) {
    int v = 12;
    for (const auto& screen : s.screens) v = std::max(v, screen.params.handheld_suspect_min_hits);
    return std::max(1, std::min(v, std::max(1, min_hits)));
}

static float stream_state_risk_threshold(const StreamState& s) {
    float v = 0.65f;
    for (const auto& screen : s.screens) v = std::max(v, screen.params.person_state_risk_threshold);
    return v;
}

static CandidateEval choose_best_eval(const std::vector<CandidateEval>& evals) {
    if (evals.empty()) return CandidateEval{};
    auto better = [](const CandidateEval& a, const CandidateEval& b) {
        if (a.accepted() != b.accepted()) return a.accepted() > b.accepted();
        if (a.accepted()) {
            const float da = candidate_delta(a.level);
            const float db = candidate_delta(b.level);
            if (std::abs(da - db) > 1e-6f) return da > db;
        }
        return a.risk_score > b.risk_score;
    };
    CandidateEval best = evals[0];
    for (size_t i = 1; i < evals.size(); ++i) {
        if (better(evals[i], best)) best = evals[i];
    }
    return best;
}

static const CandidateEval* best_eval_for_person(const std::vector<CandidateEval>& evals, int person_index) {
    const CandidateEval* best = nullptr;
    for (const auto& ev : evals) {
        if (ev.person_index != person_index) continue;
        if (!best || ev.risk_score > best->risk_score) best = &ev;
    }
    return best;
}

static const CandidateEval* best_eval_for_phone(const std::vector<CandidateEval>& evals, const Det& phone) {
    const CandidateEval* best = nullptr;
    for (const auto& ev : evals) {
        if (iou(ev.phone.box, phone.box) < 0.80f) continue;
        if (!best || ev.risk_score > best->risk_score) best = &ev;
    }
    return best;
}

static std::vector<Det> decode_pose(const TrtRunner& runner, int batch_index, int channels,
                                    const LetterboxMeta& meta, int width, int height, float kp_thr) {
    std::vector<Det> candidates;
    candidates.reserve(128);
    std::vector<int> slots(runner.compacted_count(batch_index));
    std::iota(slots.begin(), slots.end(), 0);
    std::sort(slots.begin(), slots.end(), [&](int lhs, int rhs) {
        return runner.compacted_original_index(batch_index, lhs) <
               runner.compacted_original_index(batch_index, rhs);
    });
    for (int slot : slots) {
        const int original_index = runner.compacted_original_index(batch_index, slot);
        const float* values = runner.compacted_values(batch_index, slot);
        const float conf = values[4];
        Det d;
        d.conf = conf;
        d.person_index = original_index;
        d.box = map_box(values[0], values[1], values[2], values[3],
                        meta, width, height);
        if (d.box.area() < 4.0f) continue;
        for (int k = 0; k < 17; ++k) {
            const int c = 5 + k * 3;
            d.kpts[k][0] = (values[c] - meta.pad_x) / meta.scale;
            d.kpts[k][1] = (values[c + 1] - meta.pad_y) / meta.scale;
            d.kpts[k][2] = values[c + 2];
        }
        int visible = 0;
        for (int idx : {0, 5, 6, 7, 8, 9, 10}) {
            if (d.kpts[idx][2] >= kp_thr) ++visible;
        }
        if (visible < 2) continue;
        candidates.push_back(d);
    }
    return nms(std::move(candidates), 0.70f, 100);
}

static Rect expand_person_roi(const Rect& r, int width, int height, float expand_x, float expand_y) {
    const float dx = r.w() * expand_x;
    const float dy = r.h() * expand_y;
    return clamp_rect(Rect{r.x1 - dx, r.y1 - dy, r.x2 + dx, r.y2 + dy}, width, height);
}

static std::vector<Det> decode_phone_batch(const TrtRunner& runner, int real_batch, int channels,
                                           const std::vector<LetterboxMeta>& metas, const std::vector<RoiJob>& jobs,
                                           int job_start, const std::vector<StreamState>& streams) {
    std::vector<Det> phones;
    for (int b = 0; b < real_batch; ++b) {
        const RoiJob& job = jobs[job_start + b];
        const StreamState& st = streams[job.stream];
        const LetterboxMeta& m = metas[b];
        std::vector<Det> candidates;
        std::vector<int> slots(runner.compacted_count(b));
        std::iota(slots.begin(), slots.end(), 0);
        std::sort(slots.begin(), slots.end(), [&](int lhs, int rhs) {
            return runner.compacted_original_index(b, lhs) < runner.compacted_original_index(b, rhs);
        });
        for (int slot : slots) {
            const float* values = runner.compacted_values(b, slot);
            const float conf = values[4];
            Det d;
            d.conf = conf;
            d.stream = job.stream;
            d.roi_index = job_start + b;
            d.person_index = job.person_index;
            Rect local = map_box(values[0], values[1], values[2], values[3], m,
                                 static_cast<int>(job.roi.w()), static_cast<int>(job.roi.h()));
            d.box = clamp_rect(Rect{local.x1 + job.roi.x1, local.y1 + job.roi.y1, local.x2 + job.roi.x1, local.y2 + job.roi.y1},
                               st.width, st.height);
            if (d.box.area() >= 4.0f) candidates.push_back(d);
        }
        auto keep = nms(std::move(candidates), 0.50f, 100);
        phones.insert(phones.end(), keep.begin(), keep.end());
    }
    return phones;
}

struct Args {
    std::string root = "/media/boshi/Data/JianKong/03_raw_videos_and_frames/2026-07-03";
    std::string pose_plan = "/media/boshi/Data/JianKong/06_training_runs/raw_trt_plans_20260706_114432/pose960_static_b7.plan";
    std::string phone_plan = "/media/boshi/Data/JianKong/06_training_runs/raw_trt_plans_20260707_phone512/phone512_own_all_static_b16.plan";
    int max_samples = 400;
    double infer_fps = 10.0;
    int pose_size = 960;
    int phone_size = 512;
    int pose_batch = 7;
    int phone_batch = 16;
    std::string calib_dir = "/media/boshi/Data/JianKong/02_configs/surveillance";
    float pose_conf = 0.25f;
    float kp_conf = 0.35f;
    float phone_conf = 0.25f;
    float person_expand_x = 0.10f;
    float person_expand_y = 0.06f;
    bool gpu_preprocess = false;
    bool pipelined_read = false;
    bool self_test = false;
    bool self_test_rules = false;
    int queue_size = 8;
    int cv_threads = 1;
    int pick_from_end = 1;
    std::string output_dir;
    bool no_video = false;
    bool override_static_phone_suppression = false;
    bool enable_static_phone_suppression = true;
    float static_window_seconds = 1.5f;
    float static_max_disp_ratio = 0.03f;
    float static_risk_multiplier = 0.2f;
    bool spatial_static_phone_suppression_enabled = true;
    int spatial_static_phone_suppression_switch = 0;
    double static_observation_seconds = 3.0;
    double static_pending_seconds = 0.75;
    double static_long_confirm_seconds = 6.0;
    double static_min_detection_ratio = 0.60;
    double static_max_gap_seconds = 0.75;
    double static_position_radius_ratio = 0.03;
    bool static_hotspot_enabled = true;
    std::string fixed_template_dir;
    std::vector<std::string> rtsp_specs;
    std::string camera_manifest;
    std::string relay_base;
    int duration_sec = 300;
    int reconnect_delay_ms = 1000;
    int open_timeout_ms = 5000;
    int read_timeout_ms = 3000;
    int source_width = 2560;
    int source_height = 1440;
    bool live_self_test = false;
    bool show_help = false;
};

static jiankong::custom_pipeline::StaticEvidenceState gated_static_state(
    const PersonTrackState& state, const CandidateEval* ev) {
    using jiankong::custom_pipeline::SpatialStaticPhase;
    using jiankong::custom_pipeline::StaticEvidenceState;
    if ((ev != nullptr && ev->static_suppressed) ||
        state.last_static_decision.phase == SpatialStaticPhase::suppressed) {
        return StaticEvidenceState::static_confirmed;
    }
    if (!state.spatial_static_enabled ||
        state.last_static_decision.phase == SpatialStaticPhase::handheld_or_moving) {
        return StaticEvidenceState::moving_or_handheld;
    }
    return StaticEvidenceState::unknown;
}

static jiankong::custom_pipeline::GatedFrameEvidence gated_frame_evidence(
    const PersonTrackState& state, const CandidateEval* ev) {
    jiankong::custom_pipeline::GatedFrameEvidence evidence;
    if (ev != nullptr) {
        evidence.phone_valid = ev->gated_phone_valid;
        evidence.high_confidence_phone = ev->high_confidence_phone;
        evidence.screen_ray_hit = ev->screen_ray_hit;
        evidence.person_association = ev->gated_person_association;
        evidence.hand_relation = ev->gated_hand_relation;
        evidence.screen_intent = ev->gated_screen_intent;
        evidence.transition_observed = ev->phone_follow_wrist ||
            state.last_static_decision.replay_shadow;
    }
    evidence.static_state = gated_static_state(state, ev);
    return evidence;
}

static void validate_spatial_static_args(const Args& args);

static Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + k);
            return argv[++i];
        };
        if (k == "--root") a.root = next();
        else if (k == "--pose-plan") a.pose_plan = next();
        else if (k == "--phone-plan") a.phone_plan = next();
        else if (k == "--max-sampled") a.max_samples = std::stoi(next());
        else if (k == "--infer-fps") a.infer_fps = std::stod(next());
        else if (k == "--pose-size") a.pose_size = std::stoi(next());
        else if (k == "--phone-size") a.phone_size = std::stoi(next());
        else if (k == "--calib-dir") a.calib_dir = next();
        else if (k == "--pose-conf") a.pose_conf = std::stof(next());
        else if (k == "--kp-conf") a.kp_conf = std::stof(next());
        else if (k == "--phone-conf") a.phone_conf = std::stof(next());
        else if (k == "--person-expand-x") a.person_expand_x = std::stof(next());
        else if (k == "--person-expand-y") a.person_expand_y = std::stof(next());
        else if (k == "--gpu-preprocess") a.gpu_preprocess = true;
        else if (k == "--pipelined-read") a.pipelined_read = true;
        else if (k == "--self-test") a.self_test = true;
        else if (k == "--self-test-rules") a.self_test_rules = true;
        else if (k == "--queue-size") a.queue_size = std::stoi(next());
        else if (k == "--cv-threads") a.cv_threads = std::stoi(next());
        else if (k == "--pick-from-end") a.pick_from_end = std::stoi(next());
        else if (k == "--output-dir") a.output_dir = next();
        else if (k == "--no-video") a.no_video = true;
        else if (k == "--enable-static-phone-suppression") {
            a.override_static_phone_suppression = true;
            a.enable_static_phone_suppression = true;
        }
        else if (k == "--disable-static-phone-suppression") {
            a.override_static_phone_suppression = true;
            a.enable_static_phone_suppression = false;
        }
        else if (k == "--static-window-seconds") {
            a.override_static_phone_suppression = true;
            a.static_window_seconds = std::stof(next());
        }
        else if (k == "--static-max-disp-ratio") {
            a.override_static_phone_suppression = true;
            a.static_max_disp_ratio = std::stof(next());
        }
        else if (k == "--static-risk-multiplier") {
            a.override_static_phone_suppression = true;
            a.static_risk_multiplier = std::stof(next());
        }
        else if (k == "--enable-spatial-static-phone-suppression") {
            if (a.spatial_static_phone_suppression_switch < 0) {
                throw std::runtime_error("conflicting spatial static phone suppression switches");
            }
            a.spatial_static_phone_suppression_switch = 1;
            a.spatial_static_phone_suppression_enabled = true;
        }
        else if (k == "--disable-spatial-static-phone-suppression") {
            if (a.spatial_static_phone_suppression_switch > 0) {
                throw std::runtime_error("conflicting spatial static phone suppression switches");
            }
            a.spatial_static_phone_suppression_switch = -1;
            a.spatial_static_phone_suppression_enabled = false;
        }
        else if (k == "--static-observation-seconds") a.static_observation_seconds = std::stod(next());
        else if (k == "--static-pending-seconds") a.static_pending_seconds = std::stod(next());
        else if (k == "--static-long-confirm-seconds") a.static_long_confirm_seconds = std::stod(next());
        else if (k == "--static-min-detection-ratio") a.static_min_detection_ratio = std::stod(next());
        else if (k == "--static-max-gap-seconds") a.static_max_gap_seconds = std::stod(next());
        else if (k == "--static-position-radius-ratio") a.static_position_radius_ratio = std::stod(next());
        else if (k == "--static-hotspot-enabled") a.static_hotspot_enabled = true;
        else if (k == "--static-hotspot-disabled") a.static_hotspot_enabled = false;
        else if (k == "--fixed-template-dir") a.fixed_template_dir = next();
        else if (k == "--rtsp") a.rtsp_specs.push_back(next());
        else if (k == "--camera-manifest") a.camera_manifest = next();
        else if (k == "--relay-base") a.relay_base = next();
        else if (k == "--duration-sec") a.duration_sec = std::stoi(next());
        else if (k == "--reconnect-delay-ms") a.reconnect_delay_ms = std::stoi(next());
        else if (k == "--open-timeout-ms") a.open_timeout_ms = std::stoi(next());
        else if (k == "--read-timeout-ms") a.read_timeout_ms = std::stoi(next());
        else if (k == "--source-width") a.source_width = std::stoi(next());
        else if (k == "--source-height") a.source_height = std::stoi(next());
        else if (k == "--live-self-test") a.live_self_test = true;
        else if (k == "--help" || k == "-h") a.show_help = true;
        else throw std::runtime_error("unknown arg " + k);
    }
    validate_spatial_static_args(a);
    return a;
}

static void validate_spatial_static_args(const Args& args) {
    const auto require_positive = [](double value, const char* option) {
        if (!std::isfinite(value) || value <= 0.0) {
            throw std::runtime_error(std::string(option) + " must be finite and positive");
        }
    };
    const auto require_ratio = [](double value, const char* option, bool allow_zero) {
        if (!std::isfinite(value) || value < 0.0 || value > 1.0 || (!allow_zero && value == 0.0)) {
            throw std::runtime_error(std::string(option) + " must be within [0,1]");
        }
    };
    require_positive(args.static_observation_seconds, "--static-observation-seconds");
    require_positive(args.static_pending_seconds, "--static-pending-seconds");
    require_positive(args.static_long_confirm_seconds, "--static-long-confirm-seconds");
    require_positive(args.static_max_gap_seconds, "--static-max-gap-seconds");
    require_ratio(args.static_min_detection_ratio, "--static-min-detection-ratio", true);
    require_ratio(args.static_position_radius_ratio, "--static-position-radius-ratio", true);
    if (args.static_long_confirm_seconds < args.static_observation_seconds) {
        throw std::runtime_error(
            "--static-long-confirm-seconds must be >= --static-observation-seconds");
    }
}

static jiankong::custom_pipeline::SpatialStaticConfig spatial_static_config_from_args(
    const Args& args) {
    jiankong::custom_pipeline::SpatialStaticConfig config;
    config.short_seconds = args.static_observation_seconds;
    config.pending_seconds = args.static_pending_seconds;
    config.long_seconds = args.static_long_confirm_seconds;
    config.min_detection_ratio = static_cast<float>(args.static_min_detection_ratio);
    config.max_gap_seconds = args.static_max_gap_seconds;
    config.radius_ratio = static_cast<float>(args.static_position_radius_ratio);
    // A stationary phone is not inherently a fixed desk phone. In live use,
    // suppress only locations confirmed in the false-positive review flow.
    config.manual_templates_only = true;
    return config;
}

static void apply_static_args_to_screens(std::vector<ScreenConfig>& screens, const Args& args) {
    if (!args.override_static_phone_suppression) return;
    for (auto& screen : screens) {
        screen.params.enable_static_phone_suppression = args.enable_static_phone_suppression;
        screen.params.static_phone_window_seconds = args.static_window_seconds;
        screen.params.static_phone_max_disp_ratio = args.static_max_disp_ratio;
        screen.params.static_phone_risk_multiplier = args.static_risk_multiplier;
    }
}

static json screen_to_json(const ScreenConfig& screen, size_t screen_index) {
    json j;
    j["screen_index"] = screen_index;
    j["screen_id"] = screen.screen_id;
    j["screen_poly"] = polygon_to_json(screen.screen_poly);
    j["desk_static_zones"] = json::array();
    for (const auto& z : screen.desk_static_zones) {
        json item;
        item["name"] = z.name;
        item["polygon"] = polygon_to_json(z.polygon);
        j["desk_static_zones"].push_back(std::move(item));
    }
    return j;
}

static int count_accepted(const std::vector<CandidateEval>& evals) {
    int n = 0;
    for (const auto& ev : evals) {
        if (ev.accepted()) ++n;
    }
    return n;
}

static int count_static_suppressed(const std::vector<CandidateEval>& evals) {
    int n = 0;
    for (const auto& ev : evals) {
        if (ev.static_suppressed) ++n;
    }
    return n;
}

static void write_neutral_static_diagnostics(json& p, const char* context_reason) {
    p["static_cluster_samples"] = 0;
    p["static_detection_ratio"] = 0.0f;
    p["static_center_spread_px"] = 0.0f;
    p["static_bbox_iou_median"] = 0.0f;
    p["static_pending"] = false;
    p["static_pending_duration"] = 0.0;
    p["static_hotspot_score"] = 0.0f;
    p["static_context_reason"] = spatial_static_context_reason(context_reason);
    p["static_shadow_hits"] = 0;
    p["static_suppressed"] = false;
    p["static_exit_reason"] = "none";
}

static void write_static_diagnostics(json& p, const CandidateEval* ev) {
    if (ev == nullptr) {
        write_neutral_static_diagnostics(p, "not_primary");
        return;
    }
    p["static_cluster_samples"] = std::max(0, ev->static_cluster_samples);
    p["static_detection_ratio"] = clampf(ev->static_detection_ratio, 0.0f, 1.0f);
    p["static_center_spread_px"] = std::max(0.0f, ev->static_center_spread_px);
    p["static_bbox_iou_median"] = clampf(ev->static_bbox_iou_median, 0.0f, 1.0f);
    p["static_pending"] = ev->static_pending;
    p["static_pending_duration"] = std::max(0.0, ev->static_pending_duration);
    p["static_hotspot_score"] = std::max(0.0f, ev->static_hotspot_score);
    p["static_context_reason"] =
        spatial_static_context_reason(ev->static_context_reason.c_str());
    p["static_shadow_hits"] = std::max(0, ev->static_shadow_hits);
    p["static_suppressed"] = ev->static_suppressed;
    p["static_exit_reason"] =
        spatial_static_exit_reason(ev->static_exit_reason.c_str());
}

static int count_desk_zone_phones(const std::vector<CandidateEval>& evals) {
    int n = 0;
    for (const auto& ev : evals) {
        if (ev.phone_in_desk_zone) ++n;
    }
    return n;
}

static float max_risk_score(const std::vector<CandidateEval>& evals) {
    float max_risk = 0.0f;
    for (const auto& ev : evals) {
        max_risk = std::max(max_risk, ev.risk_score);
    }
    return max_risk;
}

static void write_event_metadata(std::ostream& jsonl,
                                 std::ostream& csv,
                                 int stream_index,
                                 const StreamState& stream,
                                 double infer_fps,
                                 double captured_at_unix_seconds,
                                 float person_expand_x,
                                 float person_expand_y,
                                 const std::vector<Det>& people,
                                 const std::vector<Det>& phones,
                                 const std::vector<CandidateEval>& evals) {
    std::set<int> alarm_tracks;
    std::set<int> suspect_tracks;
    std::set<int> visible_tracks;
    int max_hits = 0;
    float state_max_risk = 0.0f;
    for (const auto& person : people) {
        visible_tracks.insert(person.track_id);
        auto it = stream.person_states.find(person.track_id);
        if (it == stream.person_states.end()) continue;
        max_hits = std::max(max_hits, it->second.window_hits);
        state_max_risk = std::max(state_max_risk, it->second.last_risk_score);
        if (it->second.alarm_triggered) alarm_tracks.insert(person.track_id);
        if (is_suspect_track(it->second, stream.frame_id)) suspect_tracks.insert(person.track_id);
    }
    for (const auto& kv : stream.person_states) {
        const auto& state = kv.second;
        if (!state.has_last_bbox) continue;
        if (is_suspect_track(state, stream.frame_id)
            && stream.frame_id - state.last_seen <= SUSPECT_HOLD_FRAMES) {
            suspect_tracks.insert(kv.first);
            state_max_risk = std::max(state_max_risk, state.last_risk_score);
            if (state.alarm_triggered) alarm_tracks.insert(kv.first);
            max_hits = std::max(max_hits, state.window_hits);
        }
    }

    const int accepted_count = count_accepted(evals);
    const int static_suppressed_count = count_static_suppressed(evals);
    const int desk_zone_phone_count = count_desk_zone_phones(evals);
    if (accepted_count <= 0 && suspect_tracks.empty()) return;

    const long long frame_index = std::max<long long>(0, stream.frames - 1);
    const double time_sec = infer_fps > 0 ? static_cast<double>(frame_index) / infer_fps : 0.0;
    const float max_risk = std::max(max_risk_score(evals), state_max_risk);

    json j;
    j["stream_index"] = stream_index;
    j["frame_index"] = frame_index;
    j["frame_id"] = stream.frame_id;
    j["time_sec"] = time_sec;
    if (captured_at_unix_seconds > 0.0) {
        j["captured_at"] = format_utc_timestamp(captured_at_unix_seconds);
    }
    j["width"] = stream.width;
    j["height"] = stream.height;
    j["input_video"] = stream.path;
    j["output_video"] = stream.output_path;
    j["person_count"] = people.size();
    j["phone_count"] = phones.size();
    j["accepted_count"] = accepted_count;
    j["static_phone_suppressed_candidates"] = static_suppressed_count;
    j["desk_zone_phone_count"] = desk_zone_phone_count;
    j["alarm_track_count"] = alarm_tracks.size();
    j["suspect_track_count"] = suspect_tracks.size();
    j["max_risk"] = max_risk;
    j["max_window_hits"] = max_hits;

    j["screens"] = json::array();
    for (size_t i = 0; i < stream.screens.size(); ++i) {
        j["screens"].push_back(screen_to_json(stream.screens[i], i));
    }

    j["persons"] = json::array();
    for (const auto& person : people) {
        const CandidateEval* ev = best_eval_for_person(evals, person.person_index);
        const PersonTrackState* st = nullptr;
        auto st_it = stream.person_states.find(person.track_id);
        if (st_it != stream.person_states.end()) st = &st_it->second;
        const bool alarm = st != nullptr && st->alarm_triggered;
        const bool risk = ev != nullptr && ev->accepted();
        const bool suspect = st != nullptr && is_suspect_track(*st, stream.frame_id);
        const Rect display_box = st != nullptr ? display_bbox_for_track(*st, person) : person.box;
        json p;
        p["person_index"] = person.person_index;
        p["track_id"] = person.track_id;
        p["box"] = rect_to_json(display_box);
        p["raw_box"] = rect_to_json(person.box);
        p["roi"] = rect_to_json(expand_person_roi(display_box, stream.width, stream.height, person_expand_x, person_expand_y));
        p["visible"] = true;
        p["risk"] = risk;
        p["alarm"] = alarm;
        p["legacy_alarm"] = st != nullptr && st->legacy_alarm_triggered;
        p["suspect"] = suspect;
        p["state"] = st != nullptr ? st->state : "";
        p["gated_review"] = st != nullptr && st->gated_event_decision.review;
        p["gated_handheld_review_hits"] = st != nullptr
            ? st->gated_event_decision.handheld_review_hits : 0;
        p["gated_associated_phone_hits"] = st != nullptr
            ? st->gated_event_decision.associated_phone_hits : 0;
        p["gated_screen_intent_hits"] = st != nullptr
            ? st->gated_event_decision.screen_intent_hits : 0;
        p["gated_reject_reason"] = st != nullptr ? st->gated_event_decision.reject_reason : "";
        p["risk_score"] = ev != nullptr ? ev->risk_score : (st != nullptr ? st->last_risk_score : 0.0f);
        p["window_hits"] = st != nullptr ? st->window_hits : 0;
        p["legacy_window_hits"] = st != nullptr ? st->legacy_window_hits : 0;
        p["handheld_phone_hits"] = st != nullptr ? st->handheld_phone_hits : 0;
        p["handheld_phone_stable_count"] = st != nullptr ? st->handheld_phone_stable_count : 0;
        p["static_phone_suppressed_hits"] = st != nullptr ? st->static_phone_suppressed_hits : 0;
        p["screen_id"] = ev != nullptr ? ev->screen_id : (st != nullptr ? st->last_screen_id : "");
        write_static_diagnostics(p, ev);
        j["persons"].push_back(std::move(p));
    }
    for (const auto& tid : suspect_tracks) {
        if (visible_tracks.count(tid)) continue;
        auto st_it = stream.person_states.find(tid);
        if (st_it == stream.person_states.end()) continue;
        const PersonTrackState& st = st_it->second;
        if (!st.has_last_bbox || stream.frame_id - st.last_seen > SUSPECT_HOLD_FRAMES) continue;
        const Rect display_box = display_bbox_for_state(st);
        json p;
        p["person_index"] = -1;
        p["track_id"] = tid;
        p["box"] = rect_to_json(display_box);
        p["raw_box"] = rect_to_json(st.last_bbox);
        p["roi"] = rect_to_json(expand_person_roi(display_box, stream.width, stream.height, person_expand_x, person_expand_y));
        p["visible"] = false;
        p["risk"] = true;
        p["alarm"] = st.alarm_triggered;
        p["legacy_alarm"] = st.legacy_alarm_triggered;
        p["suspect"] = true;
        p["state"] = st.state;
        p["gated_review"] = st.gated_event_decision.review;
        p["gated_handheld_review_hits"] = st.gated_event_decision.handheld_review_hits;
        p["gated_associated_phone_hits"] = st.gated_event_decision.associated_phone_hits;
        p["gated_screen_intent_hits"] = st.gated_event_decision.screen_intent_hits;
        p["gated_reject_reason"] = st.gated_event_decision.reject_reason;
        p["risk_score"] = st.last_risk_score;
        p["window_hits"] = st.window_hits;
        p["legacy_window_hits"] = st.legacy_window_hits;
        p["handheld_phone_hits"] = st.handheld_phone_hits;
        p["handheld_phone_stable_count"] = st.handheld_phone_stable_count;
        p["static_phone_suppressed_hits"] = st.static_phone_suppressed_hits;
        p["screen_id"] = st.last_screen_id;
        p["last_seen_age"] = stream.frame_id - st.last_seen;
        write_neutral_static_diagnostics(p, "not_primary");
        j["persons"].push_back(std::move(p));
    }

    j["phones"] = json::array();
    for (const auto& ph : phones) {
        const CandidateEval* ev = best_eval_for_phone(evals, ph);
        json p;
        p["box"] = rect_to_json(ph.box);
        p["confidence"] = ph.conf;
        p["person_index"] = ph.person_index;
        p["accepted"] = ev != nullptr && ev->accepted();
        p["alarm"] = ev != nullptr && ev->person_alarm;
        p["risk_score"] = ev != nullptr ? ev->risk_score : 0.0f;
        p["track_id"] = ev != nullptr ? ev->track_id : -1;
        p["screen_id"] = ev != nullptr ? ev->screen_id : "";
        p["level"] = ev != nullptr ? ev->level : "";
        p["reject_reason"] = ev != nullptr ? ev->reject_reason : "";
        p["gated_phone_valid"] = ev != nullptr && ev->gated_phone_valid;
        p["high_confidence_phone"] = ev != nullptr && ev->high_confidence_phone;
        p["gated_person_association"] = ev != nullptr
            ? jiankong::custom_pipeline::evidence_name(ev->gated_person_association) : "unknown";
        p["gated_hand_relation"] = ev != nullptr
            ? jiankong::custom_pipeline::evidence_name(ev->gated_hand_relation) : "unknown";
        p["gated_screen_intent"] = ev != nullptr
            ? jiankong::custom_pipeline::evidence_name(ev->gated_screen_intent) : "unknown";
        p["screen_ray_hit"] = ev != nullptr && ev->screen_ray_hit;
        p["exact_screen_ray_hit"] = ev != nullptr && ev->exact_screen_ray_hit;
        p["corridor_screen_ray_hit"] = ev != nullptr && ev->corridor_screen_ray_hit;
        p["candidate_reason"] = ev != nullptr ? ev->candidate_reason : "";
        p["zone_reason"] = ev != nullptr ? ev->zone_reason : "";
        p["static_zone_score"] = ev != nullptr ? ev->static_zone_score : 0.0f;
        p["person_match_score"] = ev != nullptr ? ev->person_match_score : 0.0f;
        p["phone_score"] = ev != nullptr ? ev->phone_score : 0.0f;
        p["phone_person_area_ratio"] = ev != nullptr ? ev->phone_person_area_ratio : 0.0f;
        p["phone_person_diag_ratio"] = ev != nullptr ? ev->phone_person_diag_ratio : 0.0f;
        p["phone_geometry_valid"] = ev != nullptr && ev->phone_geometry_valid;
        p["phone_hand_score"] = ev != nullptr ? ev->phone_hand_score : 0.0f;
        p["screen_relation_score"] = ev != nullptr ? ev->screen_relation_score : 0.0f;
        p["pose_score"] = ev != nullptr ? ev->pose_score : 0.0f;
        p["temporal_score"] = ev != nullptr ? ev->temporal_score : 0.0f;
        p["best_angle"] = ev != nullptr ? ev->best_angle : 180.0f;
        p["best_ray_hit"] = ev != nullptr && ev->best_ray_hit;
        p["handheld_suspect"] = ev != nullptr && is_handheld_phone_suspect_candidate(*ev);
        p["phone_static"] = ev != nullptr && ev->phone_static;
        p["phone_static_duration"] = ev != nullptr ? ev->phone_static_duration : 0.0f;
        p["phone_motion_px"] = ev != nullptr ? ev->phone_motion_px : 0.0f;
        p["phone_in_desk_zone"] = ev != nullptr && ev->phone_in_desk_zone;
        p["static_suppressed"] = ev != nullptr && ev->static_suppressed;
        p["static_risk_multiplier"] = ev != nullptr ? ev->static_risk_multiplier : 1.0f;
        p["phone_follow_wrist"] = ev != nullptr && ev->phone_follow_wrist;
        p["final_risk_score"] = ev != nullptr ? ev->risk_score : 0.0f;
        p["static_cluster_samples"] = ev != nullptr ? ev->static_cluster_samples : 0;
        p["static_detection_ratio"] = ev != nullptr ? ev->static_detection_ratio : 0.0f;
        p["static_center_spread_px"] = ev != nullptr ? ev->static_center_spread_px : 0.0f;
        p["static_bbox_iou_median"] = ev != nullptr ? ev->static_bbox_iou_median : 0.0f;
        p["static_pending"] = ev != nullptr && ev->static_pending;
        p["static_pending_duration"] = ev != nullptr ? ev->static_pending_duration : 0.0;
        p["static_hotspot_score"] = ev != nullptr ? ev->static_hotspot_score : 0.0f;
        p["static_context_reason"] = ev != nullptr
            ? spatial_static_context_reason(ev->static_context_reason.c_str()) : "disabled";
        p["static_shadow_hits"] = ev != nullptr ? ev->static_shadow_hits : 0;
        p["static_suppressed"] = ev != nullptr && ev->static_suppressed;
        p["static_exit_reason"] = ev != nullptr
            ? spatial_static_exit_reason(ev->static_exit_reason.c_str()) : "none";
        p["fixed_template_near"] = ev != nullptr && ev->fixed_template_near;
        p["fixed_template_match"] = ev != nullptr && ev->fixed_template_match;
        p["fixed_template_score"] =
            ev != nullptr ? ev->fixed_template_score : 0.0f;
        p["fixed_template_id"] =
            ev != nullptr ? ev->fixed_template_id : "";
        j["phones"].push_back(std::move(p));
    }

    jsonl << j.dump() << '\n';
    csv << stream_index << ','
        << frame_index << ','
        << std::fixed << std::setprecision(3) << time_sec << ','
        << accepted_count << ','
        << alarm_tracks.size() << ','
        << std::setprecision(4) << max_risk << ','
        << max_hits << ','
        << people.size() << ','
        << phones.size() << ','
        << std::quoted(stream.output_path) << ','
        << std::quoted(stream.path) << '\n';
}

static int run_self_test() {
    Det a;
    a.box = Rect{0, 0, 100, 200};
    a.conf = 0.80f;
    a.kpts[5][2] = a.kpts[6][2] = a.kpts[9][2] = 0.90f;
    Det b;
    b.box = Rect{8, 12, 95, 190};
    b.conf = 0.75f;
    b.kpts[5][2] = b.kpts[6][2] = b.kpts[10][2] = 0.90f;
    Det c;
    c.box = Rect{220, 30, 320, 210};
    c.conf = 0.70f;
    c.kpts[5][2] = c.kpts[6][2] = c.kpts[9][2] = 0.90f;
    auto people = dedupe_people({a, b, c}, 0.35f);
    if (people.size() != 2) {
        std::cerr << "[SELF_TEST] expected 2 deduped people, got " << people.size() << std::endl;
        return 2;
    }
    Det phone;
    phone.box = Rect{20, 20, 35, 45};
    if (!phone_hits_roi(phone, Rect{0, 0, 100, 100})) {
        std::cerr << "[SELF_TEST] phone should hit ROI" << std::endl;
        return 3;
    }
    if (phone_hits_roi(phone, Rect{200, 200, 300, 300})) {
        std::cerr << "[SELF_TEST] phone should not hit far ROI" << std::endl;
        return 4;
    }
    cv::Mat screen_test(120, 160, CV_8UC3, cv::Scalar(0, 0, 0));
    draw_poly(screen_test, {{20, 20}, {140, 24}, {130, 96}, {28, 88}}, cv::Scalar(255, 80, 0), 2, "SCREEN1", 0.30);
    int screen_pixels = 0;
    for (int y = 0; y < screen_test.rows; ++y) {
        const cv::Vec3b* row = screen_test.ptr<cv::Vec3b>(y);
        for (int x = 0; x < screen_test.cols; ++x) {
            if (row[x][0] > 150 && row[x][1] > 20 && row[x][2] < 120) ++screen_pixels;
        }
    }
    if (screen_pixels < 20) {
        std::cerr << "[SELF_TEST] screen polygon draw produced too few pixels: " << screen_pixels << std::endl;
        return 5;
    }
    LiveWallClockAnchor live_clock_anchor;
    jiankong::custom_pipeline::DeviceFrameView live_clock_frame;
    live_clock_frame.pts_ns = 10000000000ULL;
    live_clock_frame.pts_valid = true;
    live_clock_frame.received_at_unix_seconds = 1000.0;
    if (std::abs(live_frame_capture_unix_seconds(live_clock_anchor, live_clock_frame) - 1000.0) > 1e-9) {
        std::cerr << "[SELF_TEST] live wall-clock PTS anchor self-test failed" << std::endl;
        return 6;
    }
    live_clock_frame.pts_ns = 10125000000ULL;
    live_clock_frame.received_at_unix_seconds = 1000.130;
    if (std::abs(live_frame_capture_unix_seconds(live_clock_anchor, live_clock_frame) - 1000.125) > 1e-9) {
        std::cerr << "[SELF_TEST] live wall-clock PTS anchor self-test failed" << std::endl;
        return 6;
    }
    live_clock_frame.pts_ns = 10250000000ULL;
    live_clock_frame.received_at_unix_seconds = 1008.0;
    if (std::abs(live_frame_capture_unix_seconds(live_clock_anchor, live_clock_frame) - 1008.0) > 1e-9 ||
        live_clock_anchor.anchor_pts_ns != live_clock_frame.pts_ns) {
        std::cerr << "[SELF_TEST] live wall-clock drift re-anchor self-test failed" << std::endl;
        return 7;
    }
    live_clock_frame.pts_ns = std::numeric_limits<std::uint64_t>::max();
    live_clock_frame.pts_valid = false;
    live_clock_frame.received_at_unix_seconds = 1009.0;
    if (std::abs(live_frame_capture_unix_seconds(live_clock_anchor, live_clock_frame) - 1009.0) > 1e-9 ||
        live_clock_anchor.initialized) {
        std::cerr << "[SELF_TEST] invalid PTS uses receive wall-clock self-test failed" << std::endl;
        return 8;
    }
    live_clock_frame.pts_ns = 2000000000ULL;
    live_clock_frame.pts_valid = true;
    live_clock_frame.received_at_unix_seconds = 1011.0;
    if (std::abs(live_frame_capture_unix_seconds(live_clock_anchor, live_clock_frame) - 1011.0) > 1e-9) {
        std::cerr << "[SELF_TEST] post-invalid PTS re-anchor self-test failed" << std::endl;
        return 9;
    }
    live_clock_frame.pts_ns = 1000000000ULL;
    live_clock_frame.received_at_unix_seconds = 1010.0;
    if (std::abs(live_frame_capture_unix_seconds(live_clock_anchor, live_clock_frame) - 1010.0) > 1e-9) {
        std::cerr << "[SELF_TEST] live wall-clock PTS reset self-test failed" << std::endl;
        return 10;
    }
    std::cout << "[SELF_TEST] ok" << std::endl;
    return 0;
}

static int run_rule_self_test() {
    ScreenConfig screen;
    screen.screen_id = "screen_01";
    screen.screen_poly = {{360, 100}, {500, 100}, {500, 250}, {360, 250}};
    screen.near_zone = {{240, 60}, {560, 60}, {560, 310}, {240, 310}};
    screen.danger_zones.push_back(Zone{"front", {{240, 60}, {560, 60}, {560, 310}, {240, 310}}, 1.0f});
    screen.params.person_state_window = 30;
    screen.params.person_state_min_hits = 16;
    screen.params.person_state_risk_threshold = 0.65f;

    Det person;
    person.box = Rect{120, 80, 230, 420};
    person.conf = 0.90f;
    person.person_index = 0;
    person.kpts[5][0] = 150; person.kpts[5][1] = 150; person.kpts[5][2] = 0.95f;
    person.kpts[6][0] = 215; person.kpts[6][1] = 150; person.kpts[6][2] = 0.95f;
    person.kpts[8][0] = 235; person.kpts[8][1] = 175; person.kpts[8][2] = 0.95f;
    person.kpts[10][0] = 270; person.kpts[10][1] = 190; person.kpts[10][2] = 0.95f;
    std::vector<Det> people{person};
    std::map<int, PersonTrackState> states;
    int next_track = assign_person_track_ids(people, states, 1, 1, screen.params.person_state_window);
    if (next_track != 2 || people[0].track_id != 1) {
        std::cerr << "[SELF_TEST_RULES] track assignment failed" << std::endl;
        return 10;
    }

    Det risky_phone;
    risky_phone.box = Rect{260, 172, 296, 210};
    risky_phone.conf = 0.95f;
    risky_phone.person_index = 0;
    CandidateEval risky = evaluate_phone(risky_phone, screen, people, states, 640, 480, 0.35f);
    if (!risky.accepted() || risky.risk_score < 0.65f || risky.phone_hand_score < 0.9f) {
        std::cerr << "[SELF_TEST_RULES] expected accepted high-risk candidate, risk="
                  << risky.risk_score << " reject=" << risky.reject_reason << std::endl;
        return 11;
    }

    Det roi_only_phone;
    roi_only_phone.box = Rect{130, 350, 155, 375};
    roi_only_phone.conf = 0.95f;
    roi_only_phone.person_index = 0;
    CandidateEval roi_only = evaluate_phone(roi_only_phone, screen, people, states, 640, 480, 0.35f);
    if (roi_only.accepted()) {
        std::cerr << "[SELF_TEST_RULES] ROI-only phone should not be accepted, risk="
                  << roi_only.risk_score << std::endl;
        return 12;
    }

    Det oversized_false_phone;
    oversized_false_phone.box = Rect{200, 120, 340, 260};
    oversized_false_phone.conf = 0.92f;
    oversized_false_phone.person_index = 0;
    CandidateEval oversized_false = evaluate_phone(
        oversized_false_phone, screen, people, states, 640, 480, 0.35f);
    if (oversized_false.accepted() || oversized_false.gated_phone_valid ||
        oversized_false.reject_reason != "implausible_phone_geometry") {
        std::cerr << "[SELF_TEST_RULES] oversized false phone geometry should be rejected"
                  << " area_ratio=" << oversized_false.phone_person_area_ratio
                  << " diag_ratio=" << oversized_false.phone_person_diag_ratio
                  << " reject=" << oversized_false.reject_reason << std::endl;
        return 23;
    }

    // This block validates the production gated alarm independently.
    // Spatial-static behavior is covered by the dedicated cases below.
    states[people[0].track_id].spatial_static_enabled = false;
    for (int frame_id = 1; frame_id <= screen.params.person_state_min_hits; ++frame_id) {
        risky.track_id = people[0].track_id;
        std::vector<CandidateEval> risky_evals{risky};
        update_person_states(people, risky_evals, states, frame_id, screen.params.person_state_window,
                             screen.params.person_state_risk_threshold, screen.params.person_state_min_hits,
                             screen.params.handheld_suspect_min_hits, 10.0);
    }
    if (!states[people[0].track_id].alarm_triggered ||
        states[people[0].track_id].window_hits < 8) {
        std::cerr << "[SELF_TEST_RULES] temporal alarm did not trigger" << std::endl;
        return 13;
    }
    if (!is_suspect_track(states[people[0].track_id], screen.params.person_state_min_hits + 3, 10)) {
        std::cerr << "[SELF_TEST_RULES] suspect track should stay active after risk frames" << std::endl;
        return 14;
    }

    Det shifted_person = people[0];
    shifted_person.box = Rect{150, 110, 260, 450};
    shifted_person.track_id = people[0].track_id;
    shifted_person.person_index = people[0].person_index;
    std::vector<Det> shifted_people{shifted_person};
    std::vector<CandidateEval> no_evals;
    update_person_states(shifted_people, no_evals, states, screen.params.person_state_min_hits + 1,
                         screen.params.person_state_window,
                         screen.params.person_state_risk_threshold,
                         screen.params.person_state_min_hits,
                         screen.params.handheld_suspect_min_hits, 10.0);
    const Rect display_box = display_bbox_for_track(states[people[0].track_id], shifted_person);
    if (std::abs(display_box.x1 - shifted_person.box.x1) < 1.0f) {
        std::cerr << "[SELF_TEST_RULES] display box should be smoothed, got raw x1=" << display_box.x1 << std::endl;
        return 15;
    }

    ScreenConfig side_screen = screen;
    side_screen.screen_id = "side_screen";
    side_screen.screen_poly = {{520, 80}, {620, 80}, {620, 260}, {520, 260}};
    side_screen.near_zone = {{500, 60}, {640, 60}, {640, 300}, {500, 300}};
    side_screen.danger_zones.clear();
    side_screen.params.handheld_suspect_min_hits = 4;
    Det side_person = person;
    side_person.person_index = 2;
    std::vector<Det> side_people{side_person};
    std::map<int, PersonTrackState> side_states;
    assign_person_track_ids(side_people, side_states, 10, 1, side_screen.params.person_state_window);
    Det side_phone;
    side_phone.box = Rect{252, 178, 288, 212};
    side_phone.conf = 0.90f;
    side_phone.person_index = side_people[0].person_index;
    CandidateEval side_eval = evaluate_phone(side_phone, side_screen, side_people, side_states, 640, 480, 0.35f);
    if (side_eval.accepted() || side_eval.gated_phone_valid) {
        std::cerr << "[SELF_TEST_RULES] side-screen phone should be rejected, risk="
                  << side_eval.risk_score << " reject=" << side_eval.reject_reason
                  << " reason=" << side_eval.candidate_reason << std::endl;
        return 16;
    }
    for (int frame_id = 1; frame_id <= side_screen.params.handheld_suspect_min_hits; ++frame_id) {
        std::vector<CandidateEval> side_evals{side_eval};
        update_person_states(side_people, side_evals, side_states, frame_id, side_screen.params.person_state_window,
                             side_screen.params.person_state_risk_threshold, side_screen.params.person_state_min_hits,
                             side_screen.params.handheld_suspect_min_hits, 10.0);
    }
    const auto& side_state = side_states[side_people[0].track_id];
    if (side_state.alarm_triggered) {
        std::cerr << "[SELF_TEST_RULES] side-screen phone must not alarm, hits="
                  << side_state.handheld_phone_hits << " alarm=" << side_state.alarm_triggered << std::endl;
        return 17;
    }

    const auto shifted_pose_person = [](const Det& source, float dx, int person_index) {
        Det shifted = source;
        shifted.person_index = person_index;
        shifted.box.x1 += dx;
        shifted.box.x2 += dx;
        for (auto& keypoint : shifted.kpts) {
            if (keypoint[2] > 0.0f) keypoint[0] += dx;
        }
        return shifted;
    };
    const auto shifted_risky_eval = [&](float phone_dx, int track_id, int person_index) {
        CandidateEval shifted = risky;
        shifted.track_id = track_id;
        shifted.person_index = person_index;
        shifted.phone.box.x1 += phone_dx;
        shifted.phone.box.x2 += phone_dx;
        return shifted;
    };

    std::vector<Det> static_people{shifted_pose_person(person, 0.0f, 3)};
    std::map<int, PersonTrackState> static_states;
    assign_person_track_ids(static_people, static_states, 20, 0, screen.params.person_state_window);
    const int static_track_id = static_people[0].track_id;
    std::vector<CandidateEval> static_evals;
    bool saw_primary_dropout = false;
    bool saw_primary_shadow = false;
    bool shadow_bound_to_primary = true;
    bool saw_primary_suppressed = false;
    bool saw_secondary_formal_hit = false;
    for (int frame_id = 0; frame_id <= 26; ++frame_id) {
        static_people[0] = shifted_pose_person(person, frame_id * 1.5f, 3);
        static_people[0].track_id = static_track_id;
        static_evals.clear();
        const bool primary_detected = frame_id != 21;
        const bool secondary_detected = frame_id % 3 == 0;
        std::size_t primary_index = 0;
        std::size_t secondary_index = 0;
        const auto append_primary = [&]() {
            primary_index = static_evals.size();
            const float jitter = static_cast<float>((frame_id % 3) - 1);
            CandidateEval primary = shifted_risky_eval(jitter, static_track_id, 3);
            primary.phone.conf = 0.55f;
            primary.risk_score = screen.params.person_state_risk_threshold + 0.01f;
            static_evals.push_back(primary);
        };
        const auto append_secondary = [&]() {
            secondary_index = static_evals.size();
            CandidateEval secondary = shifted_risky_eval(
                90.0f + frame_id * 12.0f, static_track_id, 3);
            secondary.phone.conf = 0.99f;
            secondary.risk_score = screen.params.person_state_risk_threshold + 0.15f;
            static_evals.push_back(secondary);
        };
        if (frame_id % 2 == 0) {
            if (primary_detected) append_primary();
            if (secondary_detected) append_secondary();
        } else {
            if (secondary_detected) append_secondary();
            if (primary_detected) append_primary();
        }
        const std::size_t shadow_before = static_states[static_track_id].static_shadow_history.size();
        update_person_states(static_people, static_evals, static_states, frame_id,
                             screen.params.person_state_window,
                             screen.params.person_state_risk_threshold,
                             screen.params.person_state_min_hits,
                             screen.params.handheld_suspect_min_hits, 8.0);
        const auto& frame_state = static_states[static_track_id];
        if (frame_id >= 6 && primary_detected) {
            if (!frame_state.last_static_decision.primary_candidate_index.has_value() ||
                *frame_state.last_static_decision.primary_candidate_index != primary_index) {
                std::cerr << "[SELF_TEST_RULES] spatial primary index mismatch frame="
                          << frame_id << std::endl;
                return 18;
            }
        }
        if (!primary_detected && secondary_detected) {
            saw_primary_dropout =
                !frame_state.last_static_decision.primary_candidate_index.has_value() &&
                frame_state.static_shadow_history.size() == shadow_before;
        }
        if (frame_state.last_static_decision.hold_candidate && primary_detected &&
            !frame_state.static_shadow_history.empty()) {
            saw_primary_shadow = true;
            shadow_bound_to_primary = shadow_bound_to_primary && std::abs(
                frame_state.static_shadow_history.back().risk_score -
                static_evals[primary_index].risk_score) < 1e-6f;
        }
        if (frame_state.last_static_decision.phase ==
                jiankong::custom_pipeline::SpatialStaticPhase::suppressed &&
            primary_detected && secondary_detected) {
            saw_primary_suppressed = static_evals[primary_index].static_suppressed &&
                !static_evals[secondary_index].static_suppressed;
            saw_secondary_formal_hit = !frame_state.candidate_history.empty() &&
                frame_state.candidate_history.back() == 1 &&
                std::abs(frame_state.risk_history.back() -
                         static_evals[secondary_index].risk_score) < 1e-6f;
        }
    }
    const auto& static_state = static_states[static_track_id];
    if (!saw_primary_dropout || !saw_primary_shadow || !shadow_bound_to_primary ||
        !saw_primary_suppressed ||
        !saw_secondary_formal_hit || static_state.alarm_triggered ||
        static_state.last_static_decision.phase !=
            jiankong::custom_pipeline::SpatialStaticPhase::suppressed) {
        std::cerr << "[SELF_TEST_RULES] spatial static suppression phase failed"
                  << " phase=" << static_cast<int>(static_state.last_static_decision.phase)
                  << " samples=" << static_state.last_static_decision.metrics.sample_count
                  << " ratio=" << static_state.last_static_decision.metrics.detection_ratio
                  << " spread=" << static_state.last_static_decision.metrics.center_spread_px
                  << " dropout=" << saw_primary_dropout
                  << " primary_shadow=" << saw_primary_shadow
                  << " shadow_bound=" << shadow_bound_to_primary
                  << " primary_suppressed=" << saw_primary_suppressed
                  << " secondary_formal=" << saw_secondary_formal_hit
                  << " hits=" << static_state.window_hits
                  << " alarm=" << static_state.alarm_triggered << std::endl;
        return 18;
    }

    std::vector<Det> history_people{shifted_pose_person(person, 0.0f, 6)};
    std::map<int, PersonTrackState> history_states;
    assign_person_track_ids(history_people, history_states, 50, 0,
                            screen.params.person_state_window);
    const int history_track_id = history_people[0].track_id;
    std::vector<CandidateEval> history_evals;
    for (int frame_id = 0; frame_id <= 27; ++frame_id) {
        history_people[0] = shifted_pose_person(person, frame_id * 1.5f, 6);
        history_people[0].track_id = history_track_id;
        history_evals.clear();
        if (frame_id < 3) {
            CandidateEval secondary = shifted_risky_eval(
                140.0f + frame_id * 40.0f, history_track_id, 6);
            secondary.phone.conf = 0.99f;
            secondary.risk_score = screen.params.person_state_risk_threshold + 0.15f;
            history_evals.push_back(secondary);
        } else {
            CandidateEval primary = shifted_risky_eval(0.0f, history_track_id, 6);
            primary.phone.conf = 0.55f;
            primary.risk_score = screen.params.person_state_risk_threshold + 0.01f;
            history_evals.push_back(primary);
        }
        update_person_states(history_people, history_evals, history_states, frame_id,
                             screen.params.person_state_window,
                             screen.params.person_state_risk_threshold,
                             screen.params.person_state_min_hits,
                             screen.params.handheld_suspect_min_hits, 8.0);
    }
    const auto& history_state = history_states[history_track_id];
    const std::size_t history_size = history_state.risk_history.size();
    const bool histories_aligned = history_size == history_state.candidate_history.size() &&
        history_size == history_state.handheld_history.size() &&
        history_size == history_state.formal_spatial_history.size();
    bool secondary_history_retained = histories_aligned && history_size == 28;
    for (std::size_t index = 0; secondary_history_retained && index < 3; ++index) {
        secondary_history_retained = history_state.candidate_history[index] == 1 &&
            history_state.risk_history[index] > screen.params.person_state_risk_threshold &&
            history_state.formal_spatial_history[index].valid;
    }
    bool primary_history_cleared = histories_aligned;
    for (std::size_t index = 3; primary_history_cleared && index < history_size; ++index) {
        primary_history_cleared = history_state.candidate_history[index] == 0 &&
            history_state.risk_history[index] == 0.0f;
    }
    bool primary_centers_exclude_secondary = false;
    if (!history_state.last_static_decision.primary_cluster_centers.empty() &&
        histories_aligned && !history_state.formal_spatial_history.empty()) {
        const auto& secondary = history_state.formal_spatial_history.front();
        primary_centers_exclude_secondary = secondary.valid;
        for (const auto& center : history_state.last_static_decision.primary_cluster_centers) {
            if (spatial_point_distance(secondary, center) <=
                history_state.last_static_decision.primary_cluster_tolerance_px) {
                primary_centers_exclude_secondary = false;
                break;
            }
        }
    }
    if (history_evals.empty() || !history_evals[0].static_suppressed ||
        history_state.last_static_decision.phase !=
            jiankong::custom_pipeline::SpatialStaticPhase::suppressed ||
        !history_state.static_shadow_history.empty() || !histories_aligned ||
        !secondary_history_retained || !primary_history_cleared ||
        !primary_centers_exclude_secondary || history_state.legacy_window_hits != 3) {
        std::cerr << "[SELF_TEST_RULES] spatial secondary formal history preservation failed"
                  << " phase=" << static_cast<int>(history_state.last_static_decision.phase)
                  << " suppressed="
                  << (history_evals.empty() ? 0 : history_evals[0].static_suppressed)
                  << " aligned=" << histories_aligned
                  << " secondary_retained=" << secondary_history_retained
                  << " primary_cleared=" << primary_history_cleared
                  << " primary_centers_exclude_secondary="
                  << primary_centers_exclude_secondary
                  << " history=" << history_size
                  << " legacy_hits=" << history_state.legacy_window_hits << std::endl;
        return 21;
    }

    std::vector<Det> diagonal_people{shifted_pose_person(person, 0.0f, 7)};
    std::map<int, PersonTrackState> diagonal_states;
    assign_person_track_ids(diagonal_people, diagonal_states, 60, 0,
                            screen.params.person_state_window);
    const int diagonal_track_id = diagonal_people[0].track_id;
    const float diagonal_radius = std::max(
        8.0f, 0.03f * (person.box.y2 - person.box.y1));
    const float diagonal_offset = 1.5f * diagonal_radius;
    std::vector<CandidateEval> diagonal_evals;
    for (int frame_id = 0; frame_id <= 27; ++frame_id) {
        diagonal_people[0] = shifted_pose_person(person, frame_id * 1.5f, 7);
        diagonal_people[0].track_id = diagonal_track_id;
        diagonal_evals.clear();
        if (frame_id < 3) {
            CandidateEval secondary = shifted_risky_eval(
                diagonal_offset, diagonal_track_id, 7);
            secondary.phone.box.y1 += diagonal_offset;
            secondary.phone.box.y2 += diagonal_offset;
            secondary.phone.conf = 0.99f;
            secondary.risk_score = screen.params.person_state_risk_threshold + 0.15f;
            diagonal_evals.push_back(secondary);
        } else {
            CandidateEval primary = shifted_risky_eval(0.0f, diagonal_track_id, 7);
            primary.phone.conf = 0.55f;
            primary.risk_score = screen.params.person_state_risk_threshold + 0.01f;
            diagonal_evals.push_back(primary);
        }
        update_person_states(diagonal_people, diagonal_evals, diagonal_states, frame_id,
                             screen.params.person_state_window,
                             screen.params.person_state_risk_threshold,
                             screen.params.person_state_min_hits,
                             screen.params.handheld_suspect_min_hits, 8.0);
    }
    const auto& diagonal_state = diagonal_states[diagonal_track_id];
    const std::size_t diagonal_size = diagonal_state.risk_history.size();
    const bool diagonal_aligned = diagonal_size == diagonal_state.candidate_history.size() &&
        diagonal_size == diagonal_state.handheld_history.size() &&
        diagonal_size == diagonal_state.formal_spatial_history.size();
    bool diagonal_secondary_retained = diagonal_aligned && diagonal_size == 28;
    for (std::size_t index = 0; diagonal_secondary_retained && index < 3; ++index) {
        diagonal_secondary_retained = diagonal_state.candidate_history[index] == 1 &&
            diagonal_state.risk_history[index] > screen.params.person_state_risk_threshold &&
            diagonal_state.formal_spatial_history[index].valid;
    }
    bool diagonal_primary_cleared = diagonal_aligned;
    for (std::size_t index = 3; diagonal_primary_cleared && index < diagonal_size; ++index) {
        diagonal_primary_cleared = diagonal_state.candidate_history[index] == 0 &&
            diagonal_state.risk_history[index] == 0.0f;
    }
    bool diagonal_secondary_matches = false;
    if (diagonal_aligned && !diagonal_state.formal_spatial_history.empty()) {
        const auto& secondary = diagonal_state.formal_spatial_history.front();
        for (const auto& center : diagonal_state.last_static_decision.primary_cluster_centers) {
            if (spatial_point_distance(secondary, center) <=
                diagonal_state.last_static_decision.primary_cluster_tolerance_px) {
                diagonal_secondary_matches = true;
                break;
            }
        }
    }
    const bool diagonal_tolerance_matches = std::abs(
        diagonal_state.last_static_decision.primary_cluster_tolerance_px -
        2.0f * diagonal_radius) < 1e-4f;
    if (diagonal_evals.empty() || !diagonal_evals[0].static_suppressed ||
        diagonal_state.last_static_decision.phase !=
            jiankong::custom_pipeline::SpatialStaticPhase::suppressed ||
        diagonal_state.last_static_decision.primary_cluster_centers.empty() ||
        !diagonal_tolerance_matches || diagonal_secondary_matches ||
        !diagonal_secondary_retained || !diagonal_primary_cleared ||
        diagonal_state.legacy_window_hits != 3) {
        std::cerr << "[SELF_TEST_RULES] spatial diagonal secondary formal history preservation failed"
                  << " phase=" << static_cast<int>(diagonal_state.last_static_decision.phase)
                  << " suppressed="
                  << (diagonal_evals.empty() ? 0 : diagonal_evals[0].static_suppressed)
                  << " centers="
                  << diagonal_state.last_static_decision.primary_cluster_centers.size()
                  << " tolerance="
                  << diagonal_state.last_static_decision.primary_cluster_tolerance_px
                  << " tolerance_matches=" << diagonal_tolerance_matches
                  << " secondary_matches=" << diagonal_secondary_matches
                  << " secondary_retained=" << diagonal_secondary_retained
                  << " primary_cleared=" << diagonal_primary_cleared
                  << " history=" << diagonal_size
                  << " legacy_hits=" << diagonal_state.legacy_window_hits << std::endl;
        return 22;
    }

    std::vector<Det> handheld_people{shifted_pose_person(person, 0.0f, 4)};
    std::map<int, PersonTrackState> handheld_states;
    assign_person_track_ids(handheld_people, handheld_states, 30, 0, screen.params.person_state_window);
    const int handheld_track_id = handheld_people[0].track_id;
    bool handheld_replayed = false;
    for (int frame_id = 0; frame_id < 24; ++frame_id) {
        const float motion = frame_id <= 6 ? 0.0f : (frame_id - 6) * 12.0f;
        handheld_people[0] = shifted_pose_person(person, motion, 4);
        handheld_people[0].track_id = handheld_track_id;
        std::vector<CandidateEval> handheld_evals{
            shifted_risky_eval(motion, handheld_track_id, 4)};
        update_person_states(handheld_people, handheld_evals, handheld_states, frame_id,
                             screen.params.person_state_window,
                             screen.params.person_state_risk_threshold,
                             screen.params.person_state_min_hits,
                             screen.params.handheld_suspect_min_hits, 8.0);
        handheld_replayed = handheld_replayed ||
            handheld_states[handheld_track_id].last_static_decision.replay_shadow;
    }
    const auto& handheld_state = handheld_states[handheld_track_id];
    if (!handheld_replayed || !handheld_state.static_shadow_history.empty() ||
        !handheld_state.alarm_triggered ||
        handheld_state.window_hits < 8 || handheld_state.legacy_window_hits != 24 ||
        handheld_state.risk_history.size() != 24 ||
        handheld_state.candidate_history.size() != 24) {
        std::cerr << "[SELF_TEST_RULES] spatial handheld replay phase failed"
                  << " phase=" << static_cast<int>(handheld_state.last_static_decision.phase)
                  << " replayed=" << handheld_replayed
                  << " shadow=" << handheld_state.static_shadow_history.size()
                   << " hits=" << handheld_state.window_hits
                   << " legacy_hits=" << handheld_state.legacy_window_hits
                  << " history=" << handheld_state.candidate_history.size()
                  << " alarm=" << handheld_state.alarm_triggered << std::endl;
        return 19;
    }

    std::vector<Det> ambiguous_people{shifted_pose_person(person, 0.0f, 5)};
    std::map<int, PersonTrackState> ambiguous_states;
    assign_person_track_ids(ambiguous_people, ambiguous_states, 40, 0, screen.params.person_state_window);
    const int ambiguous_track_id = ambiguous_people[0].track_id;
    bool ambiguous_replayed = false;
    for (int frame_id = 0; frame_id <= 24; ++frame_id) {
        ambiguous_people[0].track_id = ambiguous_track_id;
        std::vector<CandidateEval> ambiguous_evals{
            shifted_risky_eval(0.0f, ambiguous_track_id, 5)};
        update_person_states(ambiguous_people, ambiguous_evals, ambiguous_states, frame_id,
                             screen.params.person_state_window,
                             screen.params.person_state_risk_threshold,
                             screen.params.person_state_min_hits,
                             screen.params.handheld_suspect_min_hits, 8.0);
        ambiguous_replayed = ambiguous_replayed ||
            ambiguous_states[ambiguous_track_id].last_static_decision.replay_shadow;
    }
    const auto& ambiguous_state = ambiguous_states[ambiguous_track_id];
    if (!ambiguous_replayed || !ambiguous_state.static_shadow_history.empty() ||
        ambiguous_state.last_static_decision.phase ==
            jiankong::custom_pipeline::SpatialStaticPhase::suppressed ||
        ambiguous_state.legacy_window_hits < screen.params.person_state_min_hits) {
        std::cerr << "[SELF_TEST_RULES] spatial ambiguous replay phase failed"
                  << " phase=" << static_cast<int>(ambiguous_state.last_static_decision.phase)
                  << " replayed=" << ambiguous_replayed
                  << " shadow=" << ambiguous_state.static_shadow_history.size()
                  << " samples=" << ambiguous_state.last_static_decision.metrics.sample_count
                  << " ratio=" << ambiguous_state.last_static_decision.metrics.detection_ratio
                  << " legacy_hits=" << ambiguous_state.legacy_window_hits << std::endl;
        return 20;
    }

    std::cout << "[SELF_TEST_RULES] ok risk=" << std::fixed << std::setprecision(3)
              << risky.risk_score << " roi_only_reject=" << roi_only.reject_reason << std::endl;
    return 0;
}

static void print_live_usage() {
    std::cout
        << "RTSP live mode:\n"
        << "  --rtsp name=rtsp://host:port/path  repeat 1..engine batch times\n"
        << "  --camera-manifest FILE --relay-base rtsp://host:port  managed camera inventory\n"
        << "  --duration-sec N                   default 300\n"
        << "  --infer-fps N                      file mode default 10; live mode requires exactly 8\n"
        << "  --reconnect-delay-ms N             default 1000\n"
        << "  --open-timeout-ms N                default 5000\n"
        << "  --read-timeout-ms N                default 3000\n"
        << "  --source-width N --source-height N defaults 2560x1440\n"
        << "  --output-dir DIR                   event metadata and live stats only\n"
        << "  --enable-spatial-static-phone-suppression  default enabled\n"
        << "  --disable-spatial-static-phone-suppression one-switch rollback\n"
        << "  --static-observation-seconds N     default 3.0\n"
        << "  --static-pending-seconds N         default 0.75\n"
        << "  --static-long-confirm-seconds N    default 6.0\n"
        << "  --static-min-detection-ratio N     default 0.60\n"
        << "  --static-max-gap-seconds N         default 0.75\n"
        << "  --static-position-radius-ratio N   default 0.03\n"
        << "  --static-hotspot-enabled|--static-hotspot-disabled  default enabled\n"
        << "  --fixed-template-dir DIR          reviewed fixed-phone templates\n"
        << "  --live-self-test                   CPU-only argument/runtime check\n";
}

static int run_live_self_test(const Args& args) {
    if (!args.camera_manifest.empty() && !args.rtsp_specs.empty()) {
        std::cerr << "[LIVE_SELF_TEST] camera manifest cannot be combined with --rtsp" << std::endl;
        return 2;
    }
    if (!args.rtsp_specs.empty() && args.rtsp_specs.size() > 8) {
        std::cerr << "[LIVE_SELF_TEST] expected at most 8 --rtsp entries, got "
                  << args.rtsp_specs.size() << std::endl;
        return 3;
    }
    std::set<std::string> names;
    for (const auto& value : args.rtsp_specs) {
        const auto spec = jiankong::parse_stream_spec(value);
        if (!names.insert(spec.name).second) {
            std::cerr << "[LIVE_SELF_TEST] duplicate stream name " << spec.name << std::endl;
            return 4;
        }
    }
    jiankong::RateGate gate(8.0);
    if (!gate.accept(1.0) || gate.accept(1.05) || !gate.accept(1.126)) {
        std::cerr << "[LIVE_SELF_TEST] rate gate failed" << std::endl;
        return 5;
    }
    jiankong::LatestFrameSlot<int> slot;
    slot.push(1, 1, 1.0);
    if (!slot.push(2, 2, 1.1) || slot.dropped() != 1) {
        std::cerr << "[LIVE_SELF_TEST] latest-frame drop failed" << std::endl;
        return 6;
    }
    const size_t streams = args.camera_manifest.empty()
        ? args.rtsp_specs.size()
        : load_camera_manifest(args.camera_manifest).size();
    std::cout << "[LIVE_SELF_TEST] ok streams=" << streams << std::endl;
    return 0;
}
int main(int argc, char** argv) {
    try {
        Args args = parse_args(argc, argv);
        const auto spatial_static_config = spatial_static_config_from_args(args);
        if (args.show_help) {
            print_live_usage();
            return 0;
        }
        if (args.self_test) return run_self_test();
        if (args.self_test_rules) return run_rule_self_test();
        if (args.live_self_test) return run_live_self_test(args);
        cv::setNumThreads(std::max(0, args.cv_threads));
        FixedPhoneTemplateRegistry fixed_template_registry(args.fixed_template_dir);

        const bool live_mode = !args.rtsp_specs.empty() || !args.camera_manifest.empty();
        if (live_mode) {
            args.no_video = true;
            args.pipelined_read = false;
            args.gpu_preprocess = true;
            if (!jiankong::custom_pipeline::live_infer_fps_is_supported(args.infer_fps)) {
                throw std::runtime_error("live mode requires --infer-fps 8 exactly");
            }
            if (args.duration_sec <= 0) throw std::runtime_error("--duration-sec must be positive");
            if (args.output_dir.empty()) throw std::runtime_error("live mode requires --output-dir");
        }

        TrtRunner pose(args.pose_plan);
        TrtRunner phone(args.phone_plan);
        args.pose_batch = pose.input_dims().d[0];
        args.pose_size = pose.input_dims().d[2];
        args.phone_batch = phone.input_dims().d[0];
        args.phone_size = phone.input_dims().d[2];
        const auto pose_out_dims = pose.output_dims();
        const auto phone_out_dims = phone.output_dims();
        const int pose_channels = pose_out_dims.d[1];
        const int pose_boxes = pose_out_dims.d[2];
        const int phone_channels = phone_out_dims.d[1];
        const int phone_boxes = phone_out_dims.d[2];

        std::vector<jiankong::StreamSpec> live_specs;
        std::vector<std::string> videos;
        std::map<std::string, std::string> managed_calibrations;
        std::map<std::string, bool> managed_has_screen;
        const bool managed_camera_manifest = !args.camera_manifest.empty();
        if (live_mode) {
            if (managed_camera_manifest) {
                if (!args.rtsp_specs.empty() || args.relay_base.empty()) {
                    throw std::runtime_error("camera manifest requires relay base and cannot be combined with --rtsp");
                }
                std::string base = args.relay_base;
                while (!base.empty() && base.back() == '/') base.pop_back();
                if (base.rfind("rtsp://", 0) != 0) {
                    throw std::runtime_error("camera manifest relay base must be rtsp://");
                }
                for (const auto& camera : load_camera_manifest(args.camera_manifest)) {
                    args.rtsp_specs.push_back(camera.view + "=" + base + "/" + camera.relay);
                    managed_calibrations.emplace(camera.view, camera.calibration);
                    managed_has_screen.emplace(camera.view, camera.has_screen);
                }
            }
            std::set<std::string> names;
            for (const auto& value : args.rtsp_specs) {
                auto spec = jiankong::parse_stream_spec(value);
                if (!names.insert(spec.name).second) throw std::runtime_error("duplicate RTSP stream name " + spec.name);
                live_specs.push_back(std::move(spec));
            }
            if (live_specs.empty() || static_cast<int>(live_specs.size()) > args.pose_batch) {
                throw std::runtime_error("RTSP stream count must be between 1 and pose batch " + std::to_string(args.pose_batch));
            }
            for (const auto& spec : live_specs) videos.push_back(spec.name + "=" + spec.url);
        } else {
            videos = find_videos_by_rank_from_end(args.root, args.pick_from_end);
            if (videos.empty()) throw std::runtime_error("no videos under " + args.root);
            if (static_cast<int>(videos.size()) > args.pose_batch) videos.resize(args.pose_batch);
        }
        if (!args.output_dir.empty()) fs::create_directories(args.output_dir);

        std::ofstream frame_events_jsonl;
        std::ofstream frame_events_csv;
        if (!args.output_dir.empty()) {
            frame_events_jsonl.open((fs::path(args.output_dir) / "frame_events.jsonl").string());
            frame_events_csv.open((fs::path(args.output_dir) / "frame_events.csv").string());
            if (!frame_events_jsonl || !frame_events_csv) {
                throw std::runtime_error("failed to open frame event metadata under " + args.output_dir);
            }
            frame_events_csv << "stream_index,frame_index,time_sec,accepted_count,alarm_track_count,max_risk,"
                             << "max_window_hits,person_count,phone_count,output_video,input_video\n";
        }

        std::vector<StreamState> streams;
        for (size_t stream_index = 0; stream_index < videos.size(); ++stream_index) {
            const auto& path = videos[stream_index];
            StreamState s;
            s.path = path;
            if (live_mode) {
                s.view = live_specs[stream_index].name;
                s.width = args.source_width;
                s.height = args.source_height;
                s.native_fps = args.infer_fps;
                s.total_frames = 0;
            } else {
                s.cap.open(path);
                if (!s.cap.isOpened()) throw std::runtime_error("failed to open " + path);
                s.width = static_cast<int>(s.cap.get(cv::CAP_PROP_FRAME_WIDTH));
                s.height = static_cast<int>(s.cap.get(cv::CAP_PROP_FRAME_HEIGHT));
                s.native_fps = s.cap.get(cv::CAP_PROP_FPS);
                if (s.native_fps <= 0) s.native_fps = 25.0;
                s.total_frames = static_cast<int>(s.cap.get(cv::CAP_PROP_FRAME_COUNT));
                make_sample_indices(s, args.infer_fps, args.max_samples);
            }
            std::string calib_name;
            if (live_mode && managed_camera_manifest) {
                const auto it = managed_calibrations.find(s.view);
                if (it == managed_calibrations.end()) {
                    throw std::runtime_error("camera manifest calibration missing for view " + s.view);
                }
                calib_name = it->second;
            } else {
                calib_name = calibration_name_for_path(live_mode ? s.view : path);
            }
            fs::path calib_path = calib_name.empty() ? fs::path() : fs::path(args.calib_dir) / calib_name;
            const bool managed_has_no_screen = managed_camera_manifest &&
                managed_has_screen.find(s.view) != managed_has_screen.end() &&
                !managed_has_screen.at(s.view);
            if (live_mode && ((!managed_camera_manifest &&
                               calib_name == "camera_01_screen_calibration_v21.json") ||
                              managed_has_no_screen)) {
                validate_explicit_zero_screen_calibration(s.view, calib_path);
                s.screens = load_calibration(calib_path.string(), s.width, s.height, true);
                s.screen_active = false;
                std::cout << "[CALIB_SELECTED] view=" << s.view
                          << " path=" << calib_path.string()
                          << " screen_active=0 screen_count=0" << std::endl;
            } else {
                s.screens = load_calibration(calib_path.string(), s.width, s.height, live_mode);
                if (live_mode) {
                    validate_live_calibration(s.view, calib_name, calib_path, s.screens);
                    s.screen_active = true;
                    std::cout << "[CALIB_SELECTED] view=" << s.view
                              << " path=" << calib_path.string()
                              << " screen_active=1 screen_count=" << s.screens.size() << std::endl;
                } else {
                    s.screen_active = !s.screens.empty();
                }
            }
            apply_static_args_to_screens(s.screens, args);
            if (live_mode && args.spatial_static_phone_suppression_enabled &&
                args.static_hotspot_enabled) {
                const std::string camera_name = safe_stem(s.view);
                s.static_hotspot_path = fs::path(args.output_dir) /
                    "static_hotspots" /
                    jiankong::custom_pipeline::static_hotspot_state_filename(
                        camera_name, stream_index);
                s.static_hotspots = std::make_unique<
                    jiankong::custom_pipeline::CameraStaticHotspotMap>(s.width, s.height);
                load_static_hotspots(s, unix_now_sec());
            }
            if (!args.output_dir.empty() && !args.no_video) {
                fs::path out_path = fs::path(args.output_dir) / (std::to_string(streams.size()) + "_" + safe_stem(path) + "_boxed.mp4");
                s.output_path = out_path.string();
                s.writer.open(out_path.string(), cv::VideoWriter::fourcc('m', 'p', '4', 'v'), args.infer_fps, cv::Size(s.width, s.height));
                if (!s.writer.isOpened()) throw std::runtime_error("failed to open output video: " + out_path.string());
                std::cout << "[OUTPUT] idx=" << streams.size() << " path=" << out_path.string() << std::endl;
            }
            std::cout << "[STREAM] idx=" << streams.size()
                      << " live=" << (live_mode ? 1 : 0)
                      << " samples=" << (live_mode ? -1 : static_cast<long long>(s.sample_indices.size()))
                      << " target_fps=" << args.infer_fps
                      << " size=" << s.width << "x" << s.height
                      << " path=" << s.path << std::endl;
            std::cout << "[CALIB] idx=" << streams.size()
                      << " file=" << (calib_name.empty() ? "NONE" : calib_path.string())
                      << " screen_active=" << (s.screen_active ? 1 : 0)
                      << " screen_count=" << s.screens.size() << std::endl;
            streams.push_back(std::move(s));
        }        if (!args.output_dir.empty()) {
            json stream_meta = json::array();
            for (size_t i = 0; i < streams.size(); ++i) {
                const auto& s = streams[i];
                json item;
                item["stream_index"] = i;
                item["input_video"] = s.path;
                item["output_video"] = s.output_path;
                item["width"] = s.width;
                item["height"] = s.height;
                item["native_fps"] = s.native_fps;
                item["infer_fps"] = args.infer_fps;
                item["sample_count"] = s.sample_indices.size();
                item["screen_active"] = s.screen_active;
                item["screen_count"] = s.screens.size();
                stream_meta.push_back(std::move(item));
            }
            std::ofstream meta_out((fs::path(args.output_dir) / "streams.json").string());
            meta_out << std::setw(2) << stream_meta << std::endl;
        }

        double t_read = 0, t_pose_prep = 0, t_pose_infer = 0, t_pose_post = 0;
        double t_roi_prep = 0, t_phone_infer = 0, t_phone_post = 0, t_draw_write = 0;
        long long total_frames = 0, total_persons = 0, total_persons_raw = 0, total_persons_deduped = 0;
        long long total_rois = 0, total_phones_raw = 0, total_phones_nms = 0;
        long long total_accepted_candidates = 0, total_alarm_frames = 0;
        long long total_static_phone_suppressed_frames = 0, total_static_phone_suppressed_candidates = 0;
        long long total_desk_zone_phone_frames = 0;

        std::vector<cv::Mat> frames(args.pose_batch);
        std::vector<int> active_streams;
        std::vector<LetterboxMeta> pose_metas(args.pose_batch);
        std::unique_ptr<GpuPreprocessor> gpu_preprocessor;
        if (args.gpu_preprocess && !live_mode) {
            gpu_preprocessor = std::make_unique<GpuPreprocessor>(streams.size());
        }
        std::vector<std::unique_ptr<FrameQueue>> queues;
        std::vector<std::thread> producers;
        std::vector<double> read_thread_times(streams.size(), 0.0);
        std::vector<double> frame_captured_at(args.pose_batch, 0.0);
        std::vector<double> frame_captured_at_unix_seconds(args.pose_batch, 0.0);
        std::vector<jiankong::custom_pipeline::DeviceFrameView> device_frames(args.pose_batch);
        std::vector<LiveWallClockAnchor> live_wall_clock_anchors(streams.size());
        std::vector<std::uint64_t> live_processed(streams.size(), 0);
        std::vector<jiankong::BoundedSampleWindow> live_latencies(streams.size());
        std::unique_ptr<jiankong::custom_pipeline::DeepStreamBatchReader> deepstream_reader;
        std::vector<jiankong::custom_pipeline::SourceReaderMetrics> live_reader_final;
        std::set<unsigned int> live_ready_sources;
        std::ofstream live_stats_jsonl;
        if (live_mode) {
            live_stats_jsonl.open((fs::path(args.output_dir) / "live_stats.jsonl").string());
            if (!live_stats_jsonl) throw std::runtime_error("failed to open live_stats.jsonl");
        }
        std::unique_ptr<jiankong::BoundedOutputWriter> live_output;
        if (live_mode) {
            // Initial opens/header precede the worker; all subsequent live file
            // operations belong to this one consumer. No StreamState escapes.
            live_output = std::make_unique<jiankong::BoundedOutputWriter>(16 * 1024 * 1024,
                [&](int channel, const std::string& bytes) {
                    if (channel == 0) frame_events_jsonl << bytes;
                    else if (channel == 1) frame_events_csv << bytes;
                    else {
                        // Publish progress only after its preceding evidence.
                        frame_events_jsonl.flush();
                        frame_events_csv.flush();
                        if (!frame_events_jsonl || !frame_events_csv)
                            throw std::runtime_error("live metadata flush failed");
                        live_stats_jsonl << bytes;
                        live_stats_jsonl.flush();
                    }
                    if (!frame_events_jsonl || !frame_events_csv || !live_stats_jsonl)
                        throw std::runtime_error("live output write failed");
                });
            std::signal(SIGTERM, request_output_stop);
            std::signal(SIGINT, request_output_stop);
        }

        const double wall0 = now_sec();
        const double live_startup_deadline = wall0 + std::max(0.001, args.open_timeout_ms / 1000.0);
        double live_measurement_started_at = live_mode ? 0.0 : wall0;
        double live_measurement_stop_at = live_mode ? 0.0 : wall0;
        double live_measurement_stopped_at = live_mode ? 0.0 : wall0;
        double live_startup_seconds = 0.0;
        double live_shutdown_seconds = 0.0;
        if (live_mode) {
            jiankong::custom_pipeline::BatchReaderConfig reader_config;
            reader_config.target_fps = static_cast<unsigned int>(std::round(args.infer_fps));
            reader_config.width = static_cast<unsigned int>(args.source_width);
            reader_config.height = static_cast<unsigned int>(args.source_height);
            reader_config.batched_push_timeout_us =
                static_cast<unsigned int>(std::round(1000000.0 / args.infer_fps));
            reader_config.reconnect_interval_seconds =
                static_cast<unsigned int>(std::max(1, args.reconnect_delay_ms / 1000));
            for (size_t si = 0; si < live_specs.size(); ++si) {
                reader_config.sources.push_back(
                    {static_cast<unsigned int>(si), live_specs[si].url});
            }
            deepstream_reader = std::make_unique<jiankong::custom_pipeline::DeepStreamBatchReader>(
                std::move(reader_config));
            deepstream_reader->start();
        } else if (args.pipelined_read) {
            queues.reserve(streams.size());
            for (size_t si = 0; si < streams.size(); ++si) {
                queues.emplace_back(std::make_unique<FrameQueue>());
            }
            for (int si = 0; si < static_cast<int>(streams.size()); ++si) {
                producers.emplace_back([&, si]() {
                    double local_read = 0.0;
                    while (true) {
                        cv::Mat frame;
                        const double rt0 = now_sec();
                        bool ok = read_next_sample(streams[si], frame);
                        local_read += now_sec() - rt0;
                        if (!ok) break;
                        FrameQueue& q = *queues[si];
                        std::unique_lock<std::mutex> lock(q.mutex);
                        q.not_full.wait(lock, [&]() {
                            return static_cast<int>(q.frames.size()) < std::max(1, args.queue_size);
                        });
                        q.frames.push_back(std::move(frame));
                        lock.unlock();
                        q.not_empty.notify_one();
                    }
                    read_thread_times[si] = local_read;
                    FrameQueue& q = *queues[si];
                    {
                        std::lock_guard<std::mutex> lock(q.mutex);
                        q.done = true;
                    }
                    q.not_empty.notify_all();
                });
            }
        }

        double last_live_stats = wall0;
        int step = 0;
        while (true) {
            if (live_mode) live_output->throw_if_failed();
            if (live_mode && output_stop_requested) break;
            if (live_mode && live_measurement_started_at > 0.0 &&
                now_sec() >= live_measurement_stop_at) {
                live_measurement_stopped_at = live_measurement_stop_at;
                live_reader_final = deepstream_reader->metrics();
                break;
            }
            active_streams.clear();
            jiankong::custom_pipeline::DeviceBatch device_batch;
            double t0 = now_sec();
            if (live_mode) {
                device_batch = deepstream_reader->pull(100);
                std::set<unsigned int> batch_source_ids;
                if (device_batch) {
                    for (const auto& view : device_batch.frames()) {
                        if (view.source_id >= streams.size()) {
                            throw std::runtime_error("DeepStream batch has unknown source id " +
                                                     std::to_string(view.source_id));
                        }
                        if (view.width != streams[view.source_id].width ||
                            view.height != streams[view.source_id].height) {
                            throw std::runtime_error("DeepStream RGBA frame changed calibration dimensions");
                        }
                        const int batch_index = static_cast<int>(active_streams.size());
                        if (batch_index >= args.pose_batch) break;
                        active_streams.push_back(static_cast<int>(view.source_id));
                        device_frames[batch_index] = view;
                        frame_captured_at[batch_index] = view.received_at_seconds;
                        frame_captured_at_unix_seconds[batch_index] = live_frame_capture_unix_seconds(
                            live_wall_clock_anchors[view.source_id], view);
                        live_ready_sources.insert(view.source_id);
                        batch_source_ids.insert(view.source_id);
                    }
                }
                if (deepstream_reader->terminal()) {
                    throw std::runtime_error("DeepStream pipeline terminated before the live measurement completed");
                }
                if (live_measurement_started_at == 0.0) {
                    const bool live_complete_batch =
                        active_streams.size() == streams.size() && batch_source_ids.size() == streams.size();
                    if (live_complete_batch) {
                        live_measurement_started_at =
                            deepstream_reader->begin_measurement(static_cast<double>(args.duration_sec));
                        live_measurement_stop_at = live_measurement_started_at + args.duration_sec;
                        live_startup_seconds = live_measurement_started_at - wall0;
                        last_live_stats = live_measurement_started_at;
                        std::cout << "[LIVE_MEASUREMENT_START] startup_sec=" << live_startup_seconds
                                  << " duration_sec=" << args.duration_sec
                                  << " ready_sources=" << live_ready_sources.size() << std::endl;
                        continue;
                    }
                    if (now_sec() >= live_startup_deadline) {
                        throw std::runtime_error(
                            "live startup timeout before all configured sources produced a frame: ready=" +
                            std::to_string(live_ready_sources.size()));
                    }
                    continue;
                }
                if (now_sec() >= live_measurement_stop_at) {
                    live_measurement_stopped_at = live_measurement_stop_at;
                    live_reader_final = deepstream_reader->metrics();
                    break;
                }
                if (active_streams.empty()) {
                    continue;
                }
            } else if (args.pipelined_read) {
                for (int si = 0; si < static_cast<int>(streams.size()); ++si) {
                    FrameQueue& q = *queues[si];
                    std::unique_lock<std::mutex> lock(q.mutex);
                    q.not_empty.wait(lock, [&]() { return !q.frames.empty() || q.done; });
                    if (!q.frames.empty()) {
                        active_streams.push_back(si);
                        frames[active_streams.size() - 1] = std::move(q.frames.front());
                        q.frames.pop_front();
                        lock.unlock();
                        q.not_full.notify_one();
                    }
                }
            } else {
                for (int si = 0; si < static_cast<int>(streams.size()); ++si) {
                    cv::Mat frame;
                    if (read_next_sample(streams[si], frame)) {
                        active_streams.push_back(si);
                        frames[active_streams.size() - 1] = frame;
                    }
                }
            }            t_read += now_sec() - t0;
            if (active_streams.empty()) break;

            t0 = now_sec();
            const int pose_preprocess_batch = pose.preprocessing_batch_size(static_cast<int>(active_streams.size()));
            if (live_mode) {
                for (int b = 0; b < pose_preprocess_batch; ++b) {
                    const int src_idx = b < static_cast<int>(active_streams.size()) ? b : 0;
                    const auto& view = device_frames[src_idx];
                    pose_metas[b] = preprocess_rgba_device_frame(
                        view,
                        0,
                        0,
                        view.width,
                        view.height,
                        args.pose_size,
                        static_cast<float*>(pose.device_input()),
                        b,
                        pose.stream());
                }
            } else if (args.gpu_preprocess) {
                for (int b = 0; b < static_cast<int>(active_streams.size()); ++b) {
                    const int si = active_streams[b];
                    gpu_preprocessor->upload_frame(si, frames[b], pose.stream());
                }
                for (int b = 0; b < pose_preprocess_batch; ++b) {
                    const int src_idx = b < static_cast<int>(active_streams.size()) ? b : 0;
                    const int si = active_streams[src_idx];
                    pose_metas[b] = gpu_preprocessor->preprocess(
                        si,
                        streams[si].width,
                        streams[si].height,
                        0,
                        0,
                        streams[si].width,
                        streams[si].height,
                        args.pose_size,
                        static_cast<float*>(pose.device_input()),
                        b,
                        pose.stream());
                }
                check_cuda(cudaStreamSynchronize(pose.stream()), "pose preprocess sync");
            } else {
                for (int b = 0; b < pose_preprocess_batch; ++b) {
                    const int src_idx = b < static_cast<int>(active_streams.size()) ? b : 0;
                    pose_metas[b] = preprocess_into(frames[src_idx], args.pose_size, pose.input(), b, args.pose_batch);
                }
            }
            t_pose_prep += now_sec() - t0;

            t0 = now_sec();
            pose.infer_compacted(live_mode || args.gpu_preprocess,
                                 static_cast<int>(active_streams.size()),
                                 pose_channels,
                                 pose_boxes,
                                 4,
                                 args.pose_conf);
            t_pose_infer += now_sec() - t0;

            std::vector<RoiJob> roi_jobs;
            std::vector<std::vector<Det>> people_by_stream(streams.size());
            std::vector<std::vector<Rect>> rois_by_stream(streams.size());
            t0 = now_sec();
            for (int b = 0; b < static_cast<int>(active_streams.size()); ++b) {
                const int si = active_streams[b];
                streams[si].frame_id += 1;
                auto people_raw = decode_pose(pose, b, pose_channels, pose_metas[b],
                                              streams[si].width, streams[si].height, args.kp_conf);
                const long long raw_count = static_cast<long long>(people_raw.size());
                auto people = dedupe_people(std::move(people_raw), args.kp_conf);
                const int window_size = stream_state_window(streams[si]);
                 streams[si].next_track_id = assign_person_track_ids(people, streams[si].person_states,
                                                                     streams[si].next_track_id,
                                                                     streams[si].frame_id,
                                                                     window_size,
                                                                     spatial_static_config,
                                                                     args.spatial_static_phone_suppression_enabled);
                const long long deduped = std::max<long long>(0, raw_count - static_cast<long long>(people.size()));
                streams[si].frames += 1;
                streams[si].persons += static_cast<long long>(people.size());
                total_frames += 1;
                total_persons_raw += raw_count;
                total_persons_deduped += deduped;
                total_persons += static_cast<long long>(people.size());
                people_by_stream[si] = people;
                if (!jiankong::custom_pipeline::should_build_phone_roi(streams[si].screen_active)) {
                    continue;
                }
                for (const auto& p : people) {
                    RoiJob job;
                    job.stream = si;
                    job.person_index = p.person_index;
                    job.source = "person_roi";
                    job.roi = expand_person_roi(p.box, streams[si].width, streams[si].height, args.person_expand_x, args.person_expand_y);
                    if (job.roi.w() >= 2 && job.roi.h() >= 2) {
                        roi_jobs.push_back(job);
                        rois_by_stream[si].push_back(job.roi);
                        streams[si].rois += 1;
                        total_rois += 1;
                    }
                }
            }
            t_pose_post += now_sec() - t0;

            std::vector<Det> all_phones;
            for (int start = 0; start < static_cast<int>(roi_jobs.size()); start += args.phone_batch) {
                const int real = std::min(args.phone_batch, static_cast<int>(roi_jobs.size()) - start);
                std::vector<LetterboxMeta> metas(args.phone_batch);
                const int phone_preprocess_batch = phone.preprocessing_batch_size(real);
                t0 = now_sec();
                if (live_mode) {
                    for (int b = 0; b < phone_preprocess_batch; ++b) {
                        const int job_idx = start + (b < real ? b : 0);
                        const RoiJob& job = roi_jobs[job_idx];
                        const auto active_it = std::find(active_streams.begin(), active_streams.end(), job.stream);
                        if (active_it == active_streams.end()) {
                            throw std::runtime_error("phone ROI references a source absent from current batch");
                        }
                        const int view_index = static_cast<int>(active_it - active_streams.begin());
                        const auto& view = device_frames[view_index];
                        const int x = static_cast<int>(std::round(job.roi.x1));
                        const int y = static_cast<int>(std::round(job.roi.y1));
                        const int w = std::max(1, static_cast<int>(std::round(job.roi.w())));
                        const int h = std::max(1, static_cast<int>(std::round(job.roi.h())));
                        metas[b] = preprocess_rgba_device_frame(
                            view,
                            x,
                            y,
                            w,
                            h,
                            args.phone_size,
                            static_cast<float*>(phone.device_input()),
                            b,
                            phone.stream());
                    }
                } else if (args.gpu_preprocess) {
                    for (int b = 0; b < phone_preprocess_batch; ++b) {
                        const int job_idx = start + (b < real ? b : 0);
                        const RoiJob& job = roi_jobs[job_idx];
                        const int si = job.stream;
                        const int x = static_cast<int>(std::round(job.roi.x1));
                        const int y = static_cast<int>(std::round(job.roi.y1));
                        const int w = std::max(1, static_cast<int>(std::round(job.roi.w())));
                        const int h = std::max(1, static_cast<int>(std::round(job.roi.h())));
                        metas[b] = gpu_preprocessor->preprocess(
                            si,
                            streams[si].width,
                            streams[si].height,
                            x,
                            y,
                            w,
                            h,
                            args.phone_size,
                            static_cast<float*>(phone.device_input()),
                            b,
                            phone.stream());
                    }
                    check_cuda(cudaStreamSynchronize(phone.stream()), "phone preprocess sync");
                } else {
                    for (int b = 0; b < phone_preprocess_batch; ++b) {
                        const int job_idx = start + (b < real ? b : 0);
                        const RoiJob& job = roi_jobs[job_idx];
                        const cv::Mat& src_frame = frames[std::find(active_streams.begin(), active_streams.end(), job.stream) - active_streams.begin()];
                        cv::Rect rr(static_cast<int>(std::round(job.roi.x1)), static_cast<int>(std::round(job.roi.y1)),
                                    std::max(1, static_cast<int>(std::round(job.roi.w()))),
                                    std::max(1, static_cast<int>(std::round(job.roi.h()))));
                        rr &= cv::Rect(0, 0, src_frame.cols, src_frame.rows);
                        cv::Mat crop = src_frame(rr);
                        metas[b] = preprocess_into(crop, args.phone_size, phone.input(), b, args.phone_batch);
                    }
                }
                t_roi_prep += now_sec() - t0;

                t0 = now_sec();
                phone.infer_compacted(live_mode || args.gpu_preprocess,
                                      real,
                                      phone_channels,
                                      phone_boxes,
                                      4,
                                      args.phone_conf);
                t_phone_infer += now_sec() - t0;

                t0 = now_sec();
                auto phones = decode_phone_batch(phone, real, phone_channels, metas, roi_jobs, start, streams);
                total_phones_raw += static_cast<long long>(phones.size());
                for (const auto& ph : phones) streams[ph.stream].phones_raw += 1;
                all_phones.insert(all_phones.end(), phones.begin(), phones.end());
                t_phone_post += now_sec() - t0;
            }

            t0 = now_sec();
            std::vector<std::vector<Det>> phones_keep_by_stream(streams.size());
            for (int si : active_streams) {
                std::vector<Det> per;
                for (const auto& ph : all_phones) {
                    if (ph.stream == si) per.push_back(ph);
                }
                auto keep = nms(std::move(per), 0.50f, 100);
                streams[si].phones_nms += static_cast<long long>(keep.size());
                total_phones_nms += static_cast<long long>(keep.size());
                phones_keep_by_stream[si] = std::move(keep);
            }
            t_phone_post += now_sec() - t0;

            std::vector<std::vector<CandidateEval>> evals_by_stream(streams.size());
            if (live_mode && !args.fixed_template_dir.empty()) {
                fixed_template_registry.maybe_reload(now_sec());
            }
            for (int si : active_streams) {
                jiankong::custom_pipeline::CameraStaticHotspotMap* static_hotspots =
                    live_mode && args.spatial_static_phone_suppression_enabled &&
                        args.static_hotspot_enabled
                    ? streams[si].static_hotspots.get() : nullptr;
                double static_hotspot_now = 0.0;
                if (live_mode) {
                    const auto active = std::find(active_streams.begin(), active_streams.end(), si);
                    static_hotspot_now = frame_captured_at_unix_seconds[
                        static_cast<std::size_t>(active - active_streams.begin())];
                }
                if (!streams[si].screen_active) {
                    const int window_size = stream_state_window(streams[si]);
                    const int min_hits = stream_state_min_hits(streams[si], window_size);
                    std::vector<CandidateEval> no_evals;
                    update_person_states(people_by_stream[si], no_evals, streams[si].person_states,
                                          streams[si].frame_id, window_size,
                                          stream_state_risk_threshold(streams[si]), min_hits,
                                          stream_handheld_suspect_min_hits(streams[si], min_hits), args.infer_fps,
                                          static_hotspots, static_hotspot_now,
                                          &streams[si].static_hotspot_dirty,
                                          args.spatial_static_phone_suppression_enabled);
                    jiankong::custom_pipeline::reset_inactive_stream_state(
                        streams[si].alert_counter,
                        streams[si].previous_alarm_tracks,
                        streams[si].person_states,
                        streams[si].frame_id,
                        window_size);
                    continue;
                }
                for (const auto& ph : phones_keep_by_stream[si]) {
                    std::vector<CandidateEval> phone_evals;
                    phone_evals.reserve(streams[si].screens.size());
                    for (const auto& screen_cfg : streams[si].screens) {
                        phone_evals.push_back(evaluate_phone(ph, screen_cfg, people_by_stream[si],
                                                             streams[si].person_states,
                                                             streams[si].width, streams[si].height,
                                                             args.kp_conf));
                    }
                    evals_by_stream[si].push_back(choose_best_eval(phone_evals));
                }
                if (live_mode && !args.fixed_template_dir.empty()) {
                    const auto active =
                        std::find(active_streams.begin(), active_streams.end(), si);
                    if (active != active_streams.end()) {
                        const auto& frame = device_frames[
                            static_cast<std::size_t>(active - active_streams.begin())];
                        std::ostringstream camera_name;
                        camera_name << "camera" << std::setfill('0') << std::setw(2)
                                    << (si + 1);
                        for (auto& ev : evals_by_stream[si]) {
                            const FixedTemplateMatchResult match =
                                fixed_template_registry.match(
                                    camera_name.str(), ev.phone.box, frame);
                            ev.fixed_template_near =
                                match.evidence !=
                                jiankong::custom_pipeline::FixedTemplateEvidence::none;
                            ev.fixed_template_match =
                                match.evidence ==
                                jiankong::custom_pipeline::FixedTemplateEvidence::matched;
                            ev.fixed_template_score = match.appearance_score;
                            ev.fixed_template_id = match.template_id;
                        }
                    }
                }
                const int window_size = stream_state_window(streams[si]);
                const int min_hits = stream_state_min_hits(streams[si], window_size);
                const float risk_thr = stream_state_risk_threshold(streams[si]);
                update_person_states(people_by_stream[si], evals_by_stream[si], streams[si].person_states,
                                     streams[si].frame_id, window_size, risk_thr, min_hits,
                                     stream_handheld_suspect_min_hits(streams[si], min_hits), args.infer_fps,
                                     static_hotspots, static_hotspot_now,
                                     &streams[si].static_hotspot_dirty,
                                     args.spatial_static_phone_suppression_enabled);

                std::set<int> visible_track_ids;
                for (const auto& person : people_by_stream[si]) {
                    if (person.track_id >= 0) visible_track_ids.insert(person.track_id);
                }
                std::set<int> active_alarm_tracks;
                int max_hits = 0;
                for (int tid : visible_track_ids) {
                    auto st = streams[si].person_states.find(tid);
                    if (st == streams[si].person_states.end()) continue;
                    max_hits = std::max(max_hits, st->second.window_hits);
                    if (st->second.alarm_triggered) active_alarm_tracks.insert(tid);
                }
                streams[si].alert_counter = static_cast<float>(max_hits);
                streams[si].alarm_frames += active_alarm_tracks.empty() ? 0 : 1;
                total_alarm_frames += active_alarm_tracks.empty() ? 0 : 1;
                bool has_static_suppressed = false;
                bool has_desk_zone_phone = false;
                for (auto& ev : evals_by_stream[si]) {
                    auto st = streams[si].person_states.find(ev.track_id);
                    if (ev.static_suppressed) {
                        has_static_suppressed = true;
                        streams[si].static_phone_suppressed_candidates += 1;
                        total_static_phone_suppressed_candidates += 1;
                    }
                    if (ev.phone_in_desk_zone) has_desk_zone_phone = true;
                    if (st == streams[si].person_states.end()) continue;
                    ev.person_alarm = st->second.alarm_triggered;
                    ev.person_window_hits = st->second.window_hits;
                    if (ev.accepted()) {
                        streams[si].accepted_candidates += 1;
                        total_accepted_candidates += 1;
                    }
                }
                streams[si].static_phone_suppressed_frames += has_static_suppressed ? 1 : 0;
                streams[si].desk_zone_phone_frames += has_desk_zone_phone ? 1 : 0;
                total_static_phone_suppressed_frames += has_static_suppressed ? 1 : 0;
                total_desk_zone_phone_frames += has_desk_zone_phone ? 1 : 0;
                streams[si].previous_alarm_tracks = std::move(active_alarm_tracks);
                prune_person_tracks(streams[si].person_states, streams[si].frame_id, window_size);
            }
            if (live_mode && args.spatial_static_phone_suppression_enabled &&
                args.static_hotspot_enabled) {
                const double persist_now = unix_now_sec();
                for (auto& stream : streams) {
                    persist_static_hotspots(stream, persist_now, false);
                }
            }

            if (!args.output_dir.empty()) {
                t0 = now_sec();
                for (int b = 0; b < static_cast<int>(active_streams.size()); ++b) {
                    const int si = active_streams[b];
                    if (!args.no_video) {
                        cv::Mat& annotated = frames[b];
                        for (size_t screen_idx = 0; screen_idx < streams[si].screens.size(); ++screen_idx) {
                            std::string label = "SCREEN" + std::to_string(screen_idx + 1);
                            draw_poly(annotated,
                                      streams[si].screens[screen_idx].screen_poly,
                                      cv::Scalar(255, 80, 0),
                                      2,
                                      label,
                                      0.32);
                        }
                        for (const auto& person : people_by_stream[si]) {
                            const CandidateEval* ev = best_eval_for_person(evals_by_stream[si], person.person_index);
                            const PersonTrackState* st = nullptr;
                            auto st_it = streams[si].person_states.find(person.track_id);
                            if (st_it != streams[si].person_states.end()) st = &st_it->second;
                            const bool alarm = st != nullptr && st->alarm_triggered;
                            const bool risk = ev != nullptr && ev->accepted();
                            const bool suspect = st != nullptr && is_suspect_track(*st, streams[si].frame_id);
                            if (!alarm && !risk && !suspect) continue;
                            const Rect draw_box = st != nullptr ? display_bbox_for_track(*st, person) : person.box;
                            const Rect roi = expand_person_roi(draw_box, streams[si].width, streams[si].height,
                                                               args.person_expand_x, args.person_expand_y);
                            cv::Scalar color = cv::Scalar(0, 220, 220);
                            int thickness = 2;
                            std::ostringstream label;
                            label << "Person" << (person.track_id > 0 ? person.track_id : person.person_index + 1);
                            if (alarm) {
                                color = cv::Scalar(0, 0, 255);
                                thickness = 4;
                                label << " ALARM";
                            } else if (risk) {
                                color = cv::Scalar(0, 165, 255);
                                thickness = 3;
                                label << " RISK " << std::fixed << std::setprecision(2) << ev->risk_score;
                            } else if (suspect) {
                                color = cv::Scalar(0, 165, 255);
                                thickness = 2;
                                label << " TRACK";
                            }
                            if (st != nullptr && st->window_hits > 0 && !alarm) {
                                label << " H" << st->window_hits;
                            }
                            if (st != nullptr && st->handheld_phone_hits > 0 && !alarm && !risk) {
                                label << " P" << st->handheld_phone_hits;
                            }
                            draw_rect(annotated, roi, color, thickness, label.str(), 0.38, true);
                        }
                        for (const auto& ph : phones_keep_by_stream[si]) {
                            const CandidateEval* ev = best_eval_for_phone(evals_by_stream[si], ph);
                            const bool accepted_phone = ev != nullptr && ev->accepted();
                            const bool handheld_phone = ev != nullptr && is_handheld_phone_suspect_candidate(*ev);
                            const bool static_phone = ev != nullptr && ev->static_suppressed;
                            cv::Scalar phone_color = cv::Scalar(120, 120, 120);
                            int phone_thickness = 1;
                            std::ostringstream label;
                            if (static_phone) {
                                phone_color = cv::Scalar(150, 150, 150);
                                phone_thickness = 2;
                                label << "STATIC PHONE";
                            } else if (accepted_phone) {
                                phone_color = cv::Scalar(0, 0, 255);
                                phone_thickness = 3;
                                label << "PHONE " << std::fixed << std::setprecision(2) << ph.conf
                                      << " R" << ev->risk_score;
                            } else if (handheld_phone) {
                                phone_color = cv::Scalar(0, 165, 255);
                                phone_thickness = 2;
                                label << "PHONE " << std::fixed << std::setprecision(2) << ph.conf
                                      << " S" << ev->risk_score;
                            } else {
                                continue;
                            }
                            draw_rect(annotated, ph.box, phone_color, phone_thickness, label.str(), 0.40, false);
                        }
                    }
                    if (!live_mode) {
                        write_event_metadata(frame_events_jsonl,
                                             frame_events_csv,
                                             si,
                                             streams[si],
                                             args.infer_fps,
                                             0.0,
                                             args.person_expand_x,
                                             args.person_expand_y,
                                             people_by_stream[si],
                                             phones_keep_by_stream[si],
                                             evals_by_stream[si]);
                    }
                    if (!args.no_video) {
                        streams[si].writer.write(frames[b]);
                    }
                }
                t_draw_write += now_sec() - t0;
            }

            if (live_mode) {
                const double completed_at = now_sec();
                const bool formal_completion = jiankong::custom_pipeline::completion_is_within_measurement(
                    completed_at, live_measurement_stop_at);
                if (!formal_completion) {
                    live_measurement_stopped_at = live_measurement_stop_at;
                    live_reader_final = deepstream_reader->metrics();
                    std::cout << "[LIVE_CUTOFF] rejected_batch_completed_at=" << completed_at
                              << " deadline=" << live_measurement_stop_at << std::endl;
                    break;
                }
                for (int b = 0; b < static_cast<int>(active_streams.size()); ++b) {
                    const int si = active_streams[b];
                    live_processed[si] += 1;
                    live_latencies[si].push_back((completed_at - frame_captured_at[b]) * 1000.0);
                    std::ostringstream json_records, csv_records;
                    write_event_metadata(json_records,
                                         csv_records,
                                         si,
                                         streams[si],
                                         args.infer_fps,
                                         frame_captured_at_unix_seconds[b],
                                         args.person_expand_x,
                                         args.person_expand_y,
                                         people_by_stream[si],
                                         phones_keep_by_stream[si],
                                         evals_by_stream[si]);
                    if (!json_records.str().empty()) {
                        live_output->submit(0, json_records.str());
                        live_output->submit(1, csv_records.str());
                    }
                }
                if (completed_at - last_live_stats >= 1.0) {
                    const auto reader_metrics = deepstream_reader->metrics();
                    const double measurement_elapsed =
                        jiankong::custom_pipeline::effective_measurement_seconds(
                            live_measurement_started_at, completed_at,
                            static_cast<double>(args.duration_sec));
                    json snapshot;
                    snapshot["elapsed_sec"] = measurement_elapsed;
                    snapshot["startup_sec"] = live_startup_seconds;
                    // Keep per-stage live timings in the periodic sidecar so
                    // concurrency changes are driven by the real bottleneck,
                    // not by an assumed GPU bottleneck.  These are averages
                    // per global inference tick (all active camera views).
                    const double completed_steps = std::max(1, step + 1);
                    snapshot["timing_ms_per_batch"] = {
                        {"pose_preprocess", 1000.0 * t_pose_prep / completed_steps},
                        {"pose_infer", 1000.0 * t_pose_infer / completed_steps},
                        {"pose_postprocess", 1000.0 * t_pose_post / completed_steps},
                        {"phone_roi_preprocess", 1000.0 * t_roi_prep / completed_steps},
                        {"phone_infer", 1000.0 * t_phone_infer / completed_steps},
                        {"phone_postprocess", 1000.0 * t_phone_post / completed_steps},
                        {"draw_write", 1000.0 * t_draw_write / completed_steps},
                    };
                    snapshot["streams"] = json::array();
                    for (size_t si = 0; si < streams.size(); ++si) {
                        const auto metric = source_reader_metrics_for(reader_metrics, si);
                        json item;
                        item["stream_index"] = si;
                        item["view"] = live_specs[si].name;
                        item["screen_active"] = streams[si].screen_active;
                        item["screen_count"] = streams[si].screens.size();
                        item["capture_fps"] = metric.admitted / std::max(0.001, measurement_elapsed);
                        item["processed_fps"] = live_processed[si] / std::max(0.001, measurement_elapsed);
                        item["captured"] = metric.admitted;
                        item["processed"] = live_processed[si];
                        item["dropped"] = metric.phase_dropped;
                        item["downstream_frames"] = metric.downstream_frames;
                        item["backlog_loss_upper_bound"] = metric.admitted_minus_pulled_upper_bound;
                        item["pts_gap_lost"] = metric.pts_gap_lost;
                        item["pts_gap_lost_is_estimate"] = true;
                        item["source_errors"] = metric.source_errors;
                        item["reconnect_triggers"] = metric.source_errors;
                        item["reconnects"] = 0;
                        item["reconnects_available"] = false;
                        item["latency_p50_ms"] =
                            jiankong::percentile(live_latencies[si].snapshot(), 0.50);
                        item["latency_p95_ms"] =
                            jiankong::percentile(live_latencies[si].snapshot(), 0.95);
                        snapshot["streams"].push_back(std::move(item));
                    }
                    const auto output_metrics = live_output->metrics();
                    snapshot["output_queue"] = {
                        {"pending_bytes", output_metrics.pending_bytes},
                        {"high_water_bytes", output_metrics.high_water_bytes},
                        {"backpressure_count", output_metrics.backpressure_count},
                        {"capacity_bytes", 16 * 1024 * 1024},
                    };
                    live_output->submit(2, snapshot.dump() + '\n');
                    last_live_stats = completed_at;
                }
            }
            ++step;
            if (step % 25 == 0) {
                const double elapsed = live_mode
                                           ? std::max(0.001, now_sec() - live_measurement_started_at)
                                           : now_sec() - wall0;
                std::cout << "[PROGRESS] step=" << step
                          << " frames=" << total_frames
                          << " fps=" << (elapsed > 0 ? total_frames / elapsed : 0.0)
                          << " persons=" << total_persons
                          << " persons_deduped=" << total_persons_deduped
                          << " rois=" << total_rois
                          << " accepted=" << total_accepted_candidates
                          << " static_suppressed=" << total_static_phone_suppressed_candidates
                          << " alarm_frames=" << total_alarm_frames
                          << std::endl;
            }
        }

        if (live_mode) {
            const double shutdown_started_at = now_sec();
            // Flush final evidence even if shutdown is between stats ticks.
            live_output->submit(2, "");
            live_output->close();
            if (live_reader_final.empty()) live_reader_final = deepstream_reader->metrics();
            deepstream_reader->stop();
            if (args.spatial_static_phone_suppression_enabled && args.static_hotspot_enabled) {
                const double persist_now = unix_now_sec();
                for (auto& stream : streams) {
                    persist_static_hotspots(stream, persist_now, true);
                }
            }
            live_shutdown_seconds = now_sec() - shutdown_started_at;
            if (live_measurement_stopped_at == 0.0) live_measurement_stopped_at = now_sec();
        } else if (args.pipelined_read) {
            for (auto& t : producers) {
                if (t.joinable()) t.join();
            }
            t_read = std::accumulate(read_thread_times.begin(), read_thread_times.end(), 0.0);
        }

        const double wall = now_sec() - wall0;
        const double live_effective_measurement = live_mode
                                                      ? jiankong::custom_pipeline::effective_measurement_seconds(
                                                            live_measurement_started_at,
                                                            live_measurement_stopped_at,
                                                            static_cast<double>(args.duration_sec))
                                                      : wall;
        const std::uint64_t formal_completed_frames = live_mode
                                                          ? std::accumulate(live_processed.begin(),
                                                                            live_processed.end(),
                                                                            std::uint64_t{0})
                                                          : static_cast<std::uint64_t>(total_frames);
        std::cout << "[SUMMARY] frames=" << formal_completed_frames
                  << " inferred_frames=" << total_frames
                  << " wall_sec=" << live_effective_measurement
                  << " aggregate_fps="
                  << (live_effective_measurement > 0
                          ? formal_completed_frames / live_effective_measurement
                          : 0.0)
                  << " persons=" << total_persons
                  << " persons_raw=" << total_persons_raw
                  << " persons_deduped=" << total_persons_deduped
                  << " rois=" << total_rois
                  << " phones_raw=" << total_phones_raw
                  << " phones_nms=" << total_phones_nms
                  << " accepted=" << total_accepted_candidates
                  << " static_suppressed_candidates=" << total_static_phone_suppressed_candidates
                  << " static_suppressed_frames=" << total_static_phone_suppressed_frames
                  << " desk_zone_phone_frames=" << total_desk_zone_phone_frames
                  << " alarm_frames=" << total_alarm_frames
                  << std::endl;
        std::cout << "[TIMING] read=" << t_read
                  << " pose_prep=" << t_pose_prep
                  << " pose_infer=" << t_pose_infer
                  << " pose_post=" << t_pose_post
                  << " roi_prep=" << t_roi_prep
                  << " phone_infer=" << t_phone_infer
                  << " phone_post=" << t_phone_post
                  << " draw_write=" << t_draw_write
                  << std::endl;
        for (size_t i = 0; i < streams.size(); ++i) {
            const auto& s = streams[i];
            std::cout << "[STREAM_SUMMARY] idx=" << i
                      << " frames=" << s.frames
                      << " persons=" << s.persons
                      << " rois=" << s.rois
                      << " phones_raw=" << s.phones_raw
                      << " phones_nms=" << s.phones_nms
                      << " accepted=" << s.accepted_candidates
                      << " static_suppressed_candidates=" << s.static_phone_suppressed_candidates
                      << " static_suppressed_frames=" << s.static_phone_suppressed_frames
                      << " desk_zone_phone_frames=" << s.desk_zone_phone_frames
                      << " alarm_frames=" << s.alarm_frames
                      << " screen_active=" << (s.screen_active ? 1 : 0)
                      << " screens=" << s.screens.size()
                      << " path=" << s.path
                      << std::endl;
        }
        if (live_mode) {
            const auto& reader_metrics = live_reader_final;
            const double planned_measurement_window = static_cast<double>(args.duration_sec);
            const double measurement_window = live_effective_measurement;
            json summary;
            summary["mode"] = "deepstream_custom_rtsp_live";
            summary["duration_sec"] = measurement_window;
            summary["planned_measurement_seconds"] = planned_measurement_window;
            summary["actual_effective_measurement_seconds"] = measurement_window;
            summary["effective_measurement_seconds"] = measurement_window;
            summary["startup_seconds"] = live_startup_seconds;
            summary["shutdown_seconds"] = live_shutdown_seconds;
            summary["total_runtime_seconds"] = wall;
            summary["measurement_started_at_monotonic"] = live_measurement_started_at;
            summary["measurement_stopped_at_monotonic"] = live_measurement_stopped_at;
            summary["target_fps_per_stream"] = args.infer_fps;
            summary["stream_count"] = live_specs.size();
            summary["screen_active_streams"] =
                std::count_if(streams.begin(), streams.end(), [](const auto& stream) {
                    return stream.screen_active;
                });
            summary["screen_inactive_streams"] =
                streams.size() - summary["screen_active_streams"].get<std::size_t>();
            summary["completed_fps"] = formal_completed_frames / std::max(0.001, measurement_window);
            summary["aggregate_target_fps"] = args.infer_fps * live_specs.size();
            summary["target_attainment"] =
                formal_completed_frames /
                std::max(1.0, measurement_window * args.infer_fps * live_specs.size());
            summary["inferred_frames_including_rejected_cutoff_batch"] = total_frames;
            summary["captured_total"] = 0;
            summary["processed_total"] = 0;
            summary["dropped_total"] = 0;
            summary["downstream_frames_total"] = 0;
            summary["backlog_loss_upper_bound_total"] = 0;
            summary["pts_gap_lost_total"] = 0;
            summary["pts_gap_lost_is_estimate"] = true;
            summary["source_errors_total"] = 0;
            summary["reconnect_triggers_total"] = 0;
            summary["reconnects_total"] = 0;
            summary["reconnects_available"] = false;
            summary["streams"] = json::array();
            for (size_t si = 0; si < live_specs.size(); ++si) {
                const auto metric = source_reader_metrics_for(reader_metrics, si);
                const std::uint64_t captured = metric.admitted;
                const std::uint64_t processed = live_processed[si];
                const std::uint64_t dropped = metric.phase_dropped;
                const std::uint64_t reconnects = 0;
                json item;
                item["stream_index"] = si;
                item["view"] = live_specs[si].name;
                item["url"] = live_specs[si].url;
                item["screen_active"] = streams[si].screen_active;
                item["screen_count"] = streams[si].screens.size();
                item["captured"] = captured;
                item["processed"] = processed;
                item["dropped"] = dropped;
                item["downstream_frames"] = metric.downstream_frames;
                item["backlog_loss_upper_bound"] = metric.admitted_minus_pulled_upper_bound;
                item["pts_gap_lost"] = metric.pts_gap_lost;
                item["pts_gap_lost_is_estimate"] = true;
                item["source_errors"] = metric.source_errors;
                item["reconnect_triggers"] = metric.source_errors;
                item["reconnects"] = reconnects;
                item["reconnects_available"] = false;
                item["capture_fps"] = captured / measurement_window;
                item["processed_fps"] = processed / measurement_window;
                item["latency_p50_ms"] =
                    jiankong::percentile(live_latencies[si].snapshot(), 0.50);
                item["latency_p95_ms"] =
                    jiankong::percentile(live_latencies[si].snapshot(), 0.95);
                summary["streams"].push_back(item);
                summary["captured_total"] = summary["captured_total"].get<std::uint64_t>() + captured;
                summary["processed_total"] = summary["processed_total"].get<std::uint64_t>() + processed;
                summary["dropped_total"] = summary["dropped_total"].get<std::uint64_t>() + dropped;
                summary["downstream_frames_total"] =
                    summary["downstream_frames_total"].get<std::uint64_t>() + metric.downstream_frames;
                summary["backlog_loss_upper_bound_total"] =
                    summary["backlog_loss_upper_bound_total"].get<std::uint64_t>() +
                    metric.admitted_minus_pulled_upper_bound;
                summary["pts_gap_lost_total"] =
                    summary["pts_gap_lost_total"].get<std::uint64_t>() + metric.pts_gap_lost;
                summary["source_errors_total"] =
                    summary["source_errors_total"].get<std::uint64_t>() + metric.source_errors;
                summary["reconnect_triggers_total"] =
                    summary["reconnect_triggers_total"].get<std::uint64_t>() + metric.source_errors;
                summary["reconnects_total"] = summary["reconnects_total"].get<std::uint64_t>() + reconnects;
                std::cout << "[LIVE_STREAM_SUMMARY] idx=" << si
                          << " view=" << live_specs[si].name
                          << " screen_active=" << (streams[si].screen_active ? 1 : 0)
                          << " screen_count=" << streams[si].screens.size()
                          << " captured=" << captured
                          << " processed=" << processed
                          << " dropped=" << dropped
                          << " downstream_frames=" << metric.downstream_frames
                          << " backlog_loss_upper_bound=" << metric.admitted_minus_pulled_upper_bound
                          << " pts_gap_lost_estimate=" << metric.pts_gap_lost
                          << " source_errors=" << metric.source_errors
                          << " reconnects=" << reconnects
                          << " capture_fps=" << item["capture_fps"]
                          << " processed_fps=" << item["processed_fps"]
                          << " p50_ms=" << item["latency_p50_ms"]
                          << " p95_ms=" << item["latency_p95_ms"] << std::endl;
            }
            std::ofstream summary_out((fs::path(args.output_dir) / "live_summary.json").string());
            summary_out << std::setw(2) << summary << std::endl;
            std::cout << "[LIVE_SUMMARY] " << summary.dump() << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
