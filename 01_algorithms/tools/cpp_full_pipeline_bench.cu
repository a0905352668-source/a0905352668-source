#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <condition_variable>
#include <deque>
#include <cmath>
#include <cstring>
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

static double now_sec() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
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
};

struct ScreenConfig {
    std::string screen_id;
    std::vector<cv::Point2f> screen_poly;
    std::vector<cv::Point2f> near_zone;
    std::vector<Zone> danger_zones;
    std::vector<Zone> ignore_zones;
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
    float phone_hand_score = 0.0f;
    float screen_relation_score = 0.0f;
    float pose_score = 0.0f;
    float temporal_score = 0.0f;
    float risk_score = 0.0f;
    float best_angle = 180.0f;
    bool best_ray_hit = false;
    bool person_alarm = false;
    int person_window_hits = 0;
    std::string candidate_reason;
    bool accepted() const { return reject_reason.empty(); }
};

struct PersonTrackState {
    int track_id = -1;
    std::deque<Rect> bbox_history;
    std::deque<int> phone_history;
    std::deque<float> risk_history;
    std::deque<int> candidate_history;
    std::deque<int> handheld_history;
    std::string state = "S0_CLEAR";
    int stable_count = 0;
    int window_hits = 0;
    int handheld_phone_hits = 0;
    int handheld_phone_stable_count = 0;
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
    long long alarm_frames = 0;
    std::vector<ScreenConfig> screens;
    std::map<int, PersonTrackState> person_states;
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
        host_input_.resize(current_input_elems_);
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

static std::vector<cv::Point2f> parse_polygon(const json& arr, float sx, float sy) {
    std::vector<cv::Point2f> poly;
    if (!arr.is_array()) return poly;
    for (const auto& p : arr) {
        if (!p.is_array() || p.size() < 2) continue;
        poly.emplace_back(p[0].get<float>() * sx, p[1].get<float>() * sy);
    }
    return poly;
}

static void update_params_from_json(ScreenParams& p, const json& params) {
    if (!params.is_object()) return;
    auto getf = [&](const char* key, float& dst) {
        if (params.contains(key) && params[key].is_number()) dst = params[key].get<float>();
    };
    auto geti = [&](const char* key, int& dst) {
        if (params.contains(key) && params[key].is_number_integer()) dst = params[key].get<int>();
    };
    getf("person_expand_x", p.person_expand_x);
    getf("person_expand_y", p.person_expand_y);
    geti("near_zone_detect_interval", p.near_zone_detect_interval);
    getf("phone_valid_conf_person_roi", p.phone_valid_conf_person_roi);
    getf("phone_valid_conf_near_zone", p.phone_valid_conf_near_zone);
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
}

static std::vector<ScreenConfig> load_calibration(const std::string& path, int width, int height) {
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
    if (!data.contains("screens") || !data["screens"].is_array()) return screens;
    int idx = 1;
    for (const auto& item : data["screens"]) {
        ScreenConfig sc;
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
        if (sc.near_zone.empty() && !sc.screen_poly.empty()) {
            const Rect sb = bbox_from_polygon(sc.screen_poly);
            const Rect near = clamp_rect(Rect{sb.x1 - sb.w() * 1.2f, sb.y1 - sb.h() * 0.9f,
                                              sb.x2 + sb.w() * 1.2f, sb.y2 + sb.h() * 0.9f}, width, height);
            sc.near_zone = {{near.x1, near.y1}, {near.x2, near.y1}, {near.x2, near.y2}, {near.x1, near.y2}};
        }
        if (!sc.screen_poly.empty()) screens.push_back(std::move(sc));
        ++idx;
    }
    return screens;
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
    return clampf(0.30f * phone_score + 0.20f * hand_score + 0.20f * screen_score + 0.15f * pose_score + 0.15f * temporal_score, 0.0f, 1.0f);
}

static std::string candidate_level(float risk) {
    if (risk < 0.40f) return "ignore";
    if (risk < 0.65f) return "weak";
    if (risk < 0.80f) return "normal";
    return "strong";
}

static bool is_handheld_phone_suspect_candidate(const CandidateEval& ev) {
    if (ev.track_id < 0 || ev.person_index < 0) return false;
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
                                   int next_track_id, int frame_id, int window_size) {
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
            states[best_id].track_id = best_id;
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
    const int stale_after = std::max(window_size * 3, 90);
    for (auto it = states.begin(); it != states.end();) {
        if (frame_id - it->second.last_seen > stale_after) it = states.erase(it);
        else ++it;
    }
}

static void update_eval_risk(CandidateEval& ev, float risk_threshold) {
    (void)risk_threshold;
    ev.risk_score = risk_score(ev.phone_score, ev.phone_hand_score, ev.screen_relation_score, ev.pose_score, ev.temporal_score);
    ev.level = candidate_level(ev.risk_score);
    if (ev.person_index >= 0 && (ev.reject_reason.empty() || ev.reject_reason == "low_risk_score")) {
        if (ev.risk_score >= 0.40f && ev.static_zone_score > 0.0f && ev.person_match_score >= 0.35f) ev.reject_reason.clear();
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
    const float required_conf = screen.params.phone_valid_conf_person_roi;
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
    if (ev.person_index < 0 && preferred_person_index >= 0) {
        for (int i = 0; i < static_cast<int>(people.size()); ++i) {
            if (people[i].person_index != preferred_person_index) continue;
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
            auto [aim, best_angle, ray_hit, reason] = aim_score(phone, *person, screen, static_score, ev.phone_hand_score, kp_thr);
            (void)reason;
            ev.best_angle = best_angle;
            ev.best_ray_hit = ray_hit;
            ev.phone_score = phone_reliability_score(phone, required_conf);
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

static void update_person_states(std::vector<Det>& people, const std::vector<CandidateEval>& evals,
                                 std::map<int, PersonTrackState>& states, int frame_id,
                                 int window_size, float risk_threshold, int min_hits, int handheld_suspect_min_hits) {
    std::map<int, CandidateEval> best_by_track;
    for (const auto& ev : evals) {
        if (ev.track_id < 0) continue;
        auto it = best_by_track.find(ev.track_id);
        if (it == best_by_track.end() || ev.risk_score > it->second.risk_score) best_by_track[ev.track_id] = ev;
    }
    for (const auto& person : people) {
        if (person.track_id < 0) continue;
        auto& state = states[person.track_id];
        state.track_id = person.track_id;
        auto it = best_by_track.find(person.track_id);
        const bool has_eval = it != best_by_track.end();
        const float risk = has_eval ? it->second.risk_score : 0.0f;
        const bool phone_seen = has_eval && it->second.phone_score > 0.0f;
        const bool candidate = has_eval && it->second.accepted() && risk >= risk_threshold;
        const bool handheld_candidate = has_eval && is_handheld_phone_suspect_candidate(it->second);
        state.smoothed_bbox = state.has_smoothed_bbox ? lerp_rect(state.smoothed_bbox, person.box, 0.35f) : person.box;
        state.has_smoothed_bbox = true;
        push_limited(state.bbox_history, person.box, window_size);
        push_limited(state.phone_history, phone_seen ? 1 : 0, window_size);
        push_limited(state.risk_history, risk, window_size);
        push_limited(state.candidate_history, candidate ? 1 : 0, window_size);
        push_limited(state.handheld_history, handheld_candidate ? 1 : 0, window_size);
        state.window_hits = std::accumulate(state.candidate_history.begin(), state.candidate_history.end(), 0);
        state.stable_count = trailing_candidate_count(state.candidate_history);
        state.handheld_phone_hits = std::accumulate(state.handheld_history.begin(), state.handheld_history.end(), 0);
        state.handheld_phone_stable_count = trailing_candidate_count(state.handheld_history);
        const int handheld_stable_min = std::max(6, (handheld_suspect_min_hits + 1) / 2);
        const bool handheld_suspect = state.handheld_phone_hits >= handheld_suspect_min_hits
                                      || state.handheld_phone_stable_count >= handheld_stable_min;
        state.alarm_triggered = state.window_hits >= min_hits;
        if (state.alarm_triggered) state.state = "S4_ALARM";
        else if (state.window_hits >= std::max(3, min_hits / 2)) state.state = "S3_SUSTAINED_RISK";
        else if (candidate) state.state = "S2_RISK";
        else if (handheld_suspect) state.state = "S2_HAND_PHONE";
        else if (phone_seen) state.state = "S1_PHONE";
        else state.state = "S0_CLEAR";
        if (has_eval) {
            state.last_risk_score = std::max(state.last_risk_score * 0.90f, risk);
            if (!it->second.screen_id.empty()) state.last_screen_id = it->second.screen_id;
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

static std::vector<Det> decode_pose(const float* out, int batch_index, int channels, int num_boxes,
                                    const LetterboxMeta& meta, int width, int height, float conf_thr, float kp_thr) {
    const float* base = out + static_cast<size_t>(batch_index) * channels * num_boxes;
    std::vector<Det> candidates;
    candidates.reserve(128);
    for (int i = 0; i < num_boxes; ++i) {
        const float conf = base[4 * num_boxes + i];
        if (conf < conf_thr) continue;
        Det d;
        d.conf = conf;
        d.person_index = i;
        d.box = map_box(base[0 * num_boxes + i], base[1 * num_boxes + i], base[2 * num_boxes + i], base[3 * num_boxes + i],
                        meta, width, height);
        if (d.box.area() < 4.0f) continue;
        for (int k = 0; k < 17; ++k) {
            const int c = 5 + k * 3;
            d.kpts[k][0] = (base[c * num_boxes + i] - meta.pad_x) / meta.scale;
            d.kpts[k][1] = (base[(c + 1) * num_boxes + i] - meta.pad_y) / meta.scale;
            d.kpts[k][2] = base[(c + 2) * num_boxes + i];
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

static std::vector<Det> decode_phone_batch(const float* out, int real_batch, int channels, int num_boxes,
                                           const std::vector<LetterboxMeta>& metas, const std::vector<RoiJob>& jobs,
                                           int job_start, const std::vector<StreamState>& streams, float conf_thr) {
    std::vector<Det> phones;
    for (int b = 0; b < real_batch; ++b) {
        const RoiJob& job = jobs[job_start + b];
        const StreamState& st = streams[job.stream];
        const LetterboxMeta& m = metas[b];
        const float* base = out + static_cast<size_t>(b) * channels * num_boxes;
        std::vector<Det> candidates;
        for (int i = 0; i < num_boxes; ++i) {
            const float conf = base[4 * num_boxes + i];
            if (conf < conf_thr) continue;
            Det d;
            d.conf = conf;
            d.stream = job.stream;
            d.roi_index = job_start + b;
            d.person_index = job.person_index;
            Rect local = map_box(base[0 * num_boxes + i], base[1 * num_boxes + i], base[2 * num_boxes + i],
                                 base[3 * num_boxes + i], m, static_cast<int>(job.roi.w()), static_cast<int>(job.roi.h()));
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
    std::string calib_dir = "/media/boshi/Data/00_active_projects/JianKong/02_configs/surveillance/recalibration_20260706/generated_v21_from_labelme_20260706_103319";
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
};

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
        else throw std::runtime_error("unknown arg " + k);
    }
    return a;
}

static json screen_to_json(const ScreenConfig& screen, size_t screen_index) {
    json j;
    j["screen_index"] = screen_index;
    j["screen_id"] = screen.screen_id;
    j["screen_poly"] = polygon_to_json(screen.screen_poly);
    return j;
}

static int count_accepted(const std::vector<CandidateEval>& evals) {
    int n = 0;
    for (const auto& ev : evals) {
        if (ev.accepted()) ++n;
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
    if (accepted_count <= 0 && suspect_tracks.empty()) return;

    const long long frame_index = std::max<long long>(0, stream.frames - 1);
    const double time_sec = infer_fps > 0 ? static_cast<double>(frame_index) / infer_fps : 0.0;
    const float max_risk = std::max(max_risk_score(evals), state_max_risk);

    json j;
    j["stream_index"] = stream_index;
    j["frame_index"] = frame_index;
    j["frame_id"] = stream.frame_id;
    j["time_sec"] = time_sec;
    j["width"] = stream.width;
    j["height"] = stream.height;
    j["input_video"] = stream.path;
    j["output_video"] = stream.output_path;
    j["person_count"] = people.size();
    j["phone_count"] = phones.size();
    j["accepted_count"] = accepted_count;
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
        p["suspect"] = suspect;
        p["state"] = st != nullptr ? st->state : "";
        p["risk_score"] = ev != nullptr ? ev->risk_score : (st != nullptr ? st->last_risk_score : 0.0f);
        p["window_hits"] = st != nullptr ? st->window_hits : 0;
        p["handheld_phone_hits"] = st != nullptr ? st->handheld_phone_hits : 0;
        p["handheld_phone_stable_count"] = st != nullptr ? st->handheld_phone_stable_count : 0;
        p["screen_id"] = ev != nullptr ? ev->screen_id : (st != nullptr ? st->last_screen_id : "");
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
        p["suspect"] = true;
        p["state"] = st.state;
        p["risk_score"] = st.last_risk_score;
        p["window_hits"] = st.window_hits;
        p["handheld_phone_hits"] = st.handheld_phone_hits;
        p["handheld_phone_stable_count"] = st.handheld_phone_stable_count;
        p["screen_id"] = st.last_screen_id;
        p["last_seen_age"] = stream.frame_id - st.last_seen;
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
        p["candidate_reason"] = ev != nullptr ? ev->candidate_reason : "";
        p["zone_reason"] = ev != nullptr ? ev->zone_reason : "";
        p["static_zone_score"] = ev != nullptr ? ev->static_zone_score : 0.0f;
        p["person_match_score"] = ev != nullptr ? ev->person_match_score : 0.0f;
        p["phone_score"] = ev != nullptr ? ev->phone_score : 0.0f;
        p["phone_hand_score"] = ev != nullptr ? ev->phone_hand_score : 0.0f;
        p["screen_relation_score"] = ev != nullptr ? ev->screen_relation_score : 0.0f;
        p["pose_score"] = ev != nullptr ? ev->pose_score : 0.0f;
        p["temporal_score"] = ev != nullptr ? ev->temporal_score : 0.0f;
        p["best_angle"] = ev != nullptr ? ev->best_angle : 180.0f;
        p["best_ray_hit"] = ev != nullptr && ev->best_ray_hit;
        p["handheld_suspect"] = ev != nullptr && is_handheld_phone_suspect_candidate(*ev);
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

    for (int frame_id = 1; frame_id <= screen.params.person_state_min_hits; ++frame_id) {
        risky.track_id = people[0].track_id;
        update_person_states(people, {risky}, states, frame_id, screen.params.person_state_window,
                             screen.params.person_state_risk_threshold, screen.params.person_state_min_hits,
                             screen.params.handheld_suspect_min_hits);
    }
    if (!states[people[0].track_id].alarm_triggered || states[people[0].track_id].window_hits < screen.params.person_state_min_hits) {
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
    update_person_states(shifted_people, {}, states, screen.params.person_state_min_hits + 1,
                         screen.params.person_state_window,
                         screen.params.person_state_risk_threshold,
                         screen.params.person_state_min_hits,
                         screen.params.handheld_suspect_min_hits);
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
    if (side_eval.accepted() || !is_handheld_phone_suspect_candidate(side_eval)) {
        std::cerr << "[SELF_TEST_RULES] side-screen phone should be suspect-only, risk="
                  << side_eval.risk_score << " reject=" << side_eval.reject_reason
                  << " reason=" << side_eval.candidate_reason << std::endl;
        return 16;
    }
    for (int frame_id = 1; frame_id <= side_screen.params.handheld_suspect_min_hits; ++frame_id) {
        update_person_states(side_people, {side_eval}, side_states, frame_id, side_screen.params.person_state_window,
                             side_screen.params.person_state_risk_threshold, side_screen.params.person_state_min_hits,
                             side_screen.params.handheld_suspect_min_hits);
    }
    const auto& side_state = side_states[side_people[0].track_id];
    if (!is_suspect_track(side_state, side_screen.params.handheld_suspect_min_hits) || side_state.alarm_triggered) {
        std::cerr << "[SELF_TEST_RULES] side-screen handheld phone should hold suspect without alarm, hits="
                  << side_state.handheld_phone_hits << " alarm=" << side_state.alarm_triggered << std::endl;
        return 17;
    }

    std::cout << "[SELF_TEST_RULES] ok risk=" << std::fixed << std::setprecision(3)
              << risky.risk_score << " roi_only_reject=" << roi_only.reject_reason << std::endl;
    return 0;
}

int main(int argc, char** argv) {
    try {
        Args args = parse_args(argc, argv);
        if (args.self_test) return run_self_test();
        if (args.self_test_rules) return run_rule_self_test();
        cv::setNumThreads(std::max(0, args.cv_threads));

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

        std::vector<std::string> videos = find_videos_by_rank_from_end(args.root, args.pick_from_end);
        if (videos.empty()) throw std::runtime_error("no videos under " + args.root);
        if (static_cast<int>(videos.size()) > args.pose_batch) videos.resize(args.pose_batch);
        if (!args.output_dir.empty()) {
            fs::create_directories(args.output_dir);
        }
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
        for (const auto& path : videos) {
            StreamState s;
            s.path = path;
            s.cap.open(path);
            if (!s.cap.isOpened()) throw std::runtime_error("failed to open " + path);
            s.width = static_cast<int>(s.cap.get(cv::CAP_PROP_FRAME_WIDTH));
            s.height = static_cast<int>(s.cap.get(cv::CAP_PROP_FRAME_HEIGHT));
            s.native_fps = s.cap.get(cv::CAP_PROP_FPS);
            if (s.native_fps <= 0) s.native_fps = 25.0;
            s.total_frames = static_cast<int>(s.cap.get(cv::CAP_PROP_FRAME_COUNT));
            const std::string calib_name = calibration_name_for_path(path);
            const fs::path calib_path = calib_name.empty() ? fs::path() : fs::path(args.calib_dir) / calib_name;
            s.screens = load_calibration(calib_path.string(), s.width, s.height);
            make_sample_indices(s, args.infer_fps, args.max_samples);
            if (!args.output_dir.empty() && !args.no_video) {
                fs::path out_path = fs::path(args.output_dir) / (std::to_string(streams.size()) + "_" + safe_stem(path) + "_boxed.mp4");
                s.output_path = out_path.string();
                s.writer.open(out_path.string(), cv::VideoWriter::fourcc('m', 'p', '4', 'v'), args.infer_fps, cv::Size(s.width, s.height));
                if (!s.writer.isOpened()) {
                    throw std::runtime_error("failed to open output video: " + out_path.string());
                }
                std::cout << "[OUTPUT] idx=" << streams.size() << " path=" << out_path.string() << std::endl;
            }
            std::cout << "[STREAM] idx=" << streams.size() << " samples=" << s.sample_indices.size()
                      << " native_fps=" << s.native_fps << " size=" << s.width << "x" << s.height
                      << " path=" << s.path << std::endl;
            std::cout << "[CALIB] idx=" << streams.size()
                      << " file=" << (calib_name.empty() ? "NONE" : calib_path.string())
                      << " screens=" << s.screens.size() << std::endl;
            streams.push_back(std::move(s));
        }
        if (!args.output_dir.empty()) {
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

        std::vector<cv::Mat> frames(args.pose_batch);
        std::vector<int> active_streams;
        std::vector<LetterboxMeta> pose_metas(args.pose_batch);
        std::unique_ptr<GpuPreprocessor> gpu_preprocessor;
        if (args.gpu_preprocess) {
            gpu_preprocessor = std::make_unique<GpuPreprocessor>(streams.size());
        }
        std::vector<std::unique_ptr<FrameQueue>> queues;
        std::vector<std::thread> producers;
        std::vector<double> read_thread_times(streams.size(), 0.0);
        if (args.pipelined_read) {
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

        const double wall0 = now_sec();
        int step = 0;
        while (true) {
            active_streams.clear();
            double t0 = now_sec();
            if (args.pipelined_read) {
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
            }
            t_read += now_sec() - t0;
            if (active_streams.empty()) break;

            t0 = now_sec();
            if (args.gpu_preprocess) {
                for (int b = 0; b < static_cast<int>(active_streams.size()); ++b) {
                    const int si = active_streams[b];
                    gpu_preprocessor->upload_frame(si, frames[b], pose.stream());
                }
                for (int b = 0; b < args.pose_batch; ++b) {
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
                for (int b = 0; b < args.pose_batch; ++b) {
                    const int src_idx = b < static_cast<int>(active_streams.size()) ? b : 0;
                    pose_metas[b] = preprocess_into(frames[src_idx], args.pose_size, pose.input(), b, args.pose_batch);
                }
            }
            t_pose_prep += now_sec() - t0;

            t0 = now_sec();
            pose.infer(args.gpu_preprocess, static_cast<int>(active_streams.size()));
            t_pose_infer += now_sec() - t0;

            std::vector<RoiJob> roi_jobs;
            std::vector<std::vector<Det>> people_by_stream(streams.size());
            std::vector<std::vector<Rect>> rois_by_stream(streams.size());
            t0 = now_sec();
            for (int b = 0; b < static_cast<int>(active_streams.size()); ++b) {
                const int si = active_streams[b];
                streams[si].frame_id += 1;
                auto people_raw = decode_pose(pose.output(), b, pose_channels, pose_boxes, pose_metas[b],
                                              streams[si].width, streams[si].height, args.pose_conf, args.kp_conf);
                const long long raw_count = static_cast<long long>(people_raw.size());
                auto people = dedupe_people(std::move(people_raw), args.kp_conf);
                const int window_size = stream_state_window(streams[si]);
                streams[si].next_track_id = assign_person_track_ids(people, streams[si].person_states,
                                                                    streams[si].next_track_id,
                                                                    streams[si].frame_id,
                                                                    window_size);
                const long long deduped = std::max<long long>(0, raw_count - static_cast<long long>(people.size()));
                streams[si].frames += 1;
                streams[si].persons += static_cast<long long>(people.size());
                total_frames += 1;
                total_persons_raw += raw_count;
                total_persons_deduped += deduped;
                total_persons += static_cast<long long>(people.size());
                people_by_stream[si] = people;
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
                t0 = now_sec();
                if (args.gpu_preprocess) {
                    for (int b = 0; b < args.phone_batch; ++b) {
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
                    for (int b = 0; b < args.phone_batch; ++b) {
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
                phone.infer(args.gpu_preprocess, real);
                t_phone_infer += now_sec() - t0;

                t0 = now_sec();
                auto phones = decode_phone_batch(phone.output(), real, phone_channels, phone_boxes, metas, roi_jobs,
                                                 start, streams, args.phone_conf);
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
            for (int si : active_streams) {
                if (streams[si].screens.empty()) {
                    const int window_size = stream_state_window(streams[si]);
                    const int min_hits = stream_state_min_hits(streams[si], window_size);
                    update_person_states(people_by_stream[si], {}, streams[si].person_states,
                                         streams[si].frame_id, window_size,
                                         stream_state_risk_threshold(streams[si]), min_hits,
                                         stream_handheld_suspect_min_hits(streams[si], min_hits));
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
                const int window_size = stream_state_window(streams[si]);
                const int min_hits = stream_state_min_hits(streams[si], window_size);
                const float risk_thr = stream_state_risk_threshold(streams[si]);
                update_person_states(people_by_stream[si], evals_by_stream[si], streams[si].person_states,
                                     streams[si].frame_id, window_size, risk_thr, min_hits,
                                     stream_handheld_suspect_min_hits(streams[si], min_hits));

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
                for (auto& ev : evals_by_stream[si]) {
                    auto st = streams[si].person_states.find(ev.track_id);
                    if (st == streams[si].person_states.end()) continue;
                    ev.person_alarm = st->second.alarm_triggered;
                    ev.person_window_hits = st->second.window_hits;
                    if (ev.accepted()) {
                        streams[si].accepted_candidates += 1;
                        total_accepted_candidates += 1;
                    }
                }
                streams[si].previous_alarm_tracks = std::move(active_alarm_tracks);
                prune_person_tracks(streams[si].person_states, streams[si].frame_id, window_size);
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
                            cv::Scalar phone_color = cv::Scalar(120, 120, 120);
                            int phone_thickness = 1;
                            std::ostringstream label;
                            if (accepted_phone) {
                                phone_color = cv::Scalar(0, 0, 255);
                                phone_thickness = 3;
                                label << "PHONE " << std::fixed << std::setprecision(2) << ph.conf
                                      << " R" << ev->risk_score;
                            } else if (handheld_phone) {
                                phone_color = cv::Scalar(0, 165, 255);
                                phone_thickness = 2;
                                label << "PHONE " << std::fixed << std::setprecision(2) << ph.conf
                                      << " S" << ev->risk_score;
                            }
                            draw_rect(annotated, ph.box, phone_color, phone_thickness, label.str(), 0.40, false);
                        }
                    }
                    write_event_metadata(frame_events_jsonl,
                                         frame_events_csv,
                                         si,
                                         streams[si],
                                         args.infer_fps,
                                         args.person_expand_x,
                                         args.person_expand_y,
                                         people_by_stream[si],
                                         phones_keep_by_stream[si],
                                         evals_by_stream[si]);
                    if (!args.no_video) {
                        streams[si].writer.write(frames[b]);
                    }
                }
                t_draw_write += now_sec() - t0;
            }

            ++step;
            if (step % 25 == 0) {
                const double elapsed = now_sec() - wall0;
                std::cout << "[PROGRESS] step=" << step
                          << " frames=" << total_frames
                          << " fps=" << (elapsed > 0 ? total_frames / elapsed : 0.0)
                          << " persons=" << total_persons
                          << " persons_deduped=" << total_persons_deduped
                          << " rois=" << total_rois
                          << " accepted=" << total_accepted_candidates
                          << " alarm_frames=" << total_alarm_frames
                          << std::endl;
            }
        }

        if (args.pipelined_read) {
            for (auto& t : producers) {
                if (t.joinable()) t.join();
            }
            t_read = std::accumulate(read_thread_times.begin(), read_thread_times.end(), 0.0);
        }

        const double wall = now_sec() - wall0;
        std::cout << "[SUMMARY] frames=" << total_frames
                  << " wall_sec=" << wall
                  << " aggregate_fps=" << (wall > 0 ? total_frames / wall : 0.0)
                  << " persons=" << total_persons
                  << " persons_raw=" << total_persons_raw
                  << " persons_deduped=" << total_persons_deduped
                  << " rois=" << total_rois
                  << " phones_raw=" << total_phones_raw
                  << " phones_nms=" << total_phones_nms
                  << " accepted=" << total_accepted_candidates
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
                      << " alarm_frames=" << s.alarm_frames
                      << " screens=" << s.screens.size()
                      << " path=" << s.path
                      << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
