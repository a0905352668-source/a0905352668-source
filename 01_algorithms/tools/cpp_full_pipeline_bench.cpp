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
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

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
        v *= static_cast<size_t>(d.d[i]);
    }
    return v;
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

struct LetterboxMeta {
    float scale = 1.0f;
    int pad_x = 0;
    int pad_y = 0;
    int in_w = 0;
    int in_h = 0;
};

struct Det {
    Rect box;
    float conf = 0;
    int stream = -1;
    int roi_index = -1;
    float kpts[17][3]{};
};

struct RoiJob {
    int stream = -1;
    Rect roi;
};

struct StreamState {
    std::string path;
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
        if (n == size_) return;
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
            const size_t elems = volume(dims);
            const size_t bytes = elems * sizeof(float);
            check_cuda(cudaMalloc(&buffers_[i], bytes), "cudaMalloc");
            if (is_input) {
                input_index_ = i;
                input_dims_ = dims;
                input_elems_ = elems;
            } else {
                output_index_ = i;
                output_dims_ = dims;
                output_elems_ = elems;
            }
            std::cerr << "[ENGINE] " << plan_path << " binding=" << i
                      << " name=" << engine_->getBindingName(i)
                      << " input=" << is_input
                      << " dims=" << dims_to_string(dims)
                      << std::endl;
        }
        if (input_index_ < 0 || output_index_ < 0) throw std::runtime_error("missing bindings");
        host_input_.resize(input_elems_);
        host_output_.resize(output_elems_);
        check_cuda(cudaStreamCreate(&stream_), "cudaStreamCreate");
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

    void infer() {
        check_cuda(cudaMemcpyAsync(buffers_[input_index_], host_input_.data(), input_elems_ * sizeof(float),
                                   cudaMemcpyHostToDevice, stream_),
                   "cudaMemcpyAsync H2D");
        if (!context_->enqueueV2(buffers_.data(), stream_, nullptr)) {
            throw std::runtime_error("enqueueV2 failed");
        }
        check_cuda(cudaMemcpyAsync(host_output_.data(), buffers_[output_index_], output_elems_ * sizeof(float),
                                   cudaMemcpyDeviceToHost, stream_),
                   "cudaMemcpyAsync D2H");
        check_cuda(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
    }

private:
    static std::string dims_to_string(const nvinfer1::Dims& d) {
        std::ostringstream oss;
        for (int i = 0; i < d.nbDims; ++i) {
            if (i) oss << "x";
            oss << d.d[i];
        }
        return oss.str();
    }

    nvinfer1::IRuntime* runtime_ = nullptr;
    nvinfer1::ICudaEngine* engine_ = nullptr;
    nvinfer1::IExecutionContext* context_ = nullptr;
    cudaStream_t stream_ = nullptr;
    std::vector<void*> buffers_;
    int input_index_ = -1;
    int output_index_ = -1;
    nvinfer1::Dims input_dims_{};
    nvinfer1::Dims output_dims_{};
    size_t input_elems_ = 0;
    size_t output_elems_ = 0;
    PinnedFloatBuffer host_input_;
    PinnedFloatBuffer host_output_;
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

static std::vector<std::string> find_last_videos(const std::string& root) {
    std::vector<std::string> out;
    for (const auto& ent : fs::directory_iterator(root)) {
        if (!ent.is_directory()) continue;
        fs::path best;
        fs::file_time_type best_time{};
        for (const auto& f : fs::directory_iterator(ent.path())) {
            if (!f.is_regular_file()) continue;
            std::string ext = lower(f.path().extension().string());
            if (ext != ".mp4" && ext != ".avi" && ext != ".mkv" && ext != ".mov") continue;
            auto mt = fs::last_write_time(f.path());
            if (best.empty() || mt > best_time || (mt == best_time && f.path().filename() > best.filename())) {
                best = f.path();
                best_time = mt;
            }
        }
        if (!best.empty()) out.push_back(best.string());
    }
    std::sort(out.begin(), out.end(), [](const std::string& a, const std::string& b) {
        int oa = view_order(a), ob = view_order(b);
        if (oa != ob) return oa < ob;
        return a < b;
    });
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
    LetterboxMeta meta;
    meta.in_w = src.cols;
    meta.in_h = src.rows;
    const float r = std::min(size / static_cast<float>(src.cols), size / static_cast<float>(src.rows));
    const int new_w = std::max(1, static_cast<int>(std::round(src.cols * r)));
    const int new_h = std::max(1, static_cast<int>(std::round(src.rows * r)));
    meta.scale = r;
    meta.pad_x = (size - new_w) / 2;
    meta.pad_y = (size - new_h) / 2;

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

static Rect expand_person_roi(const Rect& r, int width, int height) {
    const float dx = r.w() * 0.35f;
    const float dy = r.h() * 0.20f;
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
    std::string phone_plan = "/media/boshi/Data/JianKong/06_training_runs/raw_trt_plans_20260706_114432/phone640_static_b16.plan";
    int max_samples = 400;
    double infer_fps = 10.0;
    int pose_size = 960;
    int phone_size = 640;
    int pose_batch = 7;
    int phone_batch = 16;
    float pose_conf = 0.25f;
    float kp_conf = 0.35f;
    float phone_conf = 0.25f;
    bool pipelined_read = false;
    int queue_size = 8;
    int cv_threads = 1;
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
        else if (k == "--pose-conf") a.pose_conf = std::stof(next());
        else if (k == "--kp-conf") a.kp_conf = std::stof(next());
        else if (k == "--phone-conf") a.phone_conf = std::stof(next());
        else if (k == "--pipelined-read") a.pipelined_read = true;
        else if (k == "--queue-size") a.queue_size = std::stoi(next());
        else if (k == "--cv-threads") a.cv_threads = std::stoi(next());
        else throw std::runtime_error("unknown arg " + k);
    }
    return a;
}

int main(int argc, char** argv) {
    try {
        Args args = parse_args(argc, argv);
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

        std::vector<std::string> videos = find_last_videos(args.root);
        if (videos.empty()) throw std::runtime_error("no videos under " + args.root);
        if (static_cast<int>(videos.size()) > args.pose_batch) videos.resize(args.pose_batch);

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
            make_sample_indices(s, args.infer_fps, args.max_samples);
            std::cout << "[STREAM] idx=" << streams.size() << " samples=" << s.sample_indices.size()
                      << " native_fps=" << s.native_fps << " size=" << s.width << "x" << s.height
                      << " path=" << s.path << std::endl;
            streams.push_back(std::move(s));
        }

        double t_read = 0, t_pose_prep = 0, t_pose_infer = 0, t_pose_post = 0;
        double t_roi_prep = 0, t_phone_infer = 0, t_phone_post = 0;
        long long total_frames = 0, total_persons = 0, total_rois = 0, total_phones_raw = 0, total_phones_nms = 0;

        std::vector<cv::Mat> frames(args.pose_batch);
        std::vector<int> active_streams;
        std::vector<LetterboxMeta> pose_metas(args.pose_batch);
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
            for (int b = 0; b < args.pose_batch; ++b) {
                const int src_idx = b < static_cast<int>(active_streams.size()) ? b : 0;
                pose_metas[b] = preprocess_into(frames[src_idx], args.pose_size, pose.input(), b, args.pose_batch);
            }
            t_pose_prep += now_sec() - t0;

            t0 = now_sec();
            pose.infer();
            t_pose_infer += now_sec() - t0;

            std::vector<RoiJob> roi_jobs;
            t0 = now_sec();
            for (int b = 0; b < static_cast<int>(active_streams.size()); ++b) {
                const int si = active_streams[b];
                auto people = decode_pose(pose.output(), b, pose_channels, pose_boxes, pose_metas[b],
                                          streams[si].width, streams[si].height, args.pose_conf, args.kp_conf);
                streams[si].frames += 1;
                streams[si].persons += static_cast<long long>(people.size());
                total_frames += 1;
                total_persons += static_cast<long long>(people.size());
                for (const auto& p : people) {
                    RoiJob job;
                    job.stream = si;
                    job.roi = expand_person_roi(p.box, streams[si].width, streams[si].height);
                    if (job.roi.w() >= 2 && job.roi.h() >= 2) {
                        roi_jobs.push_back(job);
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
                t_roi_prep += now_sec() - t0;

                t0 = now_sec();
                phone.infer();
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
            for (int si : active_streams) {
                std::vector<Det> per;
                for (const auto& ph : all_phones) {
                    if (ph.stream == si) per.push_back(ph);
                }
                auto keep = nms(std::move(per), 0.50f, 100);
                streams[si].phones_nms += static_cast<long long>(keep.size());
                total_phones_nms += static_cast<long long>(keep.size());
            }
            t_phone_post += now_sec() - t0;

            ++step;
            if (step % 25 == 0) {
                const double elapsed = now_sec() - wall0;
                std::cout << "[PROGRESS] step=" << step
                          << " frames=" << total_frames
                          << " fps=" << (elapsed > 0 ? total_frames / elapsed : 0.0)
                          << " persons=" << total_persons
                          << " rois=" << total_rois
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
                  << " rois=" << total_rois
                  << " phones_raw=" << total_phones_raw
                  << " phones_nms=" << total_phones_nms
                  << std::endl;
        std::cout << "[TIMING] read=" << t_read
                  << " pose_prep=" << t_pose_prep
                  << " pose_infer=" << t_pose_infer
                  << " pose_post=" << t_pose_post
                  << " roi_prep=" << t_roi_prep
                  << " phone_infer=" << t_phone_infer
                  << " phone_post=" << t_phone_post
                  << std::endl;
        for (size_t i = 0; i < streams.size(); ++i) {
            const auto& s = streams[i];
            std::cout << "[STREAM_SUMMARY] idx=" << i
                      << " frames=" << s.frames
                      << " persons=" << s.persons
                      << " rois=" << s.rois
                      << " phones_raw=" << s.phones_raw
                      << " phones_nms=" << s.phones_nms
                      << " path=" << s.path
                      << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
