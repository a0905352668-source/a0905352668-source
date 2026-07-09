#include <algorithm>
#include <atomic>
#include <chrono>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/opencv.hpp>

struct StreamResult {
    std::string path;
    int index = 0;
    double native_fps = 0.0;
    int width = 0;
    int height = 0;
    long long decoded_frames = 0;
    long long sampled_frames = 0;
    double elapsed_sec = 0.0;
    bool ok = false;
    std::string error;
};

static double now_sec() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

static StreamResult run_stream(const std::string& path, int index, int interval, int max_sampled) {
    StreamResult r;
    r.path = path;
    r.index = index;
    double t0 = now_sec();
    cv::VideoCapture cap(path);
    if (!cap.isOpened()) {
        r.error = "failed to open";
        r.elapsed_sec = now_sec() - t0;
        return r;
    }
    r.native_fps = cap.get(cv::CAP_PROP_FPS);
    r.width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    r.height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    cv::Mat frame;
    long long frame_idx = 0;
    interval = std::max(1, interval);
    while (true) {
        bool should_retrieve = (frame_idx % interval) == 0;
        bool ok = false;
        if (should_retrieve) {
            ok = cap.read(frame);
            if (!ok) break;
            ++r.sampled_frames;
        } else {
            ok = cap.grab();
            if (!ok) break;
        }
        ++r.decoded_frames;
        ++frame_idx;
        if (max_sampled > 0 && r.sampled_frames >= max_sampled) break;
    }
    r.elapsed_sec = now_sec() - t0;
    r.ok = true;
    return r;
}

int main(int argc, char** argv) {
    int interval = 2;
    int max_sampled = 0;
    std::vector<std::string> videos;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--interval" && i + 1 < argc) {
            interval = std::stoi(argv[++i]);
        } else if (arg == "--max-sampled" && i + 1 < argc) {
            max_sampled = std::stoi(argv[++i]);
        } else {
            videos.push_back(arg);
        }
    }
    if (videos.empty()) {
        std::cerr << "usage: cpp_video_decode_bench [--interval N] [--max-sampled N] video..." << std::endl;
        return 2;
    }
    cv::setNumThreads(1);
    std::vector<StreamResult> results(videos.size());
    std::vector<std::thread> threads;
    double wall0 = now_sec();
    for (size_t i = 0; i < videos.size(); ++i) {
        threads.emplace_back([&, i]() {
            results[i] = run_stream(videos[i], static_cast<int>(i), interval, max_sampled);
        });
    }
    for (auto& t : threads) t.join();
    double wall = now_sec() - wall0;
    long long total_decoded = 0;
    long long total_sampled = 0;
    for (const auto& r : results) {
        total_decoded += r.decoded_frames;
        total_sampled += r.sampled_frames;
        std::cout << "STREAM idx=" << r.index
                  << " ok=" << (r.ok ? 1 : 0)
                  << " sampled=" << r.sampled_frames
                  << " decoded=" << r.decoded_frames
                  << " elapsed=" << r.elapsed_sec
                  << " sampled_fps=" << (r.elapsed_sec > 0 ? r.sampled_frames / r.elapsed_sec : 0.0)
                  << " decoded_fps=" << (r.elapsed_sec > 0 ? r.decoded_frames / r.elapsed_sec : 0.0)
                  << " native_fps=" << r.native_fps
                  << " size=" << r.width << "x" << r.height
                  << " path=" << r.path;
        if (!r.error.empty()) std::cout << " error=" << r.error;
        std::cout << std::endl;
    }
    std::cout << "TOTAL streams=" << videos.size()
              << " interval=" << interval
              << " total_sampled=" << total_sampled
              << " total_decoded=" << total_decoded
              << " wall_sec=" << wall
              << " aggregate_sampled_fps=" << (wall > 0 ? total_sampled / wall : 0.0)
              << " aggregate_decoded_fps=" << (wall > 0 ? total_decoded / wall : 0.0)
              << std::endl;
    return 0;
}
