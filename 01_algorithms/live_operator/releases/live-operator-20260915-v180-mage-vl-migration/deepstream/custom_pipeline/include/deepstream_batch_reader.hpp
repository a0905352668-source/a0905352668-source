#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace jiankong::custom_pipeline {

struct RtspSource {
    unsigned int id = 0;
    std::string uri;
};

struct BatchReaderConfig {
    std::vector<RtspSource> sources;
    unsigned int target_fps = 8;
    unsigned int width = 2560;
    unsigned int height = 1440;
    unsigned int batched_push_timeout_us = 125000;
    unsigned int reconnect_interval_seconds = 5;
    int reconnect_attempts = -1;
    unsigned int gpu_id = 0;
};

struct DeviceFrameView {
    const std::uint8_t* rgba = nullptr;
    std::size_t pitch_bytes = 0;
    int width = 0;
    int height = 0;
    unsigned int source_id = 0;
    unsigned int batch_id = 0;
    std::uint64_t pts_ns = 0;
    bool pts_valid = false;
    double received_at_seconds = 0.0;
    double received_at_unix_seconds = 0.0;
};

struct SourceReaderMetrics {
    unsigned int source_id = 0;
    std::uint64_t admitted = 0;
    std::uint64_t phase_dropped = 0;
    std::uint64_t downstream_frames = 0;
    std::uint64_t admitted_minus_pulled_upper_bound = 0;
    std::uint64_t pts_gap_lost = 0;
    std::uint64_t source_errors = 0;
};

inline std::uint64_t estimate_missing_pts_frames(std::uint64_t previous_pts_ns,
                                                 std::uint64_t current_pts_ns,
                                                 std::uint64_t expected_interval_ns) {
    if (expected_interval_ns == 0 || current_pts_ns <= previous_pts_ns) return 0;
    const std::uint64_t gap = current_pts_ns - previous_pts_ns;
    const std::uint64_t periods = (gap + expected_interval_ns / 2) / expected_interval_ns;
    return periods > 1 ? periods - 1 : 0;
}

class DeviceBatch {
public:
    DeviceBatch();
    ~DeviceBatch();
    DeviceBatch(DeviceBatch&&) noexcept;
    DeviceBatch& operator=(DeviceBatch&&) noexcept;
    DeviceBatch(const DeviceBatch&) = delete;
    DeviceBatch& operator=(const DeviceBatch&) = delete;

    const std::vector<DeviceFrameView>& frames() const;
    explicit operator bool() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    explicit DeviceBatch(std::unique_ptr<Impl> impl);
    friend class DeepStreamBatchReader;
};

class DeepStreamBatchReader {
public:
    explicit DeepStreamBatchReader(BatchReaderConfig config);
    ~DeepStreamBatchReader();
    DeepStreamBatchReader(const DeepStreamBatchReader&) = delete;
    DeepStreamBatchReader& operator=(const DeepStreamBatchReader&) = delete;

    void start();
    double begin_measurement(double duration_seconds);
    DeviceBatch pull(unsigned int timeout_ms);
    void stop();
    bool terminal() const;
    std::vector<SourceReaderMetrics> metrics() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace jiankong::custom_pipeline
