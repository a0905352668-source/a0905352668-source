#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cmath>
#include <cstdint>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace jiankong {

struct StreamSpec {
    std::string name;
    std::string url;
};

inline StreamSpec parse_stream_spec(const std::string& value) {
    const auto pos = value.find('=');
    if (pos == std::string::npos || pos == 0 || pos + 1 >= value.size()) {
        throw std::invalid_argument("RTSP stream must be name=rtsp://url");
    }
    StreamSpec spec{value.substr(0, pos), value.substr(pos + 1)};
    if (spec.url.rfind("rtsp://", 0) != 0 && spec.url.rfind("rtsps://", 0) != 0) {
        throw std::invalid_argument("RTSP stream URL must start with rtsp:// or rtsps://");
    }
    return spec;
}

class RateGate {
public:
    explicit RateGate(double target_fps)
        : period_(target_fps > 0.0 ? 1.0 / target_fps : 0.0) {
        if (target_fps <= 0.0) throw std::invalid_argument("target_fps must be positive");
    }

    bool accept(double timestamp) {
        if (!initialized_ || timestamp + 1e-9 >= next_due_) {
            initialized_ = true;
            next_due_ = timestamp + period_;
            return true;
        }
        return false;
    }

private:
    double period_ = 0.0;
    double next_due_ = 0.0;
    bool initialized_ = false;
};

template <typename T>
struct CapturedFrame {
    T payload;
    std::uint64_t sequence = 0;
    double captured_at = 0.0;
};

template <typename T>
class LatestFrameSlot {
public:
    bool push(T payload, std::uint64_t sequence, double captured_at) {
        std::lock_guard<std::mutex> lock(mutex_);
        const bool replaced = slot_.has_value();
        if (replaced) ++dropped_;
        slot_ = CapturedFrame<T>{std::move(payload), sequence, captured_at};
        return replaced;
    }

    std::optional<CapturedFrame<T>> pop_latest() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!slot_.has_value()) return std::nullopt;
        auto result = std::move(slot_);
        slot_.reset();
        return result;
    }

    std::uint64_t dropped() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return dropped_;
    }

private:
    mutable std::mutex mutex_;
    std::optional<CapturedFrame<T>> slot_;
    std::uint64_t dropped_ = 0;
};

inline double percentile(std::vector<double> values, double q) {
    if (values.empty()) return 0.0;
    q = std::max(0.0, std::min(1.0, q));
    std::sort(values.begin(), values.end());
    const auto rank = static_cast<std::size_t>(std::ceil(q * values.size()));
    return values[std::max<std::size_t>(1, rank) - 1];
}

class BoundedSampleWindow {
public:
    static constexpr std::size_t kDefaultCapacity = 4096;

    explicit BoundedSampleWindow(std::size_t capacity = kDefaultCapacity)
        : capacity_(capacity) {
        if (capacity_ == 0) throw std::invalid_argument("sample window capacity must be positive");
        values_.reserve(capacity_);
    }

    void push_back(double value) {
        if (values_.size() < capacity_) {
            values_.push_back(value);
            return;
        }
        values_[next_] = value;
        next_ = (next_ + 1) % capacity_;
    }

    std::size_t size() const { return values_.size(); }
    std::size_t capacity() const { return capacity_; }
    std::vector<double> snapshot() const { return values_; }

private:
    std::size_t capacity_;
    std::size_t next_ = 0;
    std::vector<double> values_;
};

struct StreamMetrics {
    std::atomic<std::uint64_t> captured{0};
    std::atomic<std::uint64_t> processed{0};
    std::atomic<std::uint64_t> dropped{0};
    std::atomic<std::uint64_t> reconnects{0};
    BoundedSampleWindow latency_ms;
    mutable std::mutex latency_mutex;

    void record_latency(double value_ms) {
        std::lock_guard<std::mutex> lock(latency_mutex);
        latency_ms.push_back(value_ms);
    }

    std::vector<double> latency_snapshot() const {
        std::lock_guard<std::mutex> lock(latency_mutex);
        return latency_ms.snapshot();
    }
};

}  // namespace jiankong
