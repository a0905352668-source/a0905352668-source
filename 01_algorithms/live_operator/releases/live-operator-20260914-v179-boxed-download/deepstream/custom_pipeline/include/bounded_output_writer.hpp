#pragma once
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <exception>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

namespace jiankong {
// One producer / one sink. Capacity includes the item currently being written.
// Full queues wait; records are never dropped. close() drains and reports errors.
class BoundedOutputWriter {
public:
    struct Metrics {
        std::size_t pending_bytes = 0;
        std::size_t high_water_bytes = 0;
        std::size_t backpressure_count = 0;
    };
    using Sink = std::function<void(int, const std::string&)>;
    BoundedOutputWriter(std::size_t capacity, Sink sink)
        : capacity_(capacity), sink_(std::move(sink)) {
        if (!capacity_ || !sink_) throw std::invalid_argument("invalid output writer");
        thread_ = std::thread([this] { run(); });
    }
    BoundedOutputWriter(const BoundedOutputWriter&) = delete;
    BoundedOutputWriter& operator=(const BoundedOutputWriter&) = delete;
    ~BoundedOutputWriter() { try { close(); } catch (...) {} }
    void submit(int channel, std::string bytes) {
        // Empty records consume one byte of budget so job count is bounded too.
        const auto size = bytes.empty() ? std::size_t{1} : bytes.size();
        if (size > capacity_) throw std::length_error("output record exceeds queue capacity");
        std::unique_lock<std::mutex> lock(mutex_);
        if (!error_ && !closing_ && size > capacity_ - metrics_.pending_bytes)
            ++metrics_.backpressure_count;
        space_.wait(lock, [&] {
            return error_ || closing_ || size <= capacity_ - metrics_.pending_bytes;
        });
        if (error_) std::rethrow_exception(error_);
        if (closing_) throw std::runtime_error("output writer is closed");
        queue_.push_back({channel, std::move(bytes), size});
        metrics_.pending_bytes += size;
        if (metrics_.pending_bytes > metrics_.high_water_bytes)
            metrics_.high_water_bytes = metrics_.pending_bytes;
        ready_.notify_one();
    }
    Metrics metrics() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return metrics_;
    }
    void throw_if_failed() const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (error_) std::rethrow_exception(error_);
    }
    void close() {
        { std::lock_guard<std::mutex> lock(mutex_); closing_ = true; }
        ready_.notify_one();
        space_.notify_all();
        if (thread_.joinable()) thread_.join();
        if (error_) std::rethrow_exception(error_);
    }
private:
    struct Item { int channel; std::string bytes; std::size_t size; };
    void run() noexcept {
        try {
            for (;;) {
                Item item;
                {
                    std::unique_lock<std::mutex> lock(mutex_);
                    ready_.wait(lock, [&] { return closing_ || !queue_.empty(); });
                    if (queue_.empty()) return;
                    item = std::move(queue_.front());
                    queue_.pop_front();
                }
                sink_(item.channel, item.bytes);
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    metrics_.pending_bytes -= item.size;
                }
                space_.notify_all();
            }
        } catch (...) {
            { std::lock_guard<std::mutex> lock(mutex_); error_ = std::current_exception(); }
            space_.notify_all();
        }
    }
    const std::size_t capacity_;
    Sink sink_;
    mutable std::mutex mutex_;
    std::condition_variable ready_, space_;
    std::deque<Item> queue_;
    Metrics metrics_;
    bool closing_ = false;
    std::exception_ptr error_;
    std::thread thread_;
};
} // namespace jiankong
