#include "deepstream_batch_reader.hpp"

#include <gst/app/gstappsink.h>
#include <gst/gst.h>
#include <gstnvdsmeta.h>
#include <nvbufsurface.h>
#include <nvdsmeta.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace jiankong::custom_pipeline {
namespace {

double steady_seconds() {
    using Clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(Clock::now().time_since_epoch()).count();
}

double system_clock_unix_seconds() {
    using Clock = std::chrono::system_clock;
    return std::chrono::duration<double>(Clock::now().time_since_epoch()).count();
}

std::uint64_t steady_nanoseconds() {
    using Clock = std::chrono::steady_clock;
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count());
}

GstElement* make_element(const char* factory, const char* name) {
    GstElement* element = gst_element_factory_make(factory, name);
    if (element == nullptr) {
        throw std::runtime_error("missing GStreamer element: " + std::string(factory));
    }
    return element;
}

bool is_video_pad(GstPad* pad) {
    GstCaps* caps = gst_pad_get_current_caps(pad);
    if (caps == nullptr) caps = gst_pad_query_caps(pad, nullptr);
    bool result = false;
    if (caps != nullptr && gst_caps_get_size(caps) > 0) {
        const GstStructure* structure = gst_caps_get_structure(caps, 0);
        const gchar* name = gst_structure_get_name(structure);
        result = name != nullptr && g_str_has_prefix(name, "video/");
    }
    if (caps != nullptr) gst_caps_unref(caps);
    return result;
}

}  // namespace

struct DeepStreamBatchReader::Impl {
    struct SourceState {
        unsigned int id = 0;
        GstPad* mux_sink_pad = nullptr;
        GstClockTime interval_ns = GST_SECOND / 8;
        GstClockTime next_due_pts = GST_CLOCK_TIME_NONE;
        GstClockTime last_pts = GST_CLOCK_TIME_NONE;
        GstClockTime last_downstream_pts = GST_CLOCK_TIME_NONE;
        std::atomic<std::uint64_t> admitted{0};
        std::atomic<std::uint64_t> phase_dropped{0};
        std::atomic<std::uint64_t> downstream_frames{0};
        std::atomic<std::uint64_t> pts_gap_lost{0};
        std::atomic<std::uint64_t> source_errors{0};
        std::atomic<std::uint64_t> measurement_started_ns{std::numeric_limits<std::uint64_t>::max()};
        std::atomic<std::uint64_t> measurement_deadline_ns{0};
    };

    explicit Impl(BatchReaderConfig value) : config(std::move(value)) {}

    static bool metrics_active(const SourceState* source) {
        const std::uint64_t now = steady_nanoseconds();
        const std::uint64_t started = source->measurement_started_ns.load(std::memory_order_acquire);
        const std::uint64_t deadline = source->measurement_deadline_ns.load(std::memory_order_acquire);
        return now >= started && now <= deadline;
    }

    static GstPadProbeReturn phase_gate(GstPad*, GstPadProbeInfo* info, gpointer user_data) {
        auto* source = static_cast<SourceState*>(user_data);
        GstBuffer* buffer = GST_PAD_PROBE_INFO_BUFFER(info);
        if (buffer == nullptr) return GST_PAD_PROBE_OK;
        const bool collect_metrics = metrics_active(source);
        const GstClockTime pts = GST_BUFFER_PTS(buffer);
        if (!GST_CLOCK_TIME_IS_VALID(pts)) {
            if (collect_metrics) source->admitted.fetch_add(1, std::memory_order_relaxed);
            return GST_PAD_PROBE_OK;
        }
        if (!GST_CLOCK_TIME_IS_VALID(source->next_due_pts) ||
            (GST_CLOCK_TIME_IS_VALID(source->last_pts) && pts < source->last_pts)) {
            source->last_pts = pts;
            source->next_due_pts = pts + source->interval_ns;
            if (collect_metrics) source->admitted.fetch_add(1, std::memory_order_relaxed);
            return GST_PAD_PROBE_OK;
        }
        source->last_pts = pts;
        if (pts < source->next_due_pts) {
            if (collect_metrics) source->phase_dropped.fetch_add(1, std::memory_order_relaxed);
            return GST_PAD_PROBE_DROP;
        }
        const GstClockTime delta = pts - source->next_due_pts;
        const GstClockTime intervals = delta / source->interval_ns + 1;
        source->next_due_pts += intervals * source->interval_ns;
        if (collect_metrics) source->admitted.fetch_add(1, std::memory_order_relaxed);
        return GST_PAD_PROBE_OK;
    }

    static void on_pad_added(GstElement*, GstPad* pad, gpointer user_data) {
        auto* source = static_cast<SourceState*>(user_data);
        if (!is_video_pad(pad) || gst_pad_is_linked(pad)) return;
        gst_pad_add_probe(pad, GST_PAD_PROBE_TYPE_BUFFER, phase_gate, source, nullptr);
        if (gst_pad_link(pad, source->mux_sink_pad) != GST_PAD_LINK_OK) {
            g_printerr("source %u failed to link to nvstreammux\n", source->id);
        }
    }

    std::optional<unsigned int> source_id_for_message(GstMessage* message) const {
        for (GstObject* object = GST_MESSAGE_SRC(message); object != nullptr;
             object = GST_OBJECT_PARENT(object)) {
            const gchar* name = GST_OBJECT_NAME(object);
            if (name == nullptr || !g_str_has_prefix(name, "source-")) continue;
            char* end = nullptr;
            const unsigned long value = std::strtoul(name + 7, &end, 10);
            if (end != name + 7 && end != nullptr && *end == '\0') {
                return static_cast<unsigned int>(value);
            }
        }
        return std::nullopt;
    }

    SourceState* find_source(unsigned int source_id) const {
        for (const auto& source : sources) {
            if (source->id == source_id) return source.get();
        }
        return nullptr;
    }

    void record_downstream(unsigned int source_id, GstClockTime pts) {
        SourceState* source = find_source(source_id);
        if (source == nullptr || !metrics_active(source)) return;
        source->downstream_frames.fetch_add(1, std::memory_order_relaxed);
        if (GST_CLOCK_TIME_IS_VALID(pts) && GST_CLOCK_TIME_IS_VALID(source->last_downstream_pts)) {
            if (pts > source->last_downstream_pts) {
                source->pts_gap_lost.fetch_add(
                    estimate_missing_pts_frames(source->last_downstream_pts, pts, source->interval_ns),
                    std::memory_order_relaxed);
            }
        }
        source->last_downstream_pts = pts;
    }

    void check_bus() {
        while (GstMessage* message = gst_bus_pop_filtered(
                   bus, static_cast<GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_EOS))) {
            if (GST_MESSAGE_TYPE(message) == GST_MESSAGE_EOS) {
                is_terminal.store(true, std::memory_order_release);
            } else {
                GError* error = nullptr;
                gchar* debug = nullptr;
                gst_message_parse_error(message, &error, &debug);
                const std::string text = error != nullptr ? error->message : "unknown GStreamer error";
                const auto source_id = source_id_for_message(message);
                if (debug != nullptr) g_free(debug);
                if (error != nullptr) g_error_free(error);
                if (source_id.has_value()) {
                    if (SourceState* source = find_source(*source_id)) {
                        if (metrics_active(source)) {
                            source->source_errors.fetch_add(1, std::memory_order_relaxed);
                        }
                        g_printerr(
                            "source-scoped GStreamer error source=%u retained for nvurisrcbin reconnect: %s\n",
                            *source_id,
                            text.c_str());
                        gst_message_unref(message);
                        continue;
                    }
                }
                gst_message_unref(message);
                throw std::runtime_error("DeepStream pipeline error: " + text);
            }
            gst_message_unref(message);
        }
    }

    void cleanup() {
        if (pipeline != nullptr) gst_element_set_state(pipeline, GST_STATE_NULL);
        for (auto& source : sources) {
            if (source->mux_sink_pad != nullptr && streammux != nullptr) {
                gst_element_release_request_pad(streammux, source->mux_sink_pad);
                gst_object_unref(source->mux_sink_pad);
                source->mux_sink_pad = nullptr;
            }
        }
        if (bus != nullptr) {
            gst_object_unref(bus);
            bus = nullptr;
        }
        if (pipeline != nullptr) {
            gst_object_unref(pipeline);
            pipeline = nullptr;
            streammux = nullptr;
            appsink = nullptr;
        }
        started = false;
    }

    BatchReaderConfig config;
    GstElement* pipeline = nullptr;
    GstElement* streammux = nullptr;
    GstElement* appsink = nullptr;
    GstBus* bus = nullptr;
    std::vector<std::unique_ptr<SourceState>> sources;
    std::atomic<bool> is_terminal{false};
    bool started = false;
};

struct DeviceBatch::Impl {
    explicit Impl(GstSample* value) : sample(value) {
        try {
            buffer = gst_sample_get_buffer(sample);
            if (buffer == nullptr || !gst_buffer_map(buffer, &map, GST_MAP_READ)) {
                throw std::runtime_error("failed to map DeepStream output buffer");
            }
            mapped = true;
            surface = reinterpret_cast<NvBufSurface*>(map.data);
            if (surface == nullptr) throw std::runtime_error("mapped NvBufSurface is null");
            if (surface->memType != NVBUF_MEM_CUDA_DEVICE &&
                surface->memType != NVBUF_MEM_CUDA_UNIFIED) {
                throw std::runtime_error("NvBufSurface is not CUDA device-accessible memory");
            }
            NvDsBatchMeta* batch_meta = gst_buffer_get_nvds_batch_meta(buffer);
            if (batch_meta == nullptr) throw std::runtime_error("nvstreammux output has no batch metadata");
            const double received = steady_seconds();
            const double received_at_unix_seconds = system_clock_unix_seconds();
            for (NvDsMetaList* node = batch_meta->frame_meta_list; node != nullptr; node = node->next) {
                auto* frame = static_cast<NvDsFrameMeta*>(node->data);
                if (frame == nullptr || frame->batch_id >= surface->numFilled) continue;
                const NvBufSurfaceParams& params = surface->surfaceList[frame->batch_id];
                if (params.dataPtr == nullptr || params.layout != NVBUF_LAYOUT_PITCH ||
                    params.colorFormat != NVBUF_COLOR_FORMAT_RGBA) {
                    throw std::runtime_error("NvBufSurface frame is not pitch-linear RGBA");
                }
                frames.push_back(DeviceFrameView{
                    static_cast<const std::uint8_t*>(params.dataPtr),
                    static_cast<std::size_t>(params.pitch),
                    static_cast<int>(params.width),
                    static_cast<int>(params.height),
                    frame->source_id,
                    frame->batch_id,
                    static_cast<std::uint64_t>(frame->buf_pts),
                    GST_CLOCK_TIME_IS_VALID(frame->buf_pts),
                    received,
                    received_at_unix_seconds,
                });
            }
        } catch (...) {
            cleanup();
            throw;
        }
    }

    ~Impl() { cleanup(); }

    void cleanup() {
        if (mapped) gst_buffer_unmap(buffer, &map);
        if (sample != nullptr) gst_sample_unref(sample);
        mapped = false;
        sample = nullptr;
        buffer = nullptr;
    }

    GstSample* sample = nullptr;
    GstBuffer* buffer = nullptr;
    GstMapInfo map{};
    bool mapped = false;
    NvBufSurface* surface = nullptr;
    std::vector<DeviceFrameView> frames;
};

DeviceBatch::DeviceBatch() = default;
DeviceBatch::DeviceBatch(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
DeviceBatch::~DeviceBatch() = default;
DeviceBatch::DeviceBatch(DeviceBatch&&) noexcept = default;
DeviceBatch& DeviceBatch::operator=(DeviceBatch&&) noexcept = default;

const std::vector<DeviceFrameView>& DeviceBatch::frames() const {
    static const std::vector<DeviceFrameView> empty;
    return impl_ != nullptr ? impl_->frames : empty;
}

DeviceBatch::operator bool() const { return impl_ != nullptr; }

DeepStreamBatchReader::DeepStreamBatchReader(BatchReaderConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))) {
    if (impl_->config.sources.empty() ||
        (impl_->config.target_fps != 8 && impl_->config.target_fps != 10)) {
        throw std::invalid_argument(
            "custom pipeline requires one or more sources at 8 or 10 FPS");
    }
    if (impl_->config.width != 2560 || impl_->config.height != 1440) {
        throw std::invalid_argument("custom pipeline preserves the 2560x1440 calibration coordinate space");
    }
}

DeepStreamBatchReader::~DeepStreamBatchReader() { stop(); }

double DeepStreamBatchReader::begin_measurement(double duration_seconds) {
    if (!impl_->started) throw std::logic_error("DeepStreamBatchReader is not started");
    if (duration_seconds <= 0.0) throw std::invalid_argument("measurement duration must be positive");
    const std::uint64_t started_ns = steady_nanoseconds();
    const std::uint64_t duration_ns =
        static_cast<std::uint64_t>(duration_seconds * static_cast<double>(GST_SECOND));
    for (const auto& source : impl_->sources) {
        if (source->measurement_started_ns.load(std::memory_order_acquire) !=
            std::numeric_limits<std::uint64_t>::max()) {
            throw std::logic_error("DeepStream measurement can only be started once");
        }
    }
    for (const auto& source : impl_->sources) {
        source->last_downstream_pts = GST_CLOCK_TIME_NONE;
        source->measurement_deadline_ns.store(started_ns + duration_ns, std::memory_order_release);
    }
    for (const auto& source : impl_->sources) {
        source->measurement_started_ns.store(started_ns, std::memory_order_release);
    }
    return static_cast<double>(started_ns) / 1000000000.0;
}

void DeepStreamBatchReader::start() {
    if (impl_->started) return;
    gst_init(nullptr, nullptr);
    try {
    impl_->pipeline = gst_pipeline_new("jiankong-custom-pipeline");
    impl_->streammux = make_element("nvstreammux", "streammux");
    GstElement* converter = make_element("nvvideoconvert", "rgba-converter");
    GstElement* capsfilter = make_element("capsfilter", "rgba-caps");
    impl_->appsink = make_element("appsink", "batch-sink");
    if (impl_->pipeline == nullptr) throw std::runtime_error("failed to create GStreamer pipeline");

    g_object_set(impl_->streammux,
                 "batch-size", static_cast<guint>(impl_->config.sources.size()),
                 "width", impl_->config.width,
                 "height", impl_->config.height,
                 "batched-push-timeout", impl_->config.batched_push_timeout_us,
                 "live-source", TRUE,
                 "sync-inputs", FALSE,
                 "gpu-id", impl_->config.gpu_id,
                 "nvbuf-memory-type", 2,
                 nullptr);
    g_object_set(converter,
                 "gpu-id", impl_->config.gpu_id,
                 "nvbuf-memory-type", 2,
                 nullptr);
    GstCaps* rgba_caps = gst_caps_from_string("video/x-raw(memory:NVMM),format=(string)RGBA");
    g_object_set(capsfilter, "caps", rgba_caps, nullptr);
    gst_caps_unref(rgba_caps);
    g_object_set(impl_->appsink,
                 "sync", FALSE,
                 "async", FALSE,
                 "max-buffers", 1,
                 "drop", TRUE,
                 "emit-signals", FALSE,
                 nullptr);

    gst_bin_add_many(GST_BIN(impl_->pipeline), impl_->streammux, converter, capsfilter,
                     impl_->appsink, nullptr);
    if (!gst_element_link_many(impl_->streammux, converter, capsfilter, impl_->appsink, nullptr)) {
        impl_->cleanup();
        throw std::runtime_error("failed to link mux, RGBA converter and appsink");
    }

    for (const RtspSource& config : impl_->config.sources) {
        const std::string name = "source-" + std::to_string(config.id);
        GstElement* source = make_element("nvurisrcbin", name.c_str());
        g_object_set(source,
                     "uri", config.uri.c_str(),
                     "rtsp-reconnect-interval", impl_->config.reconnect_interval_seconds,
                     "rtsp-reconnect-attempts", impl_->config.reconnect_attempts,
                     "disable-audio", TRUE,
                     nullptr);
        gst_bin_add(GST_BIN(impl_->pipeline), source);

        auto state = std::make_unique<Impl::SourceState>();
        state->id = config.id;
        state->interval_ns = GST_SECOND / impl_->config.target_fps;
        const std::string pad_name = "sink_" + std::to_string(config.id);
        state->mux_sink_pad = gst_element_request_pad_simple(impl_->streammux, pad_name.c_str());
        if (state->mux_sink_pad == nullptr) {
            impl_->cleanup();
            throw std::runtime_error("failed to request nvstreammux pad " + pad_name);
        }
        Impl::SourceState* state_ptr = state.get();
        impl_->sources.push_back(std::move(state));
        g_signal_connect(source, "pad-added", G_CALLBACK(Impl::on_pad_added), state_ptr);
        if (GstPad* pad = gst_element_get_static_pad(source, "src")) {
            Impl::on_pad_added(source, pad, state_ptr);
            gst_object_unref(pad);
        }
    }

    impl_->bus = gst_element_get_bus(impl_->pipeline);
    impl_->is_terminal.store(false, std::memory_order_release);
    if (gst_element_set_state(impl_->pipeline, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) {
        impl_->cleanup();
        throw std::runtime_error("DeepStream pipeline refused PLAYING state");
    }
    impl_->started = true;
    } catch (...) {
        impl_->cleanup();
        throw;
    }
}

DeviceBatch DeepStreamBatchReader::pull(unsigned int timeout_ms) {
    if (!impl_->started) throw std::logic_error("DeepStreamBatchReader is not started");
    impl_->check_bus();
    GstSample* sample = gst_app_sink_try_pull_sample(
        GST_APP_SINK(impl_->appsink), static_cast<GstClockTime>(timeout_ms) * GST_MSECOND);
    impl_->check_bus();
    if (sample == nullptr) return DeviceBatch{};
    DeviceBatch result(std::make_unique<DeviceBatch::Impl>(sample));
    for (const auto& frame : result.frames()) {
        impl_->record_downstream(frame.source_id, static_cast<GstClockTime>(frame.pts_ns));
    }
    return result;
}

void DeepStreamBatchReader::stop() {
    if (impl_ != nullptr) impl_->cleanup();
}

bool DeepStreamBatchReader::terminal() const {
    return impl_->is_terminal.load(std::memory_order_acquire);
}

std::vector<SourceReaderMetrics> DeepStreamBatchReader::metrics() const {
    std::vector<SourceReaderMetrics> result;
    result.reserve(impl_->sources.size());
    for (const auto& source : impl_->sources) {
        const std::uint64_t admitted = source->admitted.load(std::memory_order_relaxed);
        const std::uint64_t downstream = source->downstream_frames.load(std::memory_order_relaxed);
        const std::uint64_t admitted_minus_pulled_upper_bound =
            admitted > downstream ? admitted - downstream : 0;
        result.push_back(SourceReaderMetrics{
            source->id,
            admitted,
            source->phase_dropped.load(std::memory_order_relaxed),
            downstream,
            admitted_minus_pulled_upper_bound,
            source->pts_gap_lost.load(std::memory_order_relaxed),
            source->source_errors.load(std::memory_order_relaxed),
        });
    }
    return result;
}

}  // namespace jiankong::custom_pipeline
