#include "deepstream_batch_reader.hpp"
#include "live_calibration_policy.hpp"
#include "live_pipeline_policy.hpp"
#include "measurement_window.hpp"
#include "rtsp_live_runtime.hpp"

#include <array>
#include <map>
#include <set>
#include <stdexcept>
#include <string>

namespace {

void require(bool condition) {
    if (!condition) throw std::runtime_error("runtime contract failed");
}

}  // namespace

int main() {
    using jiankong::custom_pipeline::ScreenActivation;
    using jiankong::custom_pipeline::classify_live_calibration;
    using jiankong::custom_pipeline::completion_is_within_measurement;
    using jiankong::custom_pipeline::effective_measurement_seconds;
    using jiankong::custom_pipeline::estimate_missing_pts_frames;
    using jiankong::custom_pipeline::live_infer_fps_is_supported;
    using jiankong::custom_pipeline::reset_inactive_stream_state;
    using jiankong::custom_pipeline::should_build_phone_roi;
    using jiankong::BoundedSampleWindow;

    require(live_infer_fps_is_supported(8.0));
    require(live_infer_fps_is_supported(10.0));
    require(!live_infer_fps_is_supported(7.999));
    require(!live_infer_fps_is_supported(8.001));
    require(!live_infer_fps_is_supported(10.001));
    require(should_build_phone_roi(true));
    require(!should_build_phone_roi(false));

    BoundedSampleWindow latency_window(3);
    latency_window.push_back(1.0);
    latency_window.push_back(2.0);
    require(latency_window.size() == 2);
    latency_window.push_back(3.0);
    latency_window.push_back(4.0);
    require(latency_window.size() == 3);
    require(latency_window.capacity() == 3);
    require(jiankong::percentile(latency_window.snapshot(), 0.0) == 2.0);
    require(jiankong::percentile(latency_window.snapshot(), 0.5) == 3.0);
    require(jiankong::percentile(latency_window.snapshot(), 1.0) == 4.0);
    bool zero_capacity_rejected = false;
    try {
        BoundedSampleWindow invalid_window(0);
    } catch (const std::invalid_argument&) {
        zero_capacity_rejected = true;
    }
    require(zero_capacity_rejected);

    BoundedSampleWindow sustained_window;
    constexpr std::size_t sustained_sample_count = 100000;
    for (std::size_t i = 0; i < sustained_sample_count; ++i) {
        sustained_window.push_back(static_cast<double>(i));
    }
    require(sustained_window.size() == BoundedSampleWindow::kDefaultCapacity);
    require(jiankong::percentile(sustained_window.snapshot(), 0.0) ==
            static_cast<double>(sustained_sample_count - BoundedSampleWindow::kDefaultCapacity));
    require(jiankong::percentile(sustained_window.snapshot(), 1.0) ==
            static_cast<double>(sustained_sample_count - 1));

    struct TrackFixture {
        int last_seen = 0;
    };
    float inactive_alert_counter = 7.0f;
    std::set<int> inactive_alarm_tracks{11, 12};
    std::map<int, TrackFixture> inactive_states{
        {11, TrackFixture{109}},
        {12, TrackFixture{110}},
        {13, TrackFixture{199}},
    };
    reset_inactive_stream_state(
        inactive_alert_counter, inactive_alarm_tracks, inactive_states, 200, 10);
    require(inactive_alert_counter == 0.0f);
    require(inactive_alarm_tracks.empty());
    require(inactive_states.count(11) == 0);
    require(inactive_states.count(12) == 1);
    require(inactive_states.count(13) == 1);

    constexpr std::uint64_t interval = 125000000ULL;
    require(estimate_missing_pts_frames(0, 0, interval) == 0);
    require(estimate_missing_pts_frames(100, 200, 0) == 0);
    require(estimate_missing_pts_frames(200, 100, interval) == 0);
    require(estimate_missing_pts_frames(0, 120000000ULL, interval) == 0);
    require(estimate_missing_pts_frames(0, 160000000ULL, interval) == 0);
    require(estimate_missing_pts_frames(0, 240000000ULL, interval) == 1);
    require(estimate_missing_pts_frames(0, 360000000ULL, interval) == 2);

    constexpr double started = 100.0;
    constexpr double deadline = 105.0;
    require(completion_is_within_measurement(100.0, deadline));
    require(completion_is_within_measurement(105.0, deadline));
    require(!completion_is_within_measurement(105.000001, deadline));
    require(effective_measurement_seconds(started, deadline, 5.0) == 5.0);
    require(effective_measurement_seconds(started, 103.5, 5.0) == 3.5);
    require(effective_measurement_seconds(started, 110.0, 5.0) == 5.0);
    require(effective_measurement_seconds(started, 99.0, 5.0) == 0.0);

    struct CalibrationFixture {
        const char* name;
        std::size_t screen_count;
        ScreenActivation expected;
    };
    const std::array<CalibrationFixture, 7> fixtures{{
        {"camera_01_screen_calibration_v21.json", 0, ScreenActivation::inactive},
        {"camera_02_screen_calibration_v21.json", 1, ScreenActivation::active},
        {"camera_mechanical_01_screen_calibration_v21.json", 1, ScreenActivation::active},
        {"camera_mechanical_02_screen_calibration_v21.json", 1, ScreenActivation::active},
        {"camera_software_01_screen_calibration_v21.json", 1, ScreenActivation::active},
        {"camera_software_02_screen_calibration_v21.json", 1, ScreenActivation::active},
        {"camera_corridor_screen_calibration_v21.json", 1, ScreenActivation::active},
    }};
    for (const auto& fixture : fixtures) {
        require(classify_live_calibration(fixture.name, true, fixture.screen_count, true) ==
                fixture.expected);
    }
    require(classify_live_calibration("camera_01_screen_calibration_v21.json", true, 1, true) ==
            ScreenActivation::invalid);
    require(classify_live_calibration("camera_02_screen_calibration_v21.json", true, 0, true) ==
            ScreenActivation::invalid);
    require(classify_live_calibration("camera_02_screen_calibration_v21.json", true, 1, false) ==
            ScreenActivation::invalid);
    require(classify_live_calibration("camera_02_screen_calibration_v21.json", false, 1, true) ==
            ScreenActivation::invalid);
    return 0;
}
