#include "static_phone_hotspot.hpp"

#include <cmath>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <utility>

using namespace jiankong::custom_pipeline;

namespace {

int failures = 0;

void require(bool value, const char* message) {
    if (!value) {
        std::cerr << "static_phone_hotspot_test: " << message << '\n';
        ++failures;
    }
}

int run_test() {
    CameraStaticHotspotMap overlapping_episode(2560, 1440);
    overlapping_episode.confirm(1111.0f, 388.0f, 0.0);
    overlapping_episode.confirm(1111.0f, 388.0f, 1.0);
    overlapping_episode.confirm(1111.0f, 388.0f, 2.0);
    require(overlapping_episode.score(1111.0f, 388.0f, 2.0) < 0.5f,
            "overlapping confirmations within three seconds must count once");
    require(overlapping_episode.cells().at({36, 12}).last_confirmed_at == 0.0,
            "ignored confirmations must not refresh the independent confirmation time");
    overlapping_episode.confirm(1111.0f, 388.0f, 3.0);
    require(overlapping_episode.score(1111.0f, 388.0f, 3.0) < 1.0f,
            "two independent confirmations must not create a hotspot");
    overlapping_episode.confirm(1111.0f, 388.0f, 6.0);
    require(overlapping_episode.score(1111.0f, 388.0f, 6.0) >= 1.0f,
            "three confirmations spaced by at least three seconds must create a hotspot");

    CameraStaticHotspotMap map(2560, 1440);
    require(map.score(1111.0f, 388.0f, 0.0) == 0.0f,
            "a new camera map must be empty");

    map.confirm(1111.0f, 388.0f, 1.0);
    require(map.cells().count({36, 12}) == 1,
            "cell size must be max(24 px, 0.012 * frame width)");
    map.confirm(1113.0f, 389.0f, 5.0);
    map.confirm(1110.0f, 387.0f, 9.0);
    require(map.score(1112.0f, 388.0f, 10.0) >= 1.0f,
            "three independent confirmations must create a hotspot");
    require(map.score(1400.0f, 900.0f, 10.0) == 0.0f,
            "a distant position must not inherit hotspot heat");

    map.penalize(1112.0f, 388.0f, 11.0);
    require(map.score(1112.0f, 388.0f, 11.0) < 1.0f,
            "one moving or handheld observation must remove hotspot confidence");

    CameraStaticHotspotMap saturated(2560, 1440);
    for (int confirmation = 0; confirmation < 12; ++confirmation) {
        saturated.confirm(1111.0f, 388.0f, static_cast<double>(confirmation));
    }
    saturated.penalize(1111.0f, 388.0f, 12.0);
    require(saturated.score(1111.0f, 388.0f, 12.0) < 1.0f,
            "one moving observation must revoke even saturated hotspot confidence");

    CameraStaticHotspotMap expired(2560, 1440);
    expired.confirm(1111.0f, 388.0f, 1.0);
    expired.confirm(1111.0f, 388.0f, 2.0);
    expired.confirm(1111.0f, 388.0f, 3.0);
    expired.decay(3.0 + 24.0 * 60.0 * 60.0);
    require(expired.score(1111.0f, 388.0f, 3.0 + 24.0 * 60.0 * 60.0) == 0.0f,
            "heat must expire after 24 hours without confirmation");
    require(expired.cells().empty(), "decay must remove expired cells");

    CameraStaticHotspotMap minimum_cell_map(1000, 600);
    minimum_cell_map.confirm(47.0f, 47.0f, 1.0);
    require(minimum_cell_map.cells().count({1, 1}) == 1,
            "small frames must use the 24 px minimum cell size");

    CameraStaticHotspotMap restored(2560, 1440);
    restored.restore(map.cells(), 12.0);
    require(restored.cells().size() == map.cells().size(),
            "valid persisted cells must restore");

    std::map<std::pair<int, int>, StaticHotspotCell> timestamp_cells;
    timestamp_cells[{0, 0}] = StaticHotspotCell{1.2f, 102.0};
    timestamp_cells[{1, 0}] = StaticHotspotCell{1.2f, 100.0 - 24.0 * 60.0 * 60.0};
    timestamp_cells[{2, 0}] = StaticHotspotCell{1.2f, 99.0};
    CameraStaticHotspotMap timestamp_checked(2560, 1440);
    timestamp_checked.restore(timestamp_cells, 100.0);
    require(timestamp_checked.cells().count({0, 0}) == 0,
            "restore must reject confirmations beyond the wall-clock tolerance");
    require(timestamp_checked.cells().count({1, 0}) == 0,
            "restore must discard cells already expired for 24 hours");
    require(timestamp_checked.cells().count({2, 0}) == 1,
            "restore must retain a current finite cell");

    CameraStaticHotspotMap invalid_restore_clock(2560, 1440);
    invalid_restore_clock.restore(
        timestamp_cells, std::numeric_limits<double>::quiet_NaN());
    require(invalid_restore_clock.cells().empty(),
            "restore must reject all cells when the current wall clock is non-finite");

    const float nan = std::numeric_limits<float>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    const std::size_t before_invalid = restored.cells().size();
    restored.confirm(nan, 388.0f, 12.0);
    restored.confirm(1112.0f, 388.0f, inf);
    restored.penalize(1112.0f, nan, 12.0);
    require(restored.score(nan, 388.0f, 12.0) == 0.0f,
            "non-finite coordinates must score zero");
    require(restored.score(1112.0f, 388.0f, inf) == 0.0f,
            "non-finite timestamps must score zero");
    require(restored.cells().size() == before_invalid,
            "non-finite input must not mutate the map");

    std::map<std::pair<int, int>, StaticHotspotCell> invalid_cells;
    invalid_cells[{0, 0}] = StaticHotspotCell{
        std::numeric_limits<float>::quiet_NaN(), 1.0};
    invalid_cells[{-1, 0}] = StaticHotspotCell{1.0f, 1.0};
    invalid_cells[{0, 1}] = StaticHotspotCell{1.0f, inf};
    CameraStaticHotspotMap defensive(-1, 0);
    defensive.restore(invalid_cells, 2.0);
    defensive.confirm(-1.0f, 0.0f, 1.0);
    require(defensive.cells().empty(),
            "invalid dimensions, cells, and out-of-frame input must remain safe");

    const std::string first_path = static_hotspot_state_filename("camera04", 0);
    const std::string second_path = static_hotspot_state_filename("camera04", 1);
    require(first_path == "camera04__stream_0.json",
            "hotspot state path must keep the readable camera name and stream index");
    require(second_path == "camera04__stream_1.json" && first_path != second_path,
            "equal camera view names must still produce unique stream state paths");

    std::ifstream pipeline_source(JIANKONG_PIPELINE_SOURCE_PATH, std::ios::binary);
    std::ostringstream pipeline_buffer;
    pipeline_buffer << pipeline_source.rdbuf();
    const std::string pipeline = pipeline_buffer.str();
    require(pipeline_source.good() || pipeline_source.eof(),
            "pipeline source must be readable for the persistence contract");
    const auto path_position = pipeline.find("static_hotspot_state_filename(");
    const auto indexed_arguments_position = pipeline.find(
        "camera_name, stream_index)", path_position);
    require(path_position != std::string::npos &&
                indexed_arguments_position != std::string::npos,
            "pipeline persistence path must use the stable current stream index");
    const auto push_position = pipeline.find("streams.push_back", path_position);
    require(path_position != std::string::npos && push_position != std::string::npos &&
                path_position < push_position,
            "stream index must be captured before streams.push_back");

    return failures == 0 ? 0 : 1;
}

}  // namespace

int main() {
    try {
        return run_test();
    } catch (const std::exception& error) {
        std::cerr << "static_phone_hotspot_test: unexpected exception: "
                  << error.what() << '\n';
        return 2;
    } catch (...) {
        std::cerr << "static_phone_hotspot_test: unexpected non-standard exception\n";
        return 3;
    }
}
