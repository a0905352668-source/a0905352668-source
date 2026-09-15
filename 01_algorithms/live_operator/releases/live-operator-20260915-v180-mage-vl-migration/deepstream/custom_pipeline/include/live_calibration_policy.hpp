#pragma once

#include <cstddef>
#include <string>

namespace jiankong::custom_pipeline {

enum class ScreenActivation { invalid, inactive, active };

inline ScreenActivation classify_live_calibration(const std::string& calibration_name,
                                                   bool has_explicit_screens_array,
                                                   std::size_t declared_screen_count,
                                                   bool all_declared_polygons_valid) {
    if (!has_explicit_screens_array) return ScreenActivation::invalid;
    if (calibration_name == "camera_01_screen_calibration_v21.json") {
        return declared_screen_count == 0 ? ScreenActivation::inactive : ScreenActivation::invalid;
    }
    return declared_screen_count > 0 && all_declared_polygons_valid
               ? ScreenActivation::active
               : ScreenActivation::invalid;
}

}  // namespace jiankong::custom_pipeline
