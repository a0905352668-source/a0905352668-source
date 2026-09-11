#pragma once

#include <algorithm>

namespace jiankong::custom_pipeline {

inline bool completion_is_within_measurement(double completed_at, double deadline) {
    return completed_at <= deadline;
}

inline double effective_measurement_seconds(double started_at,
                                            double stopped_at,
                                            double planned_seconds) {
    if (planned_seconds <= 0.0 || stopped_at <= started_at) return 0.0;
    return std::min(planned_seconds, stopped_at - started_at);
}

}  // namespace jiankong::custom_pipeline
