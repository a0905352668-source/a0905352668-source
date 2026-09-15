#pragma once

#include <algorithm>

namespace jiankong::custom_pipeline {

inline bool live_infer_fps_is_supported(double infer_fps) {
    return infer_fps == 8.0 || infer_fps == 10.0;
}

inline bool should_build_phone_roi(bool screen_active) {
    return screen_active;
}

template <typename StateMap>
inline void prune_stale_track_states(StateMap& states, int frame_id, int window_size) {
    const int stale_after = std::max(window_size * 3, 90);
    for (auto it = states.begin(); it != states.end();) {
        if (frame_id - it->second.last_seen > stale_after) {
            it = states.erase(it);
        } else {
            ++it;
        }
    }
}

template <typename TrackSet, typename StateMap>
inline void reset_inactive_stream_state(float& alert_counter,
                                        TrackSet& previous_alarm_tracks,
                                        StateMap& states,
                                        int frame_id,
                                        int window_size) {
    alert_counter = 0.0f;
    previous_alarm_tracks.clear();
    prune_stale_track_states(states, frame_id, window_size);
}

}  // namespace jiankong::custom_pipeline
