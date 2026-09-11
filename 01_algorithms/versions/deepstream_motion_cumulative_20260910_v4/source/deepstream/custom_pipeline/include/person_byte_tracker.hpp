#pragma once
#include "BYTETracker.h"
#include <vector>
#include <algorithm>
#include <cmath>
#include <stdexcept>
namespace jiankong::custom_pipeline {
struct PersonTrackMatch { int input_index; int track_id; };
class PersonByteTracker {
public:
    explicit PersonByteTracker(int fps = 10, float threshold = .25f)
        : tracker_(fps, 30, threshold), threshold_(threshold) {
        if (fps < 1 || !std::isfinite(threshold) || threshold < .10f || threshold > 1.f)
            throw std::invalid_argument("invalid person tracker FPS/confidence");
    }
    std::vector<PersonTrackMatch> update(const std::vector<Object>& input) {
        std::vector<Object> valid;
        for (size_t i=0; i<input.size(); ++i) {
            auto d=input[i];
            const auto& r=d.rect;
            // Pixel coordinates above this generous bound are not valid camera
            // detections and can overflow Kalman covariance/IoU calculations.
            constexpr float max_coordinate=1e6f;
            if (!std::isfinite(r.x) || !std::isfinite(r.y) ||
                !std::isfinite(r.width) || !std::isfinite(r.height) ||
                std::abs(r.x)>max_coordinate || std::abs(r.y)>max_coordinate ||
                r.width>max_coordinate || r.height>max_coordinate ||
                r.x+r.width<=r.x || r.y+r.height<=r.y ||
                r.width<1e-3f || r.height<1e-3f || !std::isfinite(d.prob) ||
                d.prob<.10f || d.prob>1.f) continue;
            d.label=static_cast<int>(i);
            valid.push_back(d);
        }
        std::vector<PersonTrackMatch> matches;
        for (const auto& t: tracker_.update(valid)) {
            if(t.detection_index<0 || t.detection_index>=static_cast<int>(input.size()))
                throw std::runtime_error("ByteTrack returned invalid detection provenance");
            // Low-score associations preserve identity only, never alarm evidence.
            if(input[t.detection_index].prob>=threshold_)
                matches.push_back({t.detection_index,t.track_id});
        }
        std::sort(matches.begin(),matches.end(),[](const auto& a,const auto& b) {
            return a.input_index<b.input_index;
        });
        return matches;
    }
private:
    BYTETracker tracker_;
    float threshold_;
};
}
