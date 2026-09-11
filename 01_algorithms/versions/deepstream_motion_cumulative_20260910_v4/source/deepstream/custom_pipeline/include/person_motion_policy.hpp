#pragma once
#include <algorithm>
#include <cmath>
#include <deque>

namespace jiankong::custom_pipeline {
// Per-track, camera-independent translation of BOTH body box and shoulders.
// Phone/wrist motion is deliberately not an input. Coordinates are source pixels.
class PersonMotionPolicy {
    struct Sample { double t, x, y, sx, sy, h; };
    std::deque<Sample> history_;
    bool moving_ = false;
public:
    bool moving() const { return moving_; }
    bool update(double t, double x, double y, double sx, double sy, double h, bool valid) {
        if (!valid || !std::isfinite(t+x+y+sx+sy+h) || h < 16) {
            // Unknown shoulders are not evidence of stopping. Bridge short
            // occlusions, but never carry a stale identity indefinitely.
            if (history_.empty() || !std::isfinite(t) || t<history_.back().t ||
                t-history_.back().t>.5) { history_.clear(); moving_=false; }
            return moving_;
        }
        if (!history_.empty()) {
            const auto& p=history_.back();
            if (t <= p.t || t-p.t > .5 || std::hypot(x-p.x,y-p.y) > .75*std::min(h,p.h)) {
                history_.clear(); moving_=false;
            }
        }
        history_.push_back({t,x,y,sx,sy,h});
        while(history_.size()>1 && (t-history_.front().t > 1.05 || history_.size()>64)) history_.pop_front();
        if(history_.size()<5 || t-history_.front().t < .7) return moving_;
        const auto& a=history_.front();
        const double scale=std::max(16.0,(a.h+h)*.5);
        const double dx=x-a.x, dy=y-a.y, dsx=sx-a.sx, dsy=sy-a.sy;
        const double distance=std::hypot(dx,dy), shoulders=std::hypot(dsx,dsy);
        double path=0, span=0;
        for(std::size_t i=1;i<history_.size();++i) {
            const auto& p=history_[i-1]; const auto& q=history_[i];
            path+=std::hypot(q.x-p.x,q.y-p.y);
            span=std::max(span,std::hypot(q.x-a.x,q.y-a.y));
        }
        const bool coherent=distance >= .75*path && dx*dsx+dy*dsy > .7*distance*shoulders;
        if(distance/scale >= .12 && shoulders/scale >= .10 && coherent) moving_=true;
        // Hysteresis prevents stop/start flicker and gives the gate a clean restart.
        else if(t-a.t >= .8 && span/scale < .10 && shoulders/scale < .12) moving_=false;
        return moving_;
    }
};
}
