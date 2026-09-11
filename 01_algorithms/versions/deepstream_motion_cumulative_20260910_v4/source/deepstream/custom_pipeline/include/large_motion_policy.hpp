#pragma once
#include <algorithm>
#include <cmath>
#include <deque>

namespace jiankong::custom_pipeline {
// Conservative large-motion filter. No hips and no phone/wrist motion input.
// A short candidate signal lets the event gate wait briefly when whole-body
// translation is already obvious but has not yet crossed the full threshold.
// Phone/wrist motion is deliberately not an input. Coordinates are source pixels.
class LargeMotionPolicy {
    struct Sample { double t, x, y, sx, sy, h; };
    std::deque<Sample> history_;
    bool moving_ = false;
    bool candidate_ = false;

    bool qualifies(std::size_t begin, double minimum_span,
                   double body_ratio, double shoulder_ratio,
                   double minimum_coherence,
                   double minimum_half_contribution) const {
        if (begin >= history_.size() || history_.size() - begin < 5) return false;
        const auto& a=history_[begin];
        const auto& z=history_.back();
        if (z.t-a.t < minimum_span) return false;
        const double scale=std::max(16.0,(a.h+z.h)*.5);
        const double dx=z.x-a.x, dy=z.y-a.y;
        const double dsx=z.sx-a.sx, dsy=z.sy-a.sy;
        const double distance=std::hypot(dx,dy);
        const double shoulders=std::hypot(dsx,dsy);
        double path=0;
        for(std::size_t i=begin+1;i<history_.size();++i) {
            const auto& p=history_[i-1]; const auto& q=history_[i];
            path+=std::hypot(q.x-p.x,q.y-p.y);
        }
        const bool coherent=distance >= minimum_coherence*path &&
            dx*dsx+dy*dsy > .7*distance*shoulders;
        std::size_t mid=begin+1;
        while(mid+1<history_.size() && history_[mid].t<(a.t+z.t)*.5) ++mid;
        const auto& m=history_[mid];
        const double d1=std::hypot(m.x-a.x,m.y-a.y);
        const double d2=std::hypot(z.x-m.x,z.y-m.y);
        const double s1=std::hypot(m.sx-a.sx,m.sy-a.sy);
        const double s2=std::hypot(z.sx-m.sx,z.sy-m.sy);
        const bool sustained=d1>=minimum_half_contribution*distance &&
            d2>=minimum_half_contribution*distance &&
            s1>=minimum_half_contribution*shoulders &&
            s2>=minimum_half_contribution*shoulders;
        return distance/scale >= body_ratio && shoulders/scale >= shoulder_ratio &&
            coherent && sustained;
    }
public:
    bool moving() const { return moving_; }
    bool candidate() const { return candidate_; }
    bool update(double t, double x, double y, double sx, double sy, double h, bool valid) {
        if (!valid || !std::isfinite(t+x+y+sx+sy+h) || h < 16) {
            // Unknown shoulders are not evidence of stopping. Bridge short
            // occlusions, but never carry a stale identity indefinitely.
            if (history_.empty() || !std::isfinite(t) || t<history_.back().t ||
                t-history_.back().t>.5) { history_.clear(); moving_=false; }
            candidate_=false;
            return moving_;
        }
        if (!history_.empty()) {
            const auto& p=history_.back();
            if (t <= p.t || t-p.t > .5 || std::hypot(x-p.x,y-p.y) > .75*std::min(h,p.h)) {
                history_.clear(); moving_=false; candidate_=false;
            }
        }
        history_.push_back({t,x,y,sx,sy,h});
        while(history_.size()>1 && (t-history_.front().t > 3.05 || history_.size()>96))
            history_.pop_front();
        candidate_=false;
        std::size_t short_begin=0;
        while(short_begin+1<history_.size() && t-history_[short_begin].t>1.05)
            ++short_begin;
        candidate_=qualifies(short_begin,.7,.30,.27,.75,.20);
        const bool fast_large=qualifies(short_begin,.7,.45,.40,.75,.20);
        // Keep the original one-second rule strict.  A separate long window
        // catches slower but continuous whole-body translation while allowing
        // bounded detector jitter and gradual acceleration.
        const bool gradual_large=qualifies(0,2.4,.22,.18,.40,.10);
        moving_=fast_large || gradual_large;
        return moving_;
    }
};
}
