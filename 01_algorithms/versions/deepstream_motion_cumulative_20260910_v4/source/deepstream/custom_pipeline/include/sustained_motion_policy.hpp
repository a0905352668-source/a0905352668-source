#pragma once
#include <algorithm>
#include <cmath>
#include <deque>

namespace jiankong::custom_pipeline {
// Offline candidate: two non-overlapping windows of coherent box/shoulder/hip
// translation. Not a gait classifier. Phone/wrist motion is not an input.
// Thresholds require video regression before production integration.
class SustainedMotionPolicy {
    struct Sample { double t, x, y, sx, sy, h, hx, hy; };
    std::deque<Sample> history_;
    bool moving_ = false;
public:
    bool moving() const { return moving_; }
    bool update(double t, double x, double y, double sx, double sy, double h, bool valid,
                double hx=0, double hy=0, bool hips_valid=false) {
        if (!valid || !hips_valid || !std::isfinite(t+x+y+sx+sy+h+hx+hy) || h < 16) {
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
        history_.push_back({t,x,y,sx,sy,h,hx,hy});
        while(history_.size()>1 && (t-history_.front().t > 2.05 || history_.size()>128)) history_.pop_front();
        if(history_.size()<10 || t-history_.front().t < 1.8) return moving_;
        const auto& a=history_.front();
        std::size_t mid=1;
        while(mid+1<history_.size() && history_[mid].t < (a.t+t)*.5) ++mid;
        auto translation=[&](std::size_t first,std::size_t last) {
            const auto& p=history_[first]; const auto& q=history_[last];
            if(q.t-p.t<.8) return false;
            double scale=std::max(16.,(p.h+q.h)*.5);
            double dx=q.x-p.x,dy=q.y-p.y,sx=q.sx-p.sx,sy=q.sy-p.sy;
            double hx=q.hx-p.hx,hy=q.hy-p.hy;
            double d=std::hypot(dx,dy),s=std::hypot(sx,sy),hp=std::hypot(hx,hy),sp=0,hpath=0;
            for(std::size_t i=first+1;i<=last;++i) {
                sp+=std::hypot(history_[i].sx-history_[i-1].sx,history_[i].sy-history_[i-1].sy);
                hpath+=std::hypot(history_[i].hx-history_[i-1].hx,history_[i].hy-history_[i-1].hy);
            }
            return d/scale>=.06 && s/scale>=.08 && hp/scale>=.08 && s>=.45*sp && hp>=.55*hpath &&
                dx*sx+dy*sy>.7*d*s && dx*hx+dy*hy>.7*d*hp;
        };
        const auto& m=history_[mid]; const auto& b=history_.back();
        double dx1=m.x-a.x,dy1=m.y-a.y,dx2=b.x-m.x,dy2=b.y-m.y;
        bool same_direction=dx1*dx2+dy1*dy2>.7*std::hypot(dx1,dy1)*std::hypot(dx2,dy2);
        if(translation(0,mid) && translation(mid,history_.size()-1) && same_direction) moving_=true;
        else {
            // Release after a bounded quiet interval, not after an arbitrary
            // long-lived latch. Suspected motion never asserts moving_.
            std::size_t start=0;
            while(start+1<history_.size() && t-history_[start].t> .85) ++start;
            const auto& p=history_[start]; double span=0;
            for(std::size_t i=start;i<history_.size();++i) {
                const auto& q=history_[i];
                span=std::max(span,std::max({std::hypot(q.x-p.x,q.y-p.y),
                    std::hypot(q.sx-p.sx,q.sy-p.sy),std::hypot(q.hx-p.hx,q.hy-p.hy)}));
            }
            if(t-p.t>=.7 && span/std::max(16.,(p.h+h)*.5)<.10) moving_=false;
        }
        return moving_;
    }
};
}
