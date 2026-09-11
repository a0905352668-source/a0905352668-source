#include "person_motion_policy.hpp"
#include "gated_event_policy.hpp"
#include <cstdlib>
#include <iostream>
using namespace jiankong::custom_pipeline;
static void require(bool ok, const char* message) {
    if (!ok) { std::cerr << message << '\n'; std::exit(1); }
}
static GatedFrameEvidence strong(bool moving) {
    GatedFrameEvidence e;
    e.phone_valid = e.high_confidence_phone = e.screen_ray_hit = true;
    e.person_association = e.hand_relation = e.screen_intent = EvidenceState::true_value;
    e.person_moving = moving;
    return e;
}
int main() {
    for(double fps : {8.0,10.0}) {
        PersonMotionPolicy slow;
        GatedEventPolicy gate; gate.set_infer_fps(fps);
        for(int i=0;i<int(4*fps);++i) {
            double x=i/fps*28;
            require(!gate.update(strong(slow.update(i/fps,x,100,x,80,200,true))).alarm,
                "slow walking comparable to event233 must be suppressed before alarm");
        }
    }
    {
        PersonMotionPolicy sway;
        for(int i=0;i<30;++i) sway.update(i*.1,i*8,100,i*8,80,200,true);
        bool released=false;
        for(int i=30;i<100;++i) {
            double a=(i-30)*.1*3.141592653589793;
            double x=252+9*std::cos(a), y=100+9*std::sin(a);
            bool moving=sway.update(i*.1,x,y,x,y-20,200,true);
            if(i>60 && !moving) released=true;
        }
        require(released, "stopped body sway must release previous movement suppression");
    }
    {
        PersonMotionPolicy partial;
        GatedEventPolicy gate; gate.set_infer_fps(10);
        for(int i=0;i<100;++i) {
            bool moving=partial.update(i*.1,i*8,100,i*8,80,200,i%6!=0);
            require(!gate.update(strong(moving)).alarm, "brief shoulder loss must not defeat walking filter");
        }
        require(partial.moving(), "brief unknown pose must preserve known motion");
        require(!partial.update(12,0,0,0,0,0,false), "stale unknown pose must expire motion state");
    }
    for (double fps : {8.0, 10.0}) {
        PersonMotionPolicy walking;
        GatedEventPolicy gate; gate.set_infer_fps(fps);
        bool moving = false;
        for (int i = 0; i < int(4*fps); ++i) {
            const double x = i/fps*80;
            moving = walking.update(i/fps, x, 100, x, 80, 200, true);
            auto d = gate.update(strong(moving));
            require(!d.alarm && !d.review, "walking must not alarm or enter review");
        }
        require(moving, "continuous whole-body translation must be detected");
        bool rearmed = false;
        for (int i = int(4*fps); i < int(8*fps); ++i) {
            moving = walking.update(i/fps, (4-1/fps)*80, 100, (4-1/fps)*80, 80, 200, true);
            auto d = gate.update(strong(moving));
            if (i < int(4.6*fps)) require(!d.alarm, "stopping must not replay walking evidence");
            rearmed |= d.alarm;
        }
        require(rearmed, "stopped user must become eligible again");
    }
    PersonMotionPolicy jitter, turning, vertical, distant, jump, gaps;
    for (int i=0;i<50;++i) {
        double t=i*.1, n=(i%2 ? 3 : -3);
        require(!jitter.update(t,100+n,100,100+n,80,200,true), "jitter is not walking");
        require(!turning.update(t,100,100,100+i*7,80,200,true), "shoulder-only motion is not walking");
        vertical.update(t,100,i*8,100,i*8-20,200,true);
        distant.update(t,i*2,100,i*2,80,50,true);
        require(!jump.update(t,i<20?100:450,100,i<20?100:450,80,200,true), "single ID jump is not walking");
    }
    require(vertical.moving(), "vertical translation must be detected");
    require(distant.moving(), "motion threshold must scale with person size");
    for(int i=0;i<7;++i) gaps.update(i*.1,i*8,100,i*8,80,200,true);
    require(!gaps.update(3,300,100,300,80,200,true), "track gap must not join unrelated motion");
    require(!gaps.update(3.1,300,100,300,80,0,false), "invalid pose must not assert walking");
    GatedEventPolicy latched;
    for(int i=0;i<20;++i) latched.update(strong(false));
    auto d=latched.update(strong(true));
    require(!d.alarm && !d.review && d.reject_reason=="person_moving", "movement must clear latched alarm");
    for(int i=0;i<9;++i) require(!latched.update(strong(false)).alarm, "stationary evidence must restart from zero");
    require(latched.update(strong(false)).alarm, "original gate resumes after ten stationary frames at8fps");
    std::cout << "person motion tests passed\n";
}
