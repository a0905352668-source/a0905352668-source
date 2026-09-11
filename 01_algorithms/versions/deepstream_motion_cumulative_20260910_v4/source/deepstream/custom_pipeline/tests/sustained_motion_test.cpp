#include "sustained_motion_policy.hpp"
#include "gated_event_policy.hpp"
#include <cmath>
#include <cstdlib>
#include <iostream>
using namespace jiankong::custom_pipeline;
using PersonMotionPolicy = SustainedMotionPolicy;
void check(bool ok, const char* why) { if(!ok) { std::cerr<<why<<'\n'; std::exit(1); } }
int main() {
    // A brief coherent reposition must never clear a real aiming sequence.
    PersonMotionPolicy reposition;
    for(int i=0;i<40;++i) {
        double x=std::min(i,9)*8.;
        check(!reposition.update(i*.1,x,100,x,80,200,true,x,150,true),
              "brief posture adjustment must not be classified as sustained walking");
    }
    for(double fps : {8.,10.}) {
        PersonMotionPolicy walk, lean, box_change, sway;
        bool detected=false;
        for(int i=0;i<4*fps;++i) {
            double t=i/fps, x=28*t;
            bool moving=walk.update(t,x,100,x,80,200,true,x,150,true);
            if(t<1.8) check(!moving,"one short window cannot confirm walking");
            detected|=moving;
            check(!lean.update(t,x,100,x,80,200,true,0,150,true),"stationary hips protect seated leaning");
            check(!box_change.update(t,x,100,0,80,200,true,0,150,true),"box change alone is not walking");
            double jitter=9*std::sin(t*6);
            check(!sway.update(t,jitter,100,jitter,80,200,true,jitter,150,true),"bounded sway is not walking");
        }
        check(detected,"sustained slow whole-body translation must be detected");
        for(int i=int(4*fps);i<6*fps;++i)
            walk.update(i/fps,28*(4-1/fps),100,28*(4-1/fps),80,200,true,28*(4-1/fps),150,true);
        check(!walk.moving(),"stopping must release suppression");
    }
    PersonMotionPolicy unknown, jump;
    PersonMotionPolicy noisy_box;
    bool noisy_detected=false;
    for(int i=0;i<40;++i) {
        double t=i*.1, x=t*28;
        noisy_detected|=noisy_box.update(t,x+(i%2?4:-4),100,x,80,200,true,x,150,true);
    }
    check(noisy_detected,"box jitter must not hide coherent shoulder and hip translation");
    for(int i=0;i<50;++i) {
        check(!unknown.update(i*.1,i*8,100,i*8,80,200,true),"missing hips must not assert walking");
        double x=i<20?0:400;
        check(!jump.update(i*.1,x,100,x,80,200,true,x,150,true),"identity jump cannot join trajectories");
    }
    PersonMotionPolicy adjustment;
    GatedEventPolicy gate;
    bool alarm=false;
    for(int i=0;i<30;++i) {
        double x=std::min(i,9)*8.;
        GatedFrameEvidence e;
        e.phone_valid=e.high_confidence_phone=e.screen_ray_hit=true;
        e.person_association=e.hand_relation=e.screen_intent=EvidenceState::true_value;
        e.person_moving=adjustment.update(i*.1,x,100,x,80,200,true,x,150,true);
        alarm|=gate.update(e).alarm;
    }
    check(alarm,"brief reposition must preserve normal alarm accumulation");
    std::cout<<"sustained motion tests passed\n";
}
