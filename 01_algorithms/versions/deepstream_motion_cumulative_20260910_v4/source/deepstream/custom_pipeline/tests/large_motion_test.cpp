#include "large_motion_policy.hpp"
#include "gated_event_policy.hpp"
#include <cstdlib>
#include <iostream>
using namespace jiankong::custom_pipeline;
using Motion = LargeMotionPolicy;
void check(bool ok,const char* s) { if(!ok){std::cerr<<s<<'\n';std::exit(1);} }
int main() {
    for(double fps:{8.,10.}) {
        Motion slow,gentle,noisy_gradual,gradual,accelerating,large,jump,sway,shoulder_only;
        bool detected=false;
        bool gentle_detected=false;
        bool noisy_gradual_detected=false;
        bool gradual_detected=false;
        bool accelerating_detected=false;
        for(int i=0;i<4*fps;++i) {
            double t=i/fps,x=t*14;
            check(!slow.update(t,x,100,x,80,200,true),"slow movement must remain eligible for reporting");
            gentle_detected|=gentle.update(t,t*20,100,t*20,80,200,true);
            const double noisy_y=100+(i%2 ? 1.4 : -1.4);
            noisy_gradual_detected|=noisy_gradual.update(
                t,t*20,noisy_y,t*20,noisy_y-20,200,true);
            gradual_detected|=gradual.update(t,t*28,100,t*28,80,200,true);
            if (t <= 3.0) {
                const double accelerating_x=2.5*t*t*t;
                accelerating_detected|=accelerating.update(
                    t,accelerating_x,100,accelerating_x,80,200,true);
            }
            detected|=large.update(t,t*120,100,t*120,80,200,true);
            double j=i<10?0:100;
            check(!jump.update(t,j,100,j,80,200,true),"one sudden jump must not count as continuous movement");
            check(!jump.candidate(),"one sudden jump must not start a motion confirmation window");
            double a=20*std::sin(t*6);
            check(!sway.update(t,a,100,a,80,200,true),"bounded posture sway must not suppress");
            check(!shoulder_only.update(t,0,100,t*120,80,200,true),"shoulder-only rotation must not suppress");
        }
        check(gentle_detected,
              "continuous gentle translation must be suppressed after enough accumulated distance");
        check(noisy_gradual_detected,
              "long-window translation must tolerate realistic detector jitter");
        check(gradual_detected,
              "continuous moderate-speed translation must be suppressed over the long window");
        check(accelerating_detected,
              "clear long-window acceleration must not require equal movement in both halves");
        check(detected,"large sustained translation must be detected without hips");
        for(int i=int(4*fps);i<6*fps;++i)
            large.update(i/fps,(4-1/fps)*120,100,(4-1/fps)*120,80,200,true);
        check(!large.moving(),"stopping must release movement filter");
    }
    Motion posture;
    Motion moderate;
    for(int i=0;i<30;++i) {
        double x=std::min(i,10)*8.;
        check(!moderate.update(i*.1,x,100,x,80,200,true),
              "moderate 40-percent reposition should not be treated as large movement");
    }
    GatedEventPolicy gate;
    bool alarm=false;
    for(int i=0;i<35;++i) {
        double x=std::min(i,9)*4.;
        bool m=posture.update(i*.1,x,100,x,80,200,true);
        check(!m,"brief small reposition must not reset evidence");
        GatedFrameEvidence e;
        e.phone_valid=e.high_confidence_phone=e.screen_ray_hit=true;
        e.person_association=e.hand_relation=e.screen_intent=EvidenceState::true_value;
        e.person_moving=m; alarm|=gate.update(e).alarm;
    }
    check(alarm,"small reposition must preserve regular alarm accumulation");
    // Regression for event 202609090517: strong phone evidence reached the
    // alarm threshold about 0.3 seconds before the same track crossed the
    // conservative large-motion threshold.  Its one-second trajectory already
    // showed coherent translation, so publication must wait briefly.
    Motion cumulative_motion;
    GatedEventPolicy cumulative_motion_gate;
    cumulative_motion_gate.set_infer_fps(10.0);
    bool saw_candidate_before_full_motion = false;
    for (int i = 0; i <= 15; ++i) {
        const double x = i <= 12 ? i * 6.0 : 72.0 + (i - 12) * 20.0;
        const bool moving = cumulative_motion.update(
            i * 0.1, x, 100.0, x, 80.0, 200.0, true);
        GatedFrameEvidence e;
        e.phone_valid = e.high_confidence_phone = e.screen_ray_hit = true;
        e.person_association = e.hand_relation = e.screen_intent =
            EvidenceState::true_value;
        e.person_moving = moving;
        e.person_motion_candidate = cumulative_motion.candidate();
        saw_candidate_before_full_motion |= e.person_motion_candidate && !moving;
        check(!cumulative_motion_gate.update(e).alarm,
              "cumulative motion trend must wait for full-motion confirmation");
    }
    check(saw_candidate_before_full_motion,
          "regression trajectory must expose a pre-confirmation motion trend");
    check(cumulative_motion.moving(),
          "cumulative regression trajectory must eventually confirm large motion");
    Motion invalid;
    for(int i=0;i<30;++i) invalid.update(i*.1,i*12,100,i*12,80,200,true);
    check(invalid.moving(),"known large movement expected before invalid samples");
    check(!invalid.update(4,0,0,0,0,0,false),"long pose loss must expire old movement");
    check(!invalid.update(4.1,0,0,0,0,200,true),"returning track must collect fresh movement evidence");
    std::cout<<"large motion tests passed\n";
}
