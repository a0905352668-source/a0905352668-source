#include "person_byte_tracker.hpp"
#include <iostream>
#include <limits>
#include <stdexcept>
#include <chrono>
using jiankong::custom_pipeline::PersonByteTracker;
static Object box(float x,float score=.9f) { return {cv::Rect_<float>(x,0,40,100),0,score}; }
static void check(bool ok,const char* why) { if(!ok) throw std::runtime_error(why); }
int main() {
    PersonByteTracker tracker;
    auto a=tracker.update({box(0),box(100)});
    check(a.size()==2,"strong detections must produce assignments");
    const int first=a[0].track_id, second=a[1].track_id;
    a=tracker.update({box(100),box(0)});
    check(a.size()==2 && a[0].track_id==second && a[1].track_id==first,"assignment must preserve exact input provenance after reorder");
    check(tracker.update({box(0,.15f)}).empty(),"low scores may maintain identity but must not be exported to business evidence");
    a=tracker.update({box(0)});
    check(a.size()==1 && a[0].track_id==first,"low confidence bridge must preserve identity");
    check(tracker.update({}).empty(),"predicted tracks must not be visible evidence");
    a=tracker.update({box(0)});
    check(a.size()==1 && a[0].track_id==first,"short occlusion should recover identity");
    check(tracker.update({box(std::numeric_limits<float>::quiet_NaN())}).empty(),"invalid boxes must be discarded");
    PersonByteTracker malformed;
    auto giant=box(0);giant.rect.width=1e30f;giant.rect.height=1e30f;
    check(malformed.update({giant}).empty(),"overflowing geometry must not enter tracker");
    check(malformed.update({box(1e20f)}).empty(),"numerically collapsed geometry must not enter tracker");
    PersonByteTracker crossing;
    int left=-1,right=-1;
    for(int frame=0;frame<41;++frame) {
        Object p=box(frame*5.f), q=box(200-frame*5.f); q.rect.y=15;
        const bool reverse=frame%2;
        auto m=crossing.update(reverse ? std::vector<Object>{q,p} : std::vector<Object>{p,q});
        check(m.size()==2,"crossing tracks must remain visible");
        if(frame==0) { left=m[0].track_id;right=m[1].track_id; }
        check(m[reverse?1:0].track_id==left && m[reverse?0:1].track_id==right,
              "velocity crossing must preserve identities despite detection reordering");
    }
    for(int count: {10,50,100}) {
        PersonByteTracker bench;
        std::vector<double> ms;
        for(int frame=0;frame<500;++frame) {
            std::vector<Object> input;
            for(int i=0;i<count;++i) { auto p=box((i%10)*100.f+(frame%3)); p.rect.y=(i/10)*150.f;input.push_back(p); }
            const auto begin=std::chrono::steady_clock::now();
            auto matches=bench.update(input);
            const double elapsed=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
            check(matches.size()==static_cast<size_t>(count),"load test lost tracks");
            if(frame>10) ms.push_back(elapsed);
        }
        std::sort(ms.begin(),ms.end());
        std::cout<<count<<" persons: CPU milliseconds p50="<<ms[ms.size()/2]<<" p95="<<ms[ms.size()*95/100]<<'\n';
    }
    std::cout<<"Person ByteTrack adapter tests passed\n";
}
