#include "BYTETracker.h"
#include <cstdlib>
#include <iostream>
static void require(bool ok,const char* message) {
    if(!ok) { std::cerr<<message<<'\n';std::exit(1); }
}
static Object box(float x,float score=0.9f,int label=0) {
    return Object{cv::Rect_<float>(x,0,40,100),label,score};
}
int main() {
    BYTETracker a(10,30), b(10,30);
    auto first=a.update({box(10,.30f,7)});
    require(first.size()==1,"production-confidence person must initialize a track");
    int id=first[0].track_id;
    auto second=a.update({box(12,.15f,9)});
    require(second.size()==1 && second[0].track_id==id,"low-confidence recovery must preserve existing track");
    require(a.update({box(200,.15f)}).empty(),"low-confidence detection must not create a track");
    auto other=b.update({box(10)});
    require(other.size()==1 && other[0].track_id==1,"camera IDs and state must be independent");
    BYTETracker expiry(10,30);
    int old=expiry.update({box(10)})[0].track_id;
    for(int i=0;i<10;++i) require(expiry.update({}).empty(),"lost prediction must not be exported as visible detection");
    auto ret=expiry.update({box(10)});
    if(ret.empty()) ret=expiry.update({box(10)});
    require(ret.size()==1 && ret[0].track_id!=old,"expired track must not reactivate on expiry boundary");
    std::cout<<"ByteTrack tests passed\n";
}
