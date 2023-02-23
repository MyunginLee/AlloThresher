#include "al/app/al_App.hpp"
#include "al/ui/al_ControlGUI.hpp"
#include "al/ui/al_Parameter.hpp"
#include "al/ui/al_PresetHandler.hpp"
#include "al/ui/al_PresetServer.hpp"
#include "al/sound/al_Reverb.hpp"

#include "synths.h"
#include "Gamma/DFT.h"
#include "configuration.h"

#include <forward_list>
#include <string>
#include <unordered_set>
#include <vector>

using namespace al;
using namespace ben;
using namespace diy;
using namespace std;
#define FFT_SIZE 4048

template <typename T>
class GrainManager {
  forward_list<T*> remove, inactive;
  unordered_set<T*> active;

 public:
  int activeGrainCount() { return active.size(); }

  // add a grain to the container with "inactive" status.
  //
  void insert_inactive(T& t) { inactive.push_front(&t); }

  // are there any inactive grains?
  //
  bool has_inactive() { return !inactive.empty(); }

  // find the first inactive grain, make it active, and return it. you better
  // have called has_inactive() before you call this!
  //
  T& get_next_inactive() {
    T* t = inactive.front();
    active.insert(t);
    inactive.pop_front();
    return *t;
  }

  // run a given function on each active grain.
  //
  void for_each_active(function<void(T& t)> f) {
    for (auto& t : active) f(*t);
  }

  // schedule an active grain for deactivation.
  //
  void schedule_for_deactivation(T& t) { remove.push_front(&t); }

  // deactivate all grains scheduled for deactivation.
  //
  void execute_deactivation() {
    for (auto e : remove) {
      active.erase(e);
      inactive.push_front(e);
    }
    remove.clear();
  }
};

const char* show_classification(float x) {
  switch (std::fpclassify(x)) {
    case FP_INFINITE:
      return "Inf";
    case FP_NAN:
      return "NaN";
    case FP_NORMAL:
      return "normal";
    case FP_SUBNORMAL:
      return "subnormal";
    case FP_ZERO:
      return "zero";
    default:
      return "unknown";
  }
}

bool bad(float x) {
  switch (std::fpclassify(x)) {
    case FP_INFINITE:
    case FP_NAN:
    case FP_SUBNORMAL:
      return true;

    case FP_ZERO:
    case FP_NORMAL:
    default:
      return false;
  }
}

struct Granulator {
  vector<diy::Array> arrayList;
  Vec3f cell_acc, cell_grv;
  Quatf cell_rot;
  Vec3f aa, ao;
  float acc_abs;
  int gest_command;
  Mesh spectrum_acc;
  std::vector<Colori> pixel;
  // knows how to load a file into the granulator
  //
  void load(string fileName) {
    arrayList.emplace_back();
    if (arrayList.back().load(fileName)) {
      printf("Loaded %s! at %08X with size %lu\n", fileName.c_str(),
             &arrayList.back(), arrayList.back().size());
    } else {
      exit(1);
    }
  }

  // we define a Grain...
  //
  struct Grain {
    diy::Array* source = nullptr;
    Line index;  // this is like a tape play head that scrubs through the source
    AttackDecay envelop;  // new class handles the fade in/out and amplitude
    float pan;

    float operator()() { return source->get(index()) * envelop(); }
  };

  // we store a "pool" of grains which may or may not be active at any time
  //
  vector<Grain> grain;
  GrainManager<Grain> manager;

  Granulator() {
    // rather than using new/delete and allocating memory on the fly, we just
    // allocate as many grains as we might need---a fixed number that we think
    // will be enough. we can find this number through trial and error. if
    // too many grains are active, we may take too long in the audio callback
    // and that will cause drop-outs and glitches.
    //
    grain.resize(1000);
    for (auto& g : grain) manager.insert_inactive(g);
  }

  // gui tweakable parameters
  //
  ParameterInt whichClip{"/clip", "", 0, "", 0, 17};
  Parameter grainDuration{"/duration", "", 0.25, "", 0.001, 3.0};
  Parameter startPosition{"/position", "", 0.25, "", 0.0, 1.0};
  Parameter peakPosition{"/envelope", "", 0.1, "", 0.0, 1.0};
  Parameter amplitudePeak{"/amplitude", "", 0.707, "", 0.0, 1.0};
  Parameter panPosition{"/pan", "", 0, "", -1., 1.};
  Parameter playbackRate{"/playback", "", 0.0, "", -1.5, 1};
  Parameter birthRate{"/frequency", "", 20, "", 0, 1000};

  // this oscillator governs the rate at which grains are created
  //
  Edge grainBirth;

  // this method makes a new grain out of a dead / inactive one.
  //
  void reincarnate(Grain& g) {
    whichClip = gest_command;
    // choose which sound clip this grain pulls from
    g.source = &arrayList[whichClip];

    // // map gestural data to input
    // grainDuration = acc_abs * 0.5; 
    grainDuration = acc_abs * 2.; 
    // TODO . match with android ao
    // startPosition = (cell_grv.y+1)/2;
    // playbackRate = cell_grv.x;
    // peakPosition = (cell_grv.z+1)/2;
    // amplitudePeak = acc_abs * 2;
    
    // Android match
    startPosition = (ao.z +180 ) / 360;// 0~1
    // cout << startPosition << endl;
    playbackRate = cell_grv.x * 1.5;
    // cout << playbackRate << endl;
    peakPosition = (ao.z +180 ) / 360;
    amplitudePeak = acc_abs * 2;

    // cout << grainDuration << startPosition << playbackRate << endl;

    // startTime and endTime are in units of sample
    float startTime = g.source->size() * startPosition;
    float endTime =
        startTime + grainDuration * SAMPLE_RATE * powf(2.0, playbackRate);

    g.index.set(startTime, endTime, grainDuration);

    // riseTime and fallTime are in units of second
    float riseTime = grainDuration * peakPosition;
    float fallTime = grainDuration - riseTime;
    g.envelop.set(riseTime, fallTime, amplitudePeak);

    g.pan = panPosition;
  }

  // make the next sample
  //
  diy::FloatPair operator()() {
    // figure out if we should generate (reincarnate) more grains; then do so.
    //
    birthRate = 10 + tan(acc_abs);
  //  birthRate = 10 + 10*(-cell_grv.y+1);
   grainBirth.frequency(birthRate);
    //  cout << "    "  << birthRate << endl;
    if (grainBirth()) {
      // we want to birth a new grain
      if (manager.has_inactive()) {
        // we have a grain to reincarnate
        reincarnate(manager.get_next_inactive());
      }
    }

    // figure out which grains are active. for each active grain, get the next
    // sample; sum all these up and return that sum.
    //
    float left = 0, right = 0;
    manager.for_each_active([&](Grain& g) {
      float f = g();
      left += f * (1 - g.pan)* (acc_abs*5);
      right += f * g.pan *(acc_abs*5);
      if (g.index.done()) {
        manager.schedule_for_deactivation(g);
      }
    });
    manager.execute_deactivation();

    return {left, right};
  }
};
