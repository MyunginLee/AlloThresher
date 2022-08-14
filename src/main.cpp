#include "al/app/al_App.hpp"
#include "al/ui/al_ControlGUI.hpp"
#include "al/ui/al_Parameter.hpp"
#include "al/ui/al_PresetHandler.hpp"
#include "al/ui/al_PresetServer.hpp"
using namespace al;

#include "synths.h"
#include "Gamma/DFT.h"
#include "configuration.h"
using namespace ben;
using namespace diy;

#include <forward_list>
#include <string>
#include <unordered_set>
#include <vector>
using namespace std;
#define FFT_SIZE 4048

// this special container for grains offers O(1) complexity for
// getting some inactive grain (for recycling) and O(n) complexity
// for deactivating completed grains.
//
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
    Array* source = nullptr;
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
  ParameterInt whichClip{"/clip", "", 0, "", 0, 8};
  Parameter grainDuration{"/duration", "", 0.25, "", 0.001, 1.0};
  Parameter startPosition{"/position", "", 0.25, "", 0.0, 1.0};
  Parameter peakPosition{"/envelope", "", 0.1, "", 0.0, 1.0};
  Parameter amplitudePeak{"/amplitude", "", 0.707, "", 0.0, 1.0};
  Parameter panPosition{"/pan", "", 0.5, "", 0.0, 1.0};
  Parameter playbackRate{"/playback", "", 0.0, "", -1.0, 1.0};
  Parameter birthRate{"/frequency", "", 55, "", 0, 1000};

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
    grainDuration = acc_abs * 5; 

    startPosition = (cell_grv.y+1)/2;
    playbackRate = cell_grv.x;
    peakPosition = (cell_grv.z+1)/2;
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
    birthRate = 10 + acc_abs * 100;
    grainBirth.frequency(birthRate);
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
      left += f * (1 - g.pan);
      right += f * g.pan;
      if (g.index.done()) {
        manager.schedule_for_deactivation(g);
      }
    });
    manager.execute_deactivation();

    return {left, right};
  }
};

struct MyApp : App {
  MyApp() {
    // this is called from the main thread
  }

  float background = 0.;

  Granulator granulator;
  ControlGUI gui;
  PresetHandler presetHandler{"GranulatorPresets"};
  PresetServer presetServer{"0.0.0.0", 9011};
  ParameterInt active{"/active", "", 0, "", 0, 1000};
  Parameter value{"/value", "", 0, "", -1, 1};
  osc::Recv server;
  Mesh mSpectrogram;
  vector<float> spectrum;
  bool showGUI = true;
  bool showSpectro = true;
  bool navi = false;
  gam::STFT stft = gam::STFT(FFT_SIZE, FFT_SIZE / 4, 0, gam::HANN, gam::MAG_FREQ);

  Vec3f cell_acc,  cell_grv;
  Vec3f o, r;
  float acc_abs;
  float* grv_block;
  float* acc_block;
  float* power_acc_block;
  // Waveform
  Texture texture;
  Line amplitude, frequency;
  Mesh waveform;
  unsigned cursor = 0;
  // Shader 
  ShaderProgram shader;
  Texture shade_texture;
  Texture texBlur;

  Mesh pointMesh;
  int click, gest_command;
  Quatf rot;

  void onInit() override {
    spectrum.resize(FFT_SIZE / 2 + 1);
  }

  void onCreate() override {
    lens().near(0.1).far(100).fovy(90); // lens view angle, how far
    texBlur.filter(Texture::LINEAR);

    // Shader
    shade_texture.create2D(256, 256, Texture::R8, Texture::RED, Texture::SHORT);
    int Nx = shade_texture.width();
    int Ny = shade_texture.height();
    std::vector<short> alpha;
    alpha.resize(Nx * Ny);
    for (int j = 0; j < Ny; ++j) {
      float y = float(j) / (Ny - 1) * 2 - 1;
      for (int i = 0; i < Nx; ++i) {
        float x = float(i) / (Nx - 1) * 2 - 1;
        float m = exp(-13 * (x * x + y * y));
        m *= pow(2, 15) - 1;  // scale by the largest positive short int
        alpha[j * Nx + i] = m;
      }
    }
    shade_texture.submit(&alpha[0]);
    // compile and link the three shaders
    //
    shader.compile(vertex, fragment, geometry);
    // create a mesh of point
    pointMesh.primitive(Mesh::POINTS);

//    pointMesh.vertex(Vec3f(cell_grv.z, cell_grv.x, cell_grv.y) *300 * CLOUD_WIDTH);
    pointMesh.vertex(Vec3f(0,0,0) *300 * CLOUD_WIDTH);
    // pointMesh.color(HSV(0.66, 1.0, 1.0));

    // prepare Waveform 
    texture.create2D(N / 2, N / 2, Texture::RGB8);    texBlur.resize(fbWidth(), fbHeight());

    int Mx = texture.width();
    int My = texture.height();
    // waveform.primitive(Mesh::POINTS);
    waveform.primitive(Mesh::LINE_STRIP);
    for (int i = 0; i < 10000; i++) waveform.vertex(i / 10000.0);


    // prepare vector for pixel data
    granulator.pixel.resize(Mx * My);

    for (int j = 0; j < My; ++j) {      
      for (int i = 0; i < Mx; ++i) {
        Color c = RGB(0.12);
        granulator.pixel[j * Mx + i] = c;
      }
    }

    // load sound files into the
    granulator.load("source/0.wav");
    granulator.load("source/1.wav");
    granulator.load("source/2.wav");
    granulator.load("source/3.wav");
    granulator.load("source/4.wav");
    granulator.load("source/5.wav");
    granulator.load("source/6.wav");
    granulator.load("source/7.wav");
    granulator.load("source/8.wav");

    gui.init();
    /*
    gui.addr(presetHandler,  //
             granulator.whichClip, granulator.grainDuration,
             granulator.startPosition, granulator.peakPosition,
             granulator.amplitudePeak, granulator.panPosition,
             granulator.playbackRate, granulator.birthRate);
            */
    gui << presetHandler  //
        << granulator.whichClip << granulator.grainDuration
        << granulator.startPosition << granulator.peakPosition
        << granulator.amplitudePeak << granulator.panPosition
        << granulator.playbackRate << granulator.birthRate << active << value;

    presetHandler << granulator.whichClip << granulator.grainDuration
                  << granulator.startPosition << granulator.peakPosition
                  << granulator.amplitudePeak << granulator.panPosition
                  << granulator.playbackRate << granulator.birthRate;
    presetHandler.setMorphTime(1.0);
    // presetServer << presetHandler;

    parameterServer() << granulator.whichClip << granulator.grainDuration
                      << granulator.startPosition << granulator.peakPosition
                      << granulator.amplitudePeak << granulator.panPosition
                      << granulator.playbackRate << granulator.birthRate;
    parameterServer().print();
    // // OSC comm
    server.open(4444,"0.0.0.0", 0.05);
    server.handler(oscDomain()->handler());
    server.start();
  }
  float shader_phase = 1;
  float halfSize = 0;
  void onAnimate(double dt) override {
    navControl().active(!gui.usingInput());
    shader_phase = 4+acc_abs*7;
    halfSize = 0.2 * shader_phase / 3;
    nav().pos(0, 0.0, 5);
    nav().quat(Quatd(1.000000, 0.000000, 0.000000, 0.000000));
    // printf("%d %d\n", audioIO().isOpen(), audioIO().isRunning());
    //
  }

  void onDraw(Graphics& g) override {
    background = 0.1*acc_abs;
    g.clear(background);
    texBlur.resize(fbWidth(), fbHeight());
    g.tint(0.98);
    g.quadViewport(texBlur, -1.005, -1.005, 2.01, 2.01);  // Outward
    // g.quadViewport(texBlur, -0.995, -0.995, 1.99, 1.99); // Inward
    // g.quadViewport(texBlur, -1.005, -1.00, 2.01, 2.0);   // Oblate
    // g.quadViewport(texBlur, -1.005, -0.995, 2.01, 1.99); // Squeeze
    // g.quadViewport(texBlur, -1, -1, 2, 2);               // non-transformed
    g.tint(1);  // set tint back to 1

    // Draw Controller Shader
    g.depthTesting(false);
    g.blending(true);
    g.blendTrans();
  //  pointMesh.vertex(Vec3f(cell_grv.z*500, cell_grv.x*500, cell_grv.y*10) * CLOUD_WIDTH);
    shade_texture.bind();
    g.pushMatrix();
    g.translate(rot.x * 2, rot.y * 2, rot.z * 0.1);
    // pointMesh.color(abs(cell_grv.x)*100, abs(cell_grv.y)*100, abs(cell_grv.z)*100);
    pointMesh.color(HSV(acc_abs*100,1+al::rnd::uniform(),1+al::rnd::uniform()));
    mSpectrogram.reset();
    mSpectrogram.primitive(Mesh::LINE_STRIP);
    // mSpectrogram.primitive(Mesh::POINTS);

    g.scale(acc_abs*10+1);
    g.shader(shader);
    g.shader().uniform("halfSize", 0.05);
    g.draw(pointMesh);
    g.popMatrix();
    shade_texture.unbind();
    // Draw Waveform
    g.pushMatrix();
    g.translate(0,0,0);
    // g.color(abs(cell_grv.x)*5+al::rnd::uniform(), abs(cell_grv.y)*20+al::rnd::uniform(), abs(cell_grv.z)*10+al::rnd::uniform());
    g.color(HSV(acc_abs*200+al::rnd::uniform(acc_abs),1+0.1*al::rnd::uniform(),1+0.1*al::rnd::uniform()));
    g.rotate(90, Vec3f(0,0,1)); 
    g.rotate(cell_acc.x*100, Vec3f(rot.x,0,0));
    g.rotate(cell_acc.y*100, Vec3f(0,rot.y,0));
    g.rotate(cell_acc.z*100, Vec3f(0,0,rot.z));

    g.scale(0.1,1,1);
    g.pointSize(acc_abs*10);
    for (int i = 0; i < FFT_SIZE / 2; i++)
    {
      mSpectrogram.color(HSV(0.5 - spectrum[i] * 100 + al::rnd::uniformS(acc_abs*100), al::rnd::uniformS(acc_abs), 1 + 0.5 *al::rnd::uniformS(acc_abs) ));
      // mSpectrogram.vertex(cos(i) *(1 + 10 * cos(spectrum[i])), sin(i) * (1+ 10 * sin(spectrum[i])), 0.0);
      mSpectrogram.vertex( i,  spectrum[i], 0.0);

    }
    g.draw(mSpectrogram);
    g.popMatrix();
    texBlur.copyFrameBuffer();

    // gui.draw(g);
  }

  void onSound(AudioIOData& io) override {
    try {
      active.set(granulator.manager.activeGrainCount());

      while (io()) {

        diy::FloatPair p = granulator();

        if (cursor == waveform.vertices().size()) cursor = 0;

        if (bad(p.left)) {
          printf("p.left is %s\n", show_classification(p.left));
        }

        if (bad(p.right)) {
          printf("p.right is %s\n", show_classification(p.right));
        }

        value.set(p.left);

        io.out(0) = p.left;
        io.out(1) = p.right;

        if (stft(p.right))
        { // Loop through all the frequency bins
          for (unsigned k = 0; k < stft.numBins(); ++k)
          {
            // Here we simply scale the complex sample
            spectrum[k] = 100*tanh(pow(stft.bin(k).real(), 1.3));
            // spectrum[k] = stft.bin(k).real();
          }
        }
      }

    } catch (const std::out_of_range& e) {
      std::cerr << "Out of Range error: " << e.what() << '\n';
    }
  }


	void onMessage(osc::Message& m) override {
    	// m.print();
    int k = 0;

		// Check that the address and tags match what we expect
    if (m.addressPattern() == "/gyrosc/grav") {
      m >> o.x;
      m >> o.y;
      m >> o.z;
    } 
    else if (m.addressPattern() == "/gyrosc/accel") {
      m >> r.x;
      m >> r.y;
      m >> r.z;
    }
   if (m.addressPattern() == string("/gyrosc/gyro"))
      {
        m >> rot.x;
        m >> rot.y;
        m >> rot.z;
      }    
    else if (m.addressPattern() == string("/gyrosc/button"))
    {
      m >> click;
      gest_command = click;
    }
    cell_acc = r;
    cell_grv = o;
    // Power of acceleration. 
    acc_abs = cbrt(cell_acc.x * cell_acc.x + cell_acc.y* cell_acc.y + cell_acc.z*cell_acc.z)/10;

    // New values in grv_block, acc_block, power_acc_block
    granulator.cell_acc = cell_acc;
    granulator.cell_grv = cell_grv;   
    granulator.acc_abs = acc_abs;
    granulator.gest_command = gest_command;
	}


};

int main() {
  MyApp app;
  app.configureAudio(SAMPLE_RATE, BLOCK_SIZE, OUTPUT_CHANNELS);
  app.start();
}
