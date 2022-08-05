// Ben Myung-in Lee. 2022.07.30. AlloThresher ver.0.220730
/*  Copyright 2022 [Myungin Lee]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
   
	This program code contains creative work of instrument "AlloThresher."
  */
 
#include "Gamma/SoundFile.h"
#include "Gamma/DFT.h"
using namespace gam;

#include "al/core/app/al_App.hpp"
#include "al/core.hpp"
#include "al/core/graphics/al_Shapes.hpp"
#include "al/util/ui/al_ControlGUI.hpp"
#include "al/util/ui/al_Parameter.hpp"
#include "al/util/ui/al_Preset.hpp"
#include <iostream>
#include <string>
using namespace al;

#include "configuration.h"
using namespace ben;
#include "synths.h"
using namespace diy;

#include <forward_list>
#include <unordered_set>
#include <vector>
using namespace std;

STFT stft_acc, stft_sound;

struct FloatPair {
  float l, r;
};

template <typename T>
class Bag {
  forward_list<T*> remove, inactive;
  unordered_set<T*> active;

 public:
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

struct Granulator {
  // knows how to load a file into the granulator
  //
  void load(string fileName) {
    SearchPaths searchPaths;
    searchPaths.addSearchPath("..");

    string filePath = searchPaths.find(fileName).filepath();
    SoundFile soundFile;
    soundFile.path(filePath);
    if (!soundFile.openRead()) {
      cout << "We could not read " << fileName << "!" << endl;
      exit(1);
    }
    if (soundFile.channels() != 1) {
      cout << fileName << " is not a mono file" << endl;
      exit(1);
    }

    diy::Array* a = new diy::Array();
    a->size = soundFile.frames();
    a->data = new float[a->size];
    soundFile.read(a->data, a->size);
    this->soundClip.push_back(a);

    soundFile.close();
  }

  // we keep a set of sound clips in memory so grains may use them
  //
  vector<diy::Array*> soundClip;
  Vec3f cell_acc, cell_grv;
  float* grv_block;
  float* acc_block;
  float* power_acc_block;
  float acc_abs;
  Mesh spectrum_acc;
  std::vector<Colori> pixel;

  // we define a Grain...
  //
  struct Grain {
    diy::Array* source = nullptr;
    Line index;  // this is like a tape play head that scrubs through the source
    AttackDecay envelop;  // new class handles the fade in/out and amplitude
    float pan;

    float operator()() {
      // the next sample from the grain is taken from the source buffer
      return envelop() * source->get(index());
    }
  };

  // we store a "pool" of grains which may or may not be active at any time
  //
  vector<Grain> grain;
  Bag<Grain> bag;

  Granulator() {
    grain.resize(1000);
    for (auto& g : grain) bag.insert_inactive(g);
  }

  // gui tweakable parameters
  //
  Parameter whichClip{"/clip", "", 0, "", 0, 9};\
  Parameter grainDuration{"/duration", "", 0.25, "", 0.001, 1.0};
  Parameter startPosition{"/position", "", 0.25, "", 0.0, 1.0};
  Parameter peakPosition{"/envelope", "", 0.1, "", 0.0, 1.0};
  Parameter amplitudePeak{"/amplitude", "", 0.707, "", 0.0, 1.0};
  Parameter panPosition{"/pan", "", 0.5, "", 0.0, 1.0};
  Parameter playbackRate{"/playback", "", 0.0, "", -1.0, 1.0};
  Parameter birthRate{"/frequency", "", 55, "", 0, 200};
  // this oscillator governs the rate at which grains are created
  //
  Edge grainBirth;
  // this method makes a new grain out of a dead / inactive one.
  //
  int column = 0;
  float switch_thres = 1;
  void recycle(Grain& g) {
    //  Do STFT of grv and acc for each axis.
    // Setting STFT window size and hopsize
    stft_acc.resize(window_size,hopSize);   // winSize, hopSize with 50% overlap
    // STFT 
    stft_acc(power_acc_block[window_size]);
/*    for (unsigned k = 0; k < stft_acc.numBins() - 1; k++) {
        float f = stft_acc.bin(k)[0]*100;
    //    pixel[column + k * N].r = f * 255;
        pixel[column + k * N].b = f * 255;
    //    pixel[column + k * N].g = f * 255;
    }
    column++;
    if (column >= N / 2) column = 0;
*/

/*    // Arragne Spectrum Elements
    for (unsigned k = 0; k < stft_acc.numBins(); k++) {
          // XXX put y-axis in dB
          spectrum_acc.vertices()[k].y = stft_acc.bin(k).mag();
    }
    spectrum_acc.primitive(Mesh::LINE_STRIP);
    for (unsigned k = 0; k < stft_acc.numBins(); k++) {
      // XXX set the x-axis to be log
      spectrum_acc.vertex(float(k) / stft_acc.numBins(), 0);
    }
*/
    // choose which sound clip this grain pulls from
    g.source = soundClip[(int)whichClip];

    // startTime and endTime are in units of sample
    grainDuration = (power_acc_block[window_size] + power_acc_block[window_size-20] + power_acc_block[window_size-40])*2; 
    startPosition = (cell_grv.y+1)/2;
    playbackRate = cell_grv.x;
    peakPosition = (cell_grv.z+1)/2;
    float startTime = g.source->size * startPosition;
    float endTime =
        startTime + grainDuration * SAMPLE_RATE * powf(2.0, playbackRate);
    g.index.set(startTime, endTime, grainDuration);

    // riseTime and fallTime are in units of second
    float riseTime = grainDuration * peakPosition;
    float fallTime = grainDuration - riseTime;
    amplitudePeak = acc_abs * 2;
    g.envelop.set(riseTime, fallTime, amplitudePeak);
    g.pan = panPosition;
  }

  // make the next sample
  //
  FloatPair operator()() {
    // figure out if we should generate (recycle) more grains; then do so.
    //
    birthRate = acc_abs * 100;
    grainBirth.frequency(birthRate);
    if (grainBirth())
      if (bag.has_inactive()) recycle(bag.get_next_inactive());

    // figure out which grains are active. for each active grain, get the next
    // sample; sum all these up and return that sum.
    //
    float left = 0, right = 0;
    bag.for_each_active([&](Grain& g) {
      float f = g();
      left += f * (1 - g.pan);
      right += f * g.pan;
      if (g.index.done()) bag.schedule_for_deactivation(g);
    });
    bag.execute_deactivation();

    return {left, right};
  }
};

struct MyApp : App {
  float background = 0.0;
  Granulator granulator;
  ControlGUI gui;
  PresetHandler presetHandler{"GranulatorPresets"};
  PresetServer presetServer{"0.0.0.0", 9011};
	osc::Recv server;
  ostringstream oss;
  string file_num;
  int NObjects;
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
  Mesh pointMesh;

  int click, gest_command;
  Quatf rot;
  void onCreate() override {
    lens().near(0.1).far(100).fovy(90); // lens view angle, how far

    // Block of acc and grv. for 3 axis x,y,z    
    grv_block = new (nothrow) float[window_size*3]; 
    acc_block = new (nothrow) float[window_size*3];
    power_acc_block = new (nothrow) float[window_size];
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
    texture.create2D(N / 2, N / 2, Texture::RGB8);
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
    // OSC 
 		server.open(4444, "", 0.05);
		server.handler(*this);
		server.start();
    NObjects = 10;
    for (int k = 0; k < NObjects; k++){
      oss << k << ".wav";
      file_num = oss.str();      
    // load sound files into the
      granulator.load(file_num);
      oss.str("");
      oss.clear();
    }

    gui.init();
    gui << presetHandler  //
        << granulator.whichClip << granulator.grainDuration
        << granulator.startPosition << granulator.peakPosition
        << granulator.amplitudePeak << granulator.panPosition
        << granulator.playbackRate << granulator.birthRate;

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
  }

	void onMessage(osc::Message& m) override {
    //	m.print();
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

    //Shift and save the values in the block. Those values are for future works
    for (int i=1; i<window_size; i++){
      power_acc_block[window_size+i-1] = power_acc_block[window_size+i];
      for (int j=0; j<3; j++){
        acc_block[window_size*j+i-1] = acc_block[window_size*j+i];
        grv_block[window_size*j+i-1] = grv_block[window_size*j+i];
      }
    }

    // New values in grv_block, acc_block, power_acc_block
    granulator.cell_acc = cell_acc;
    granulator.cell_grv = cell_grv;   
    granulator.acc_block = acc_block;
    granulator.grv_block = grv_block;
    granulator.power_acc_block = power_acc_block;
    granulator.acc_abs = acc_abs;
	}

  float shader_phase = 1;
  float halfSize = 0;
  void onAnimate(double dt) override {
//    navControl().active(!gui.usingInput());
    // cout << cell_grv << endl;
    shader_phase = 4+acc_abs*7;
    halfSize = 0.2 * shader_phase / 3;
    nav().pos(0, 0.0, 5);
    nav().quat(Quatd(1.000000, 0.000000, 0.000000, 0.000000));
  }

  void onDraw(Graphics& g) override {
//    texture.submit(granulator.pixel);
    g.clear(background);
/*  g.blending(true);
    g.blendModeTrans();
    g.quadViewport(texture);*/

    // Draw Controller Shader
    g.depthTesting(false);
    g.blending(true);
    g.blendModeTrans();

  //  pointMesh.vertex(Vec3f(cell_grv.z*500, cell_grv.x*500, cell_grv.y*10) * CLOUD_WIDTH);
    shade_texture.bind();
    g.pushMatrix();
    g.translate(rot.x * 2, rot.y * 2, rot.z * 0.1);
    // pointMesh.color(abs(cell_grv.x)*100, abs(cell_grv.y)*100, abs(cell_grv.z)*100);
    pointMesh.color(HSV(acc_abs*100,1+al::rnd::uniform(),1+al::rnd::uniform()));

    g.scale(acc_abs*10+1);
    g.shader(shader);
    g.shader().uniform("halfSize", halfSize < 0.05 ? 0.05 : halfSize);
    g.draw(pointMesh);
    g.popMatrix();
    shade_texture.unbind();
    // Draw Waveform
    g.pushMatrix();
    g.translate(0,5,0);
    // g.color(abs(cell_grv.x)*5+al::rnd::uniform(), abs(cell_grv.y)*20+al::rnd::uniform(), abs(cell_grv.z)*10+al::rnd::uniform());
    g.color(HSV(acc_abs*200+al::rnd::uniform(acc_abs),1+0.1*al::rnd::uniform(),1+0.1*al::rnd::uniform()));
    g.rotate(90, Vec3f(0,0,1)); 
    g.rotate(cell_acc.x*100, Vec3f(rot.x,0,0));
    g.rotate(cell_acc.y*100, Vec3f(0,rot.y,0));
    g.rotate(cell_acc.z*100, Vec3f(0,0,rot.z));

    g.scale(10,200,1);
    g.pointSize(acc_abs*10);
    g.draw(waveform);
    g.pushMatrix();
    gui.draw(g);
  }

  void onSound(AudioIOData& io) override {
    while (io()) {
      FloatPair p = granulator();
      waveform.vertices()[cursor].y = (p.l + p.r)/4;
      cursor++;
      if (cursor == waveform.vertices().size()) cursor = 0;

      io.out(0) = p.l*2;
      io.out(1) = p.r*2;
    }
  }
};

int main() {
  MyApp app;
  app.initAudio(SAMPLE_RATE, BLOCK_SIZE, OUTPUT_CHANNELS, INPUT_CHANNELS);
  app.start();
}
