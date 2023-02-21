#include "configuration.h"
#include "granula.cpp"
using namespace al;
using namespace gam;
using namespace ben;
using namespace diy;
static bool fullscreen = true;

using namespace std;
#define FFT_SIZE 4048
struct CommonState
{

};
struct AgentCommonData
{

};
struct AndroidSynth : public PositionedVoice
{
    // Unit generators
    float mNoiseMix;
    gam::Pan<> mPan;
    gam::ADSR<> mAmpEnv;
    gam::EnvFollow<> mEnvFollow; // envelope follower to connect audio output to graphics
    gam::DSF<> mOsc;
    gam::NoiseWhite<> mNoise;
    gam::Reson<> mRes;
    gam::Env<2> mCFEnv;
    gam::Env<2> mBWEnv;
    Reverb<float> reverb;
    float amp, freq, att, rels, sus, noise, cf1,cf2, cfrise, bw1, bw2, bwrise,
          hmnum, hmamp, rev;
    gam::STFT stft = gam::STFT(FFT_SIZE, FFT_SIZE / 4, 0, gam::HANN, gam::MAG_FREQ);
    Mesh mSpectrogram;
    vector<float> spectrum;
    AgentCommonData *commonData;

  void init() override
  {
    commonData = static_cast<AgentCommonData *>(userData());
    mAmpEnv.curve(0);               // linear segments
    mAmpEnv.levels(0, 1.0, 1.0, 0); // These tables are not normalized, so scale to 0.3
    mAmpEnv.sustainPoint(2);        // Make point 2 sustain until a release is issued
    mCFEnv.curve(0);
    mBWEnv.curve(0);
    mOsc.harmonics(12);
    amp = 0;
    freq = 1000;
    att = 0.1;
    rels = 3.;
    sus = 0.8;
    noise = 0.8;
    cf1 = 400;
    cf2 = 400;
    cfrise = 0.5;
    bw1 = 700;
    bw2 = 900;
    bwrise = 0.5;
    hmnum = 12;
    hmamp = 1;
    rev = 0.7;
// amp, freq, att, rels, sus, curv, noise, cf1,cf2, cfrise, bw1, bw2, bwrise,
//           hmnum, hmamp, reverb;
  }
  void update(double dt) override
  {
    noise = 0.8;
  }
  void onProcess(Graphics &g) override
  {
    mSpectrogram.reset();
    mSpectrogram.primitive(Mesh::POINTS);
    for (int i = 0; i < FFT_SIZE / 2; i++)
    {
      mSpectrogram.color(HSV(0.5 - spectrum[i] * 100,100,100));
      mSpectrogram.vertex(i, spectrum[i], 0.0);
    }
    g.meshColor(); // Use the color in the mesh
    g.pushMatrix();
    g.pointSize(10);
    g.translate(0, 0, 0);
    g.scale(10.0 / FFT_SIZE, 1000, 1.0);
    g.draw(mSpectrogram);
    g.popMatrix();

  }
void onProcess(AudioIOData &io) override
  {
    while (io())
    {
        // // mix oscillator with noise
        // float s1 = mOsc() * (1 - noise) + mNoise() * noise;

        // // apply resonant filter
        // mRes.set(mCFEnv(), mBWEnv());
        // s1 = mRes(s1);
        // s1 = mOsc();

        // // appy amplitude envelope
        // // s1 *= mAmpEnv() * amp;
        // float wet1, wet2;
        // reverb(s1, wet1, wet2);

        // mEnvFollow(wet1);
        // mPan(wet1, wet1, wet2);
			  // if(stft(wet1)){
        //   for (unsigned k = 0; k < stft.numBins(); ++k)
        //   {
        //       // Here we simply scale the complex sample
        //       spectrum[k] = tanh(pow(stft.bin(k).real(), 1.3));
        //   }

        //   io.out(0) += wet1;
        //   io.out(1) += wet2;
        // }
    }
  }
  void onTriggerOn() override
  {    
    cout << "triggered on" << endl;
  }
};

struct MyApp : DistributedAppWithState<CommonState>
{
  float background = 0.;
  Granulator granulator;
  DistributedScene scene;
  vector<AndroidSynth *> andsynth;
  AgentCommonData agentCommon;

  // AndroidSynth synth;
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
  gam::Pan<> mPan;

  // Right hand : iPhone
  Vec3f cell_acc, cell_grv;
  Quatf cell_rot;
  float acc_abs, android_acc_abs, filter_coeff;
  float *grv_block;
  float *acc_block;
  float *power_acc_block;
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
  int gest_command;
  Vec3f imag, amag;
  Vec3f aa, ao;
  float imag_power, amag_power, cross_angle_mean_square;
  gam::Biquad<> mFilter{};
  Reverb<float> reverb;
  int fbh, fbw;
  float fb_idx; // frame blur width, height
  void PlatformSetupSize()
  {
    int total_width, total_height;
    al::sphere::getFullscreenDimension(&total_width, &total_height);
    std::cout << total_width << ", " << total_height << std::endl;
    dimensions(0, 0, total_width, total_height);
  }
  void onInit() override
  {
    if (al_get_hostname() == "moxi" || fullscreen)
    {
      PlatformSetupSize();
    }
    registerDynamicScene(scene);
    scene.setDefaultUserData(&this->agentCommon);
    scene.registerSynthClass<AndroidSynth>("androidsynth");


    auto newAndroid = scene.allocateVoice<AndroidSynth>();
    andsynth.push_back(newAndroid);
    scene.al::PolySynth::triggerOn(newAndroid,0,100);
    spectrum.resize(FFT_SIZE / 2 + 1);
    mFilter.zero();
    reverb.bandwidth(0.6f); // Low-pass amount on input, in [0,1]
    reverb.damping(0.5f);   // High-frequency damping, in [0,1]
    reverb.decay(0.6f);     // Tail decay factor, in [0,1]

    // Diffusion amounts
    // Values near 0.7 are recommended. Moving further away from 0.7 will lead
    // to more distinct echoes.
    reverb.diffusion(0.76, 0.666, 0.707, 0.571);
    audioIO().print();
  }

  void onCreate() override
  {
    lens().near(0.1).far(100).fovy(90); // lens view angle, how far
    texBlur.filter(Texture::LINEAR);

    // Shader
    shade_texture.create2D(256, 256, Texture::R8, Texture::RED, Texture::SHORT);
    int Nx = shade_texture.width();
    int Ny = shade_texture.height();
    std::vector<short> alpha;
    alpha.resize(Nx * Ny);
    for (int j = 0; j < Ny; ++j)
    {
      float y = float(j) / (Ny - 1) * 2 - 1;
      for (int i = 0; i < Nx; ++i)
      {
        float x = float(i) / (Nx - 1) * 2 - 1;
        float m = exp(-13 * (x * x + y * y));
        m *= pow(2, 15) - 1; // scale by the largest positive short int
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
    pointMesh.vertex(Vec3f(0, 0, 0) * 300 * CLOUD_WIDTH);
    // pointMesh.color(HSV(0.66, 1.0, 1.0));

    // prepare Waveform
    texture.create2D(N / 2, N / 2, Texture::RGB8);
    fb_idx = 1;
    fbw = fbWidth()*fb_idx;
    fbh = fbHeight()*fb_idx;
    texBlur.resize(fbw, fbh);

    int Mx = texture.width();
    int My = texture.height();
    // waveform.primitive(Mesh::POINTS);
    waveform.primitive(Mesh::LINE_STRIP);
    for (int i = 0; i < 10000; i++)
      waveform.vertex(i / 10000.0);

    // prepare vector for pixel data
    granulator.pixel.resize(Mx * My);

    for (int j = 0; j < My; ++j)
    {
      for (int i = 0; i < Mx; ++i)
      {
        Color c = RGB(0.12);
        granulator.pixel[j * Mx + i] = c;
      }
    }

    // load sound files into the
    granulator.load("source/0_dub.wav");
    // granulator.load("source/1_oingd.wav");
    granulator.load("source/1_laugh.wav");
    // granulator.load("source/2_nidea.wav");
    granulator.load("source/2_dontcare.wav");
    granulator.load("source/3_sponge.wav");
    granulator.load("source/4_click.wav");
    granulator.load("source/5_jazz.wav");
    granulator.load("source/6_beatbox.wav");
    granulator.load("source/7_oingd.wav");
    // granulator.load("source/7_pew.wav");
    granulator.load("source/8_harpsi.wav");
    granulator.load("source/9_violin.wav");
    granulator.load("source/10_lux.wav");
    granulator.load("source/11_emile.wav");
    granulator.load("source/12_mong.wav");
    granulator.load("source/13_atz.wav");
    // granulator.load("source/11_drugs.wav");
    granulator.load("source/14_kor.wav");
    granulator.load("source/15_sanjo.wav");
    // granulator.load("source/16_jazz.wav");
    // granulator.load("source/17_jazz.wav");
    gui.init();
    /*
    gui.addr(presetHandler,  //
             granulator.whichClip, granulator.grainDuration,
             granulator.startPosition, granulator.peakPosition,
             granulator.amplitudePeak, granulator.panPosition,
             granulator.playbackRate, granulator.birthRate);
            */
    gui << presetHandler //
        << granulator.whichClip << granulator.grainDuration
        << granulator.startPosition << granulator.peakPosition
        << granulator.amplitudePeak << granulator.panPosition
        << granulator.playbackRate << granulator.birthRate << active;

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
    server.open(4444, "0.0.0.0", 0.05);
    server.handler(oscDomain()->handler());
    server.start();
  }
  float shader_phase = 1;
  float halfSize = 0;
  void onAnimate(double dt) override
  {
    navControl().active(!gui.usingInput());
    shader_phase = 4 + acc_abs * 7;
    halfSize = 0.2 * shader_phase / 3;
    nav().pos(0.1 * cell_acc.x, 0.1 * cell_acc.y, 3 - acc_abs);
    nav().quat(Quatd(1.000000, 0.000000, 0.000000, 0.000000));
    // printf("%d %d\n", audioIO().isOpen(), audioIO().isRunning());
    //

    // granular source command is determined by the android angle,, **
    // cout << ao.x << " " <<     int( (ao.x+180) / 36)<< endl; 
    gest_command = int( (ao.x+180) / 23);
    // cout << gest_command;
    // Power of acceleration.
    acc_abs = cbrt(cell_acc.x * cell_acc.x + cell_acc.y * cell_acc.y + cell_acc.z * cell_acc.z) * 0.1;
    android_acc_abs = cbrt(aa.x*aa.x + aa.y*aa.y + aa.z*aa.z) * 0.02;
    // cout << acc_abs << "   " << android_acc_abs  << "   " << android_acc_abs / acc_abs<< endl;  
     // New values in grv_block, acc_block, power_acc_block
    // cout << android_acc_abs << endl; // Print android acc 
    granulator.cell_acc = cell_acc;
    granulator.cell_grv = cell_grv;
    granulator.cell_rot = cell_rot;
    // cout << cell_rot.x << endl;
    granulator.panPosition = -0.05*cell_rot.z;
    granulator.acc_abs = acc_abs;
    granulator.gest_command = gest_command;
    granulator.ao = ao;
    // Filter realtime
    filter_coeff = 100+android_acc_abs * 1000;
    mFilter.freq(filter_coeff);
    mFilter.res(filter_coeff); 
    mFilter.type(LOW_PASS);
    mFilter.zero();
    // reverb realtime
    // reverb.bandwidth(0.9f); // Low-pass amount on input, in [0,1]
    // reverb.damping(android_acc_abs);   // High-frequency damping, in [0,1]
    reverb.decay(0.1*acc_abs+(-ao.y)/90);     // Tail decay factor, in [0,1]
    // cout << ao.x << "  " << ao.y << "  "  << ao.z << endl;
    // reverb.damping(0.1f+0.8*(ao.y+180)/360);   // High-frequency damping, in [0,1]
    scene.update(dt);

    // Diffusion amounts
    // Values near 0.7 are recommended. Moving further away from 0.7 will lead
    // to more distinct echoes.
    // reverb.diffusion(0.5 + android_acc_abs, 0.566+ android_acc_abs, 0.707, 0.571);
    // reverb.diffusion(0.76, 0.666, 0.707, 0.571);

    // cross_angle_mean_square = (rot - ao / 180).mag();
    // cout << cross_angle_mean_square << endl;
    // ao / 180 
    // cout << rot.x << " " << rot.y << " " << rot.z << "   " << ao/180 << endl;
    // Magnet test
    // amag_power = amag.mag();
    // imag_power = imag.mag();
    // cout << amag_power + imag_power<< "= " << amag_power << "   " << imag_power << endl;
  }

  void onDraw(Graphics &g) override
  {
    // background = 0.1*acc_abs;
    background = 0.0;
    g.clear(background);
    // scene.render(g);
    fbw = fbWidth()*fb_idx;
    fbh = fbHeight()*fb_idx;
    texBlur.resize(fbw, fbh); // index = 5
    // g.tint(0.98 - 0.1 * acc_abs);
    // g.tint(0.88 + 0.05 * acc_abs); // proper ?

    // g.tint(0.93 + 0.05 * acc_abs); // proper ?
    g.tint(0.91 + 0.01 * acc_abs); // proper ?


    // g.quadViewport(texBlur, -1.005, -1.005, 2.01, 2.01); // Outward
    // g.quadViewport(texBlur, -1. - android_acc_abs*0.1, -1.- android_acc_abs*0.1
    //               , 2 + android_acc_abs*0.2, 2 + android_acc_abs*0.2); // Outward. good straight!
    float direction = -ao.y / 9000;
    float bnf = aa.magSqr() * direction + 0.1f+0.8*(ao.y+180)/100;

    lens().near(0.1).far(1000).fovy(90+ 10 * bnf); // lens view angle, how far
    // cout << bnf << endl;
    g.quadViewport(texBlur, -1. - bnf*0.1, 
                            -1. - bnf*0.1
                  , 2*fb_idx + bnf*0.2*fb_idx, 
                  2*fb_idx + bnf*0.2*fb_idx); // Outward. back and fowards!

    // g.quadViewport(texBlur, -0.995, -0.995, 1.99, 1.99); // Inward
    // g.quadViewport(texBlur, -1.005, -1.00, 2.01, 2.0);   // Oblate
    // g.quadViewport(texBlur, -1.005, -0.995, 2.01, 1.99); // Squeeze
    // g.quadViewport(texBlur, -1, -1, 2, 2);               // non-transformed
    g.tint(1); // set tint back to 1
    // shade_texture.bind();

    // Draw Controller Shader
    g.depthTesting(false);
    g.blending(true);
    g.blendTrans();
    g.pushMatrix();
    g.translate(cell_rot.x * 2, cell_rot.y * 2, cell_rot.z * 0.1);
    // pointMesh.color(abs(cell_grv.x)*100, abs(cell_grv.y)*100, abs(cell_grv.z)*100);
    // pointMesh.color(HSV(acc_abs * 100, 1 + al::rnd::uniform(), 1 + al::rnd::uniform()));
    mSpectrogram.reset();
    // mSpectrogram.primitive(Mesh::LINE_STRIP);
    mSpectrogram.primitive(Mesh::POINTS);

    g.scale(acc_abs * 10 + 1);
    g.shader(shader);
    g.shader().uniform("halfSize", 0.05);
    // g.draw(pointMesh);
    g.popMatrix();
    // shade_texture.unbind();
    // Draw Waveform
    g.pushMatrix();
    // g.translate(0, 0, 0);
    // g.color(abs(cell_grv.x)*5+al::rnd::uniform(), abs(cell_grv.y)*20+al::rnd::uniform(), abs(cell_grv.z)*10+al::rnd::uniform());
    // g.color(HSV(acc_abs * (gest_command)*0.2 + 0.1* (gest_command) + 0.1 * al::rnd::uniform(acc_abs),0.7+ al::rnd::uniform(acc_abs),0.9+0.1*al::rnd::uniform(acc_abs)));
// vivid!
    // g.color(HSV(0.1 * (gest_command) + 0.1 * al::rnd::uniform(acc_abs), android_acc_abs + al::rnd::uniform(acc_abs), 0.7 + 1 * al::rnd::uniform(acc_abs))); 
    g.color(HSV(0.1 * (gest_command) + 0.1 * al::rnd::uniform(acc_abs), android_acc_abs + al::rnd::uniform(acc_abs), 0.2+android_acc_abs + 1 * al::rnd::uniform(acc_abs)));

    // g.rotate(90, Vec3f(0, 0, 1));
    // g.rotate(cell_acc.x * 100, Vec3f(rot.x, 0, 0));
    // g.rotate(cell_acc.y * 100, Vec3f(0, rot.y, 0));
    // g.rotate(cell_acc.z * 100, mFilter3f(cell_grv.x, 0, 0));
    g.rotate(cell_acc.y * 100, Vec3f(0, cell_grv.y, 0));
    g.rotate(cell_acc.z * 100, Vec3f(0, 0, cell_grv.z));
    g.scale(0.1, 10, 1);
    g.pointSize(acc_abs * 8);
    for (int i = 0; i < FFT_SIZE / 2; i++)
    {
      mSpectrogram.color(HSV(0.5 - spectrum[i] * 100 + al::rnd::uniformS(acc_abs * 100), al::rnd::uniformS(acc_abs)+ spectrum[i] * 1000, 1 + spectrum[i] * 100 + 0.5 * al::rnd::uniformS(acc_abs)));
      // mSpectrogram.vertex(cos(i) *(1 + 10 * cos(spectrum[i])), sin(i) * (1+ 10 * sin(spectrum[i])), 0.0);
      // mSpectrogram.vertex( 10*cos( 0.01 * i )*sin( 0.1 * i*android_acc_abs + cell_acc.y * 100) 
      // , 10*sin(0.01*i+cell_acc.z * 100)*cos(0.1*i*android_acc_abs)*(100* spectrum[i] * (1 + android_acc_abs)), cos( 0.01 * i ) );
      mSpectrogram.vertex( 10*cos( 0.01 * i )*sin( 0.1 * i*android_acc_abs + cell_acc.y * 100) 
      , 10*sin(0.01*i+cell_acc.z * 100)*(100* spectrum[i] * (1 + android_acc_abs)), cos( 0.01 * i ) *100* spectrum[i]);    }
    // cout << android_acc_abs << endl;
    g.draw(mSpectrogram);
    g.popMatrix();
    texBlur.copyFrameBuffer();

    if(showGUI)
      gui.draw(g);
  }

  void onSound(AudioIOData &io) override
  {
    try
    {
      active.set(granulator.manager.activeGrainCount());
      // scene.render(io); 

      while (io())
      {

        diy::FloatPair p = granulator();

        if (cursor == waveform.vertices().size())
          cursor = 0;

        if (bad(p.left))
        {
          printf("p.left is %s\n", show_classification(p.left));
        }

        if (bad(p.right))
        {
          printf("p.right is %s\n", show_classification(p.right));
        }

        value.set(p.left);

        // float fl = mFilter(p.left);
        // float fr = mFilter(p.right);
// no filter
        float fl = (p.left);
        float fr = (p.right);

        float rv_r1, rv_l1, rv_r2, rv_l2 ;
        // reverb(fl , rv_r1, rv_l1);
        rv_r1 = fr;
        rv_l1 = fl;
        reverb(fr , rv_r2, rv_l2);
        // fl = rv_r1 + rv_r2;
        // fr = rv_l1 + rv_l2;
        // io.out(0) = (fl);
        // io.out(1) = (fr);
        // cout << rv_l1<< endl;
        mPan.pos(-granulator.panPosition);
        mPan(rv_l2, rv_l2, rv_r2);
        io.out(0) = (rv_r2);
        io.out(1) = (rv_l2);

        if (stft(rv_r2))
        { // Loop through all the frequency bins
          for (unsigned k = 0; k < stft.numBins(); ++k)
          {
            // Here we simply scale the complex sample
            spectrum[k] = 100 * tanh(pow(stft.bin(k).real(), 1.3));
            // spectrum[k] = stft.bin(k).real();
          }
        }
      }
    }
    catch (const std::out_of_range &e)
    {
      std::cerr << "Out of Range error: " << e.what() << '\n';
    }
  }
  bool onKeyDown(Keyboard const &k) override
  {
    switch (k.key())
    {
    case ' ':
      showGUI = !showGUI;
      break;
    }
    return true;
  }

  void onMessage(osc::Message &m) override
  {
    int k = 0;

    // Check that the address and tags match what we expect
    if (m.addressPattern() == "/gyrosc/grav")
    {
      m >> cell_grv.x;
      m >> cell_grv.y;
      m >> cell_grv.z;
    }
    else if (m.addressPattern() == "/gyrosc/accel")
    {
      m >> cell_acc.x;
      m >> cell_acc.y;
      m >> cell_acc.z;
    }
    else if (m.addressPattern() == string("/gyrosc/gyro"))
    {
      m >> cell_rot.x;
      m >> cell_rot.y;
      m >> cell_rot.z;
      // cout << cell_rot.z << endl;
    }
    else if (m.addressPattern() == string("/gyrosc/button"))
    {
      // m >> gest_command;
    }
    // else if (m.addressPattern() == string("/gyrosc/mag"))
    // {
    //   m >> imag.x;
    //   m >> imag.y;
    //   m >> imag.z;
    // }
    else if (m.addressPattern() == string("/aa/x")) 
    { // android accel
      m >> aa.x;
    }
    else if (m.addressPattern() == string("/aa/y"))
    {
      m >> aa.y;
    }
    else if (m.addressPattern() == string("/aa/z"))
    {
      m >> aa.z;
    }
    else if (m.addressPattern() == string("/ao/x")) 
    { // android accel
      m >> ao.x;
    }
    else if (m.addressPattern() == string("/ao/y"))
    {
      m >> ao.y;
    }
    else if (m.addressPattern() == string("/ao/z"))
    {
      m >> ao.z;
    }
    // else if (m.addressPattern() == string("/am/x")) 
    // { // android accel
    //   m >> amag.x;
    // }
    // else if (m.addressPattern() == string("/am/y"))
    // {
    //   m >> amag.y;
    // }
    // else if (m.addressPattern() == string("/am/z"))
    // {
    //   m >> amag.z;
    // }
  }
};

int main()
{
  MyApp app;
  app.configureAudio(SAMPLE_RATE, BLOCK_SIZE, OUTPUT_CHANNELS);
  app.start();
}
