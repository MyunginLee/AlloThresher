#pragma once

// C++ STD library
#include <chrono>
#include <iostream>
#include <string>
#include <vector>
// no; let the std:: stand out
// using namespace std;

#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"

//
#include "al/math/al_Random.hpp"  // rnd::uniform()

namespace diy {

// NOTES
// XXX don't use unsigned unless you really need to. See
// https://jacobegner.blogspot.com/2019/11/unsigned-integers-are-dangerous.html

// xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
// xx GLOBALS
// xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

const int SAMPLE_RATE = 48000;  // 44100;
const int BLOCK_SIZE = 512;
const int OUTPUT_CHANNELS = 2;
const int INPUT_CHANNELS = 2;

// How do UGens learn the sample rate?
//
// Lots of UGens need to know the sample rate, which might/could change from
// time to time. How do we let this UGens know what the sample rate is? We make
// each of them inherit from a class that "listens" for changes. How? The base
// class, in its constructor, puts itself on a global list of listeners. We
// set/change the sample rate by telling it to the global list, which in turn
// notifies each of the listeners.
//
// See Publish/Subscribe or Observer pattern
//

#define DEFAULT_PLAYBACK_RATE (48000)

// "Observer" class forward declaration
//
struct PlaybackRateObserver {
  double playbackRate{DEFAULT_PLAYBACK_RATE};
  PlaybackRateObserver* nextObserver{nullptr};
  virtual void onPlaybackRateChange(double playbackRate);
  PlaybackRateObserver();
  // virtual ~PlaybackRateObserver();
};

// "Subject" class forward declaration
//
class PlaybackRateSubject {
  double playbackRate{DEFAULT_PLAYBACK_RATE};
  PlaybackRateObserver* list{nullptr};

  PlaybackRateSubject() {}

 public:
  PlaybackRateSubject(PlaybackRateSubject const&) = delete;
  void operator=(PlaybackRateSubject const&) = delete;

  void notifyObservers(double playbackRate);
  void addObserver(PlaybackRateObserver* observer);
  static PlaybackRateSubject& instance();
};

// The one and only way to get an instance of PlaybackRateSubject
PlaybackRateSubject& PlaybackRateSubject::instance() {
  static PlaybackRateSubject instance;
  return instance;
}

// convenience function for setting playback rate
void setPlaybackRate(double playbackRate) {
  PlaybackRateSubject::instance().notifyObservers(playbackRate);
}

// the default action for PlaybackRateObserver
void PlaybackRateObserver::onPlaybackRateChange(double playbackRate) {
  this->playbackRate = playbackRate;
}

// When a PlaybackRateObserver is born, it adds itself to the list of observers
PlaybackRateObserver::PlaybackRateObserver() {
  PlaybackRateSubject::instance().addObserver(this);
}

void PlaybackRateSubject::notifyObservers(double playbackRate) {
  if (playbackRate != this->playbackRate) {
    for (auto* o = list; o != nullptr; o = o->nextObserver) {
      o->onPlaybackRateChange(playbackRate);
    }
  }
}

void PlaybackRateSubject::addObserver(PlaybackRateObserver* observer) {
  // this is an add-to-front operation on a linked list!
  observer->nextObserver = list;
  list = observer;
};

// xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
// xx HELPER FUNCTIONS
// xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

inline float map(float value, float low, float high, float Low, float High) {
  return Low + (High - Low) * ((value - low) / (high - low));
}
inline float norm(float value, float low, float high) {
  return (value - low) / (high - low);
}
inline float lerp(float a, float b, float t) { return (1.0f - t) * a + t * b; }
inline float mtof(float m) { return 8.175799f * powf(2.0f, m / 12.0f); }
inline float ftom(float f) { return 12.0f * log2f(f / 8.175799f); }
inline float dbtoa(float db) { return 1.0f * powf(10.0f, db / 20.0f); }
inline float atodb(float a) { return 20.0f * log10f(a / 1.0f); }
inline float sigmoid(float x) { return 2.0f / (1.0f + expf(-x)) - 1.0f; }

template <typename F>
inline F wrap(F value, F high = 1, F low = 0) {
  assert(high > low);
  if (value >= high) {
    F range = high - low;
    value -= range;
    if (value >= high) {
      // less common case; must use division
      value -= (F)(unsigned)((value - low) / range);
    }
  } else if (value < low) {
    F range = high - low;
    value += range;
    if (value < low) {
      // less common case; must use division
      value += (F)(unsigned)((high - value) / range);
    }
  }
  return value;
}

float saw(float phase) { return phase * 2 - 1; }
float rect(float phase) { return phase < 0.5 ? -1 : 1; }
float tri(float phase) {
  float f = 2 * phase - 1;
  f = (f < -0.5) ? -1 - f : (f > 0.5 ? 1 - f : f);
  return 2 * f;
}

// TODO: make tic/toc for a more familiar/direct measurement
//
struct BlockTimer {
  std::chrono::high_resolution_clock::time_point begin;
  BlockTimer() : begin(std::chrono::high_resolution_clock::now()) {}
  ~BlockTimer() {
    double t = std::chrono::duration<double>(
                   std::chrono::high_resolution_clock::now() - begin)
                   .count();
    if (t > 0) std::cout << "...took " << t << " seconds." << std::endl;
  }
};

// xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
// xx SYNTHS!
// xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

// this will get used all over the place
//
struct Phasor : PlaybackRateObserver {
  float value{0};
  float increment{0};

  void frequency(float hertz) { increment = hertz / playbackRate; }
  void modulate(float hertz) { increment += hertz / playbackRate; }
  void period(float seconds) { frequency(1.0f / seconds); }
  float frequency() const { return increment * playbackRate; }
  void zero() { value = increment = 0.0f; }

  // we assume this method called once per sample to produce a upward ramp
  // waveform strictly on the interval [0.0, 1.0).
  //
  float operator()() {
    float v = value;  // we return the current value

    // side effect; compute the new value for use next time
    value += increment;
    value = wrap(value);

    return v;
  }
};

// for the eventual name change....
//
// [[deprecated("Use Ramp instead.")]] typedef Timer Phasor;
//

// the partials roll off very quickly when compared to naiive Saw and
// Square; do we really need a band-limited triangle? Meh.
struct Tri : Phasor {
  float operator()() { return tri(Phasor::operator()()); }
};

// a DC-blocking (high-pass) filter
//
struct DCblock {
  float x1 = 0, y1 = 0;
  float operator()(float in1) {
    float y = in1 - x1 + y1 * 0.9997;
    x1 = in1;
    y1 = y;
    return y;
  }
};

// one sample delay; to mimic Max's [gen~ ] object
// not really used...
struct History {
  float _value = 0;
  float operator()(float value) {
    float returnValue = _value;
    _value = value;
    return returnValue;
  }
};

class Biquad : PlaybackRateObserver {
  // Audio EQ Cookbook
  // http://www.musicdsp.org/files/Audio-EQ-Cookbook.txt

  // x[n-1], x[n-2], y[n-1], y[n-2]
  float x1 = 0, x2 = 0, y1 = 0, y2 = 0;

  // filter coefficients
  float b0 = 1, b1 = 0, b2 = 0, a1 = 0, a2 = 0;

 public:
  float operator()(float x0) {
    // Direct Form 1, normalized...
    float y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
    y2 = y1;
    y1 = y0;
    x2 = x1;
    x1 = x0;
    return y0;
  }

  void normalize(float a0) {
    b0 /= a0;
    b1 /= a0;
    b2 /= a0;
    a1 /= a0;
    a2 /= a0;
    // print();
  }

  void print() {
    printf("b0:%f ", b0);
    printf("b1:%f ", b1);
    printf("b2:%f ", b2);
    printf("a1:%f ", a1);
    printf("a2:%f ", a2);
    printf("\n");
  }

  void lpf(float f0, float Q) {
    float w0 = 2 * M_PI * f0 / playbackRate;
    float alpha = sin(w0) / (2 * Q);
    b0 = (1 - cos(w0)) / 2;
    b1 = 1 - cos(w0);
    b2 = (1 - cos(w0)) / 2;
    float a0 = 1 + alpha;
    a1 = -2 * cos(w0);
    a2 = 1 - alpha;

    normalize(a0);
  }

  void hpf(float f0, float Q) {
    float w0 = 2 * M_PI * f0 / playbackRate;
    float alpha = sin(w0) / (2 * Q);
    b0 = (1 + cos(w0)) / 2;
    b1 = -(1 + cos(w0));
    b2 = (1 + cos(w0)) / 2;
    float a0 = 1 + alpha;
    a1 = -2 * cos(w0);
    a2 = 1 - alpha;

    normalize(a0);
  }

  void bpf(float f0, float Q) {
    float w0 = 2 * M_PI * f0 / playbackRate;
    float alpha = sin(w0) / (2 * Q);
    b0 = Q * alpha;
    b1 = 0;
    b2 = -Q * alpha;
    float a0 = 1 + alpha;
    a1 = -2 * cos(w0);
    a2 = 1 - alpha;

    normalize(a0);
  }

  void notch(float f0, float Q) {
    float w0 = 2 * M_PI * f0 / playbackRate;
    float alpha = sin(w0) / (2 * Q);
    b0 = 1;
    b1 = -2 * cos(w0);
    b2 = 1;
    float a0 = 1 + alpha;
    a1 = -2 * cos(w0);
    a2 = 1 - alpha;

    normalize(a0);
  }

  void apf(float f0, float Q) {
    float w0 = 2 * M_PI * f0 / playbackRate;
    float alpha = sin(w0) / (2 * Q);
    b0 = 1 - alpha;
    b1 = -2 * cos(w0);
    b2 = 1 + alpha;
    float a0 = 1 + alpha;
    a1 = -2 * cos(w0);
    a2 = 1 - alpha;

    normalize(a0);
  }
};

struct Timer : PlaybackRateObserver {
  float phase = 0.0;        // on the interval [0, 1)
  float increment = 0.001;  // led to an low F

  void frequency(float hertz) { increment = hertz / playbackRate; }
  void period(float seconds) { frequency(1 / seconds); }

  bool operator()() {
    phase += increment;
    if (phase > 1) {
      phase -= 1;
      return true;
    }
    return false;
  }
};

[[deprecated("Use Timer instead.")]] typedef Timer Edge;

struct OnePole : PlaybackRateObserver {
  float b0 = 1, a1 = 0, yn1 = 0;
  void frequency(float hertz) {
    a1 = exp(-2.0f * 3.14159265358979323846f * hertz / playbackRate);
    b0 = 1.0f - a1;
  }
  void period(float seconds) { frequency(1 / seconds); }
  float operator()(float xn) { return yn1 = b0 * xn + a1 * yn1; }
};

struct Buffer : std::vector<float> {
  int sampleRate{SAMPLE_RATE};

  void operator()(float f) {
    push_back(f);
    //
  }
  void save(const std::string& fileName) const { save(fileName.c_str()); }
  void save(const char* fileName) const {
    drwav_data_format format;
    format.container = drwav_container_riff;
    format.format = DR_WAVE_FORMAT_IEEE_FLOAT;
    format.channels = 1;
    format.sampleRate = sampleRate;
    format.bitsPerSample = 32;

    drwav wav;
    if (!drwav_init_file_write(&wav, fileName, &format, nullptr)) {
      std::cerr << "filed to init file " << fileName << std::endl;
      return;
    }
    drwav_uint64 framesWritten = drwav_write_pcm_frames(&wav, size(), data());
    if (framesWritten != size()) {
      std::cerr << "failed to write all samples to " << fileName << std::endl;
    }
    drwav_uninit(&wav);
  }

  bool load(const std::string& fileName) { return load(fileName.c_str()); }
  bool load(const char* filePath) {
    unsigned int channels;
    unsigned int sampleRate;
    drwav_uint64 totalPCMFrameCount;
    float* pSampleData = drwav_open_file_and_read_pcm_frames_f32(
        filePath, &channels, &sampleRate, &totalPCMFrameCount, NULL);
    if (pSampleData == NULL) {
      printf("failed to load %s\n", filePath);
      return false;
    }

    //
    if (channels == 1)
      for (int i = 0; i < totalPCMFrameCount; i++) {
        push_back(pSampleData[i]);
      }
    else if (channels == 2) {
      for (int i = 0; i < totalPCMFrameCount; i++) {
        push_back((pSampleData[2 * i] + pSampleData[2 * i + 1]) / 2);
      }
    } else {
      printf("can't handle %d channels\n", channels);
      return false;
    }

    drwav_free(pSampleData, NULL);
    return true;
  }

  float get(float index) const {
    // correct indices that are out or range i.e., outside the interval [0.0,
    // size) in such a way that the buffer data seems to repeat forever to the
    // left and forever to the right.
    //
    index = wrap(index, (float)size());
    assert(index >= 0.0f);
    assert(index < size());

    const int i =
        floor(index);  // XXX modf would give us i and t; faster or slower?
    const int j = i == (size() - 1) ? 0 : i + 1;
    // const unsigned j = (i + 1) % size();
    // XXX is the ternary op (?:) faster or slower than the mod op (%) ?
    assert(j < size());

    const float x0 = std::vector<float>::operator[](i);
    const float x1 = std::vector<float>::operator[](j);  // loop around
    const float t = index - i;
    return x1 * t + x0 * (1 - t);
  }

  float operator[](const float index) const { return get(index); }
  float phasor(float index) const { return get(size() * index); }

  void add(float index, const float value) {
    index = wrap(index, (float)size());
    assert(index >= 0.0f);
    assert(index < size());

    const int i = floor(index);
    const int j = (i == (size() - 1)) ? 0 : i + 1;
    const float t = index - i;
    std::vector<float>::operator[](i) += value * (1 - t);
    std::vector<float>::operator[](j) += value * t;
  }
};

[[deprecated("Use Buffer instead.")]] typedef Buffer Array;

struct FloatPair {
  float left, right;
  float& operator[](int i) {
    if (i == 0)
      return left;
    else
      return right;
  }
};

/*
struct StereoArray : std::vector<FloatPair> {
  void save(const char* fileName) const {
    drwav_data_format format;
    format.channels = 2;
    format.container = drwav_container_riff;
    format.format = DR_WAVE_FORMAT_IEEE_FLOAT;
    format.sampleRate = 44100;
    format.bitsPerSample = 32;
    drwav* pWav = drwav_open_file_write(fileName, &format);
    if (pWav == nullptr) {
      std::cerr << "failed to write " << fileName << std::endl;
      drwav_close(pWav);
      return;
    }
    drwav_uint64 samplesWritten = drwav_write(pWav, size(), data());
    if (samplesWritten != size()) {
      std::cerr << "failed to write all samples to " << fileName <<
std::endl; drwav_close(pWav); return;
    }
    drwav_close(pWav);
  }

  bool load(const char* fileName) {
    drwav* pWav = drwav_open_file(fileName);
    if (pWav == nullptr) return false;

    if (pWav->channels == 1) {
      printf("ERROR: Use Array; %s is mono\n", fileName);
    }

    // XXX
    // check the format. make sure it is float

    if (pWav->channels == 2) {
      float* pSampleData = (float*)malloc((size_t)pWav->totalPCMFrameCount *
                                          pWav->channels * sizeof(float));
      drwav_read_f32(pWav, pWav->totalPCMFrameCount, pSampleData);
      drwav_close(pWav);

      resize(pWav->totalPCMFrameCount);
      for (unsigned i = 0; i < size(); ++i) {
        at(i).left = pSampleData[pWav->channels * i];
        at(i).right = pSampleData[pWav->channels * i + 1];
      }
      return true;
    }

    return false;
  }

  void operator()(float l, float r) {
    push_back({l, r});
    //
  }
};
*/

float sine(float phase) {
  struct SineBuffer : Buffer {
    SineBuffer() {
      resize(100000);
      for (unsigned i = 0; i < size(); ++i)  //
        at(i) = sin(i * M_PI * 2 / size());
    }
    float operator()(float phase) { return phasor(phase); }
  };

  static SineBuffer instance;
  return instance(phase);
}

// Taylor Series expansion of sine on the interval [-1, 1). Using Horner's
// method to minimize the number of operations.
template <class F>
inline F sine7(F x) {
  F xx = x * x;
  return x * (F(3.138982) +
              xx * (F(-5.133625) + xx * (F(2.428288) - xx * F(0.433645))));
}

struct ModulatedSine : PlaybackRateObserver {
  float value{0};
  float increment{0};

  void frequency(float hertz) { increment = hertz / playbackRate; }

  float operator()(float modulation = 0.0f) {
    float v = value;
    // XXX is there a way to use different units for modulation to get rid of
    // the divide in the expression below?
    value = wrap(value + increment + modulation / playbackRate, -1.0, 1.0);
    return sine7(v);
  }
};

struct Sine : Phasor {
  float operator()() { return sine(Phasor::operator()()); }
};

// This is a low-level structure for saving a sequence of samples. It offers
// separate read/write operations.
//
struct DelayLine : Buffer {
  unsigned next{0};

  DelayLine(float capacity = 1 /* seconds */) {
    resize(ceil(capacity * SAMPLE_RATE), 0);
  }

  void write(float f) {
    at(next) = f;
    next++;
    if (next == size())  //
      next = 0;
  }
  void operator()(float f) {  // alias
    write(f);
  }

  // z-operator is in samples
  //
  float z(float delay) {
    float index = next - delay;
    if (index < 0)  //
      index += size();
    return get(index);
  }

  // integer based lookup
  float z(int delay) {
    int index = next - delay;
    if (index < 0)  //
      index += size();
    return at(index);
  }

  float read(float delayTime /* seconds */) {
    return z(delayTime * SAMPLE_RATE);
  }
};

struct DelayModulation {
  Sine sine;
  DelayLine delayLine;
  float amplitude;

  void set(float rate, float depth) {
    sine.frequency(rate);
    amplitude = depth;
  }

  float operator()(float f) {
    // XXX should i do the write before the read?
    // this 0.001 fudge factor should use SCIENCE
    //
    float d = 0.001 + (1 + sine()) * amplitude;
    float v = delayLine.read(d);
    delayLine.write(f);
    return v;
  }
};

// This is a simple echo effect. On each call, you give it some input and
// you get some output from delayTime samples ago. If you want "multi-tap"
// or separate read/write operations, use another object.
//
struct Echo : DelayLine {
  float delayTime{0};  // in samples, floating point

  void period(float seconds) { delayTime = seconds * SAMPLE_RATE; }
  void frequency(float hertz) { period(1 / hertz); }

  float operator()(float sample) {
    float returnValue = read(delayTime / SAMPLE_RATE);
    write(sample);
    return returnValue;
  }
};

// How is this different than a one-pole filter?
//
struct EnvelopeFilter {
  float tau{1};
  float value{0};
  void set(float t) {
    // where did we get this from? (see Pirkle 2013 p429)
    // why does it work?
    // what's up with log(0.01)
    tau = pow(2.718281828459045, log(0.01) / (SAMPLE_RATE * t));
  }
  float operator()(float f) { return value = f + (value - f) * tau; }
};

struct EnvelopeFilterAsymmetric {
  float attack{1};
  float release{1};
  float value{0};
  void set(float a, float r) {
    attack = pow(2.718281828459045, log(0.01) / (SAMPLE_RATE * a));
    release = pow(2.718281828459045, log(0.01) / (SAMPLE_RATE * a));
  }
  float operator()(float f) {
    return value = f + (value - f) * ((f > value) ? attack : release);
  }
};

struct ShortTermRootMeanSquared {
  float squaredSum{0};
  float alpha{0.5f};

  void set(float a) { alpha = a; }

  float operator()(float f) {
    // https://en.wikipedia.org/wiki/Moving_average#Exponential%20moving%20average
    squaredSum = alpha * (f * f) + (1 - alpha) * squaredSum;
    return sqrt(squaredSum);
  }
};

struct ShortTermPeakExpensive {
  std::vector<float> data;
  unsigned index{0};

  void set(float seconds) {
    data.clear();
    data.resize(int(seconds * SAMPLE_RATE) + 1);
  }

  float operator()(float f) {
    data[index] = abs(f);
    index++;
    if (index == data.size())  //
      index = 0;
    float maximum = 0;
    for (float p : data)
      if (p > maximum)  //
        maximum = p;
    return maximum;
  }
};

// ???
struct ShortTermPeak {
  std::vector<float> data;
  unsigned index{0};
  float maximum{0};

  void set(float seconds) {
    data.clear();
    data.resize(int(seconds * SAMPLE_RATE) + 1);
  }

  float operator()(float f) {
    data[index] = abs(f);
    if (data[index] > maximum)  //
      maximum = data[index];
    float returnValue = maximum;
    index++;
    if (index == data.size()) {
      index = 0;
      maximum = 0;
    }
    return returnValue;
  }
};

struct Line : PlaybackRateObserver {
  float value = 0, target = 0, seconds = 1, increment = 0;

  void set() {
    // slope per sample
    increment = (target - value) / (seconds * playbackRate);
  }
  void set(float v, float t, float s) {
    value = v;
    target = t;
    seconds = s;
    set();
  }
  void set(float t, float s) {
    target = t;
    seconds = s;
    set();
  }
  void set(float t) {
    target = t;
    set();
  }

  bool done() { return value == target; }

  float operator()() {
    if (value != target) {
      value += increment;
      if ((increment < 0) ? (value < target) : (value > target)) value = target;
    }
    return value;
  }
};

struct ExpSeg {
  Line line;
  void set(double v, double t, double s) { line.set(log2(v), log2(t), s); }
  void set(double t, double s) { line.set(log2(t), s); }
  void set(double t) { line.set(log2(t)); }
  float operator()() { return pow(2.0f, line()); }
};

/*
struct ExpSeg {
  // WARNING:
  // - value > 0 and target > 0
  // - precision error, especially on long segments
  double value = 0, target = 0, seconds = 1 / SAMPLE_RATE, rate = 1;

  void set() {
    rate = pow(2.0, log2(target / value) / (seconds * SAMPLE_RATE));
  }
  void set(double v, double t, double s) {
    value = v;
    target = t;
    seconds = s;
    set();
  }
  void set(double t, double s) {
    target = t;
    seconds = s;
    set();
  }
  void set(double t) {
    target = t;
    set();
  }

  bool done() { return value == target; }

  float operator()() {
    float v = value;
    if (value != target) {
      value *= rate;
      if ((rate < 1) ? (value < target) : (value > target)) value = target;
    }
    return v;
  }
};
*/

struct AttackDecay {
  Line attack, decay;

  void set(float riseTime, float fallTime, float peakValue) {
    attack.set(0, peakValue, riseTime);
    decay.set(peakValue, 0, fallTime);
  }

  float operator()() {
    if (!attack.done()) return attack();
    return decay();
  }
};

struct ADSR {
  Line attack, decay, release;
  int state = 0;

  void set(float a, float d, float s, float r) {
    attack.set(0, 1, a);
    decay.set(1, s, d);
    release.set(s, 0, r);
  }

  void on() {
    attack.value = 0;
    decay.value = 1;
    state = 1;
  }

  void off() {
    release.value = decay.target;
    state = 3;
  }

  float operator()() {
    switch (state) {
      default:
      case 0:
        return 0;
      case 1:
        if (!attack.done()) return attack();
        if (!decay.done()) return decay();
        state = 2;
      case 2:  // sustaining...
        return decay.target;
      case 3:
        return release();
    }
  }
  void print() {
    printf("  state:%d\n", state);
    printf("  attack:%f\n", attack.seconds);
    printf("  decay:%f\n", decay.seconds);
    printf("  sustain:%f\n", decay.target);
    printf("  release:%f\n", release.seconds);
  }
};

struct Table : Phasor, Buffer {
  Table(unsigned size = 4096) { resize(size); }

  virtual float operator()() {
    const float index = value * size();
    const float v = get(index);
    Phasor::operator()();
    return v;
  }
};

struct SoundPlayer : Phasor, Buffer {
  void rate(float ratio) { period((size() / playbackRate) / ratio); }

  virtual float operator()() {
    const float index = value * size();
    const float v = get(index);
    Phasor::operator()();
    return v;
  }
};

float noise(float phase) {
  struct NoiseBuffer : Buffer {
    NoiseBuffer() {
      resize(20 * 44100);
      for (unsigned i = 0; i < size(); ++i)  //
        at(i) = al::rnd::uniformS();
    }
    float operator()(float phase) {
      //
      return phasor(phase);
    }
  };

  static NoiseBuffer instance;
  return instance(phase);
}

float normal(float phase) {
  struct NormalDistribution : Buffer {
    NormalDistribution() {
      resize(20 * 44100);
      for (unsigned i = 0; i < size(); ++i)  //
        at(i) = al::rnd::normal();
      float maximum = 0;
      for (unsigned i = 0; i < size(); ++i)
        if (abs(at(i)) > maximum)  //
          maximum = abs(at(i));
      for (unsigned i = 0; i < size(); ++i)  //
        at(i) /= maximum;
    }
    float operator()(float phase) {
      //
      return phasor(phase);
    }
  };
  //
  static NormalDistribution instance;
  return instance(phase);
}

struct MeanFilter {
  float x1{0};
  float operator()(float x0) {
    float v = (x0 + x1) / 2;
    x1 = x0;
    return v;
  }
};

struct PluckedString : DelayLine, PlaybackRateObserver {
  MeanFilter filter;

  float gain{1};
  float t60{1};
  float delayTime{1};  // in seconds

  void frequency(float hertz) { period(1 / hertz); }
  void period(float seconds) {
    delayTime = seconds;
    recalculate();
  }

  void decayTime(float _t60) {
    t60 = _t60;
    recalculate();
  }

  void set(float frequency, float decayTime) {
    delayTime = 1 / frequency;
    t60 = decayTime;
    recalculate();
  }

  // given t60 and frequency (seconds and Hertz), calculate the gain...
  //
  // for a given frequency, our algorithm applies *gain* frequency-many
  // times per second. given a t60 time we can calculate how many times (n)
  // gain will be applied in those t60 seconds. we want to reduce the signal
  // by 60dB over t60 seconds or over n-many applications. this means that
  // we want gain to be a number that, when multiplied by itself n times,
  // becomes 60 dB quieter than it began.
  //
  void recalculate() {
    int n = t60 / delayTime;
    gain = pow(dbtoa(-60), 1.0f / n);
    // printf("t:%f\tf:%f\tn:%d\tg:%f\n", t60, 1 / delayTime, n, gain);
    // fflush(stdout);
  }

  float operator()() {
    float v = filter(read(delayTime)) * gain;
    write(v);
    return v;
  }

  void pluck(float gain = 1) {
    // put noise in the last N sample memory positions. N depends on
    // frequency
    //
    int n = int(ceil(delayTime * playbackRate));
    for (int i = 0; i < n; ++i) {
      int index = next - i;
      if (index < 0)  //
        index += size();
      at(index) = gain * noise(float(i) / n);
    }
  }
};

struct QuasiBandlimited {
  //
  // from "Synthesis of Quasi-Bandlimited Analog Waveforms Using Frequency
  // Modulation" by Peter Schoffhauzer
  // (http://scp.web.elte.hu/papers/synthesis1.pdf)
  //
  const float a0 = 2.5;   // precalculated coeffs
  const float a1 = -1.5;  // for HF compensation

  // variables
  float osc;      // output of the saw oscillator
  float osc2;     // output of the saw oscillator 2
  float phase;    // phase accumulator
  float w;        // normalized frequency
  float scaling;  // scaling amount
  float DC;       // DC compensation
  float norm;     // normalization amount
  float last;     // delay for the HF filter

  float Frequency, Filter, PulseWidth;

  QuasiBandlimited() {
    reset();
    Frequency = 1.0;
    Filter = 0.85;
    PulseWidth = 0.5;
    recalculate();
  }

  void reset() {
    // zero oscillator and phase
    osc = 0.0;
    osc2 = 0.0;
    phase = 0.0;
  }

  void recalculate() {
    w = Frequency / SAMPLE_RATE;  // normalized frequency
    float n = 0.5 - w;
    scaling = Filter * 13.0f * powf(n, 4.0f);  // calculate scaling
    DC = 0.376 - w * 0.752;                    // calculate DC compensation
    norm = 1.0 - 2.0 * w;                      // calculate normalization
  }

  void frequency(float f) {
    Frequency = f;
    recalculate();
  }

  void filter(float f) {
    Filter = f;
    recalculate();
  }

  void pulseWidth(float w) {
    PulseWidth = w;
    recalculate();
  }

  void step() {
    // increment accumulator
    phase += 2.0 * w;
    if (phase >= 1.0) phase -= 2.0;
    if (phase <= -1.0) phase += 2.0;
  }

  // process loop for creating a bandlimited saw wave
  float saw() {
    step();

    // calculate next sample
    osc = (osc + sinf(2 * M_PI * (phase + osc * scaling))) * 0.5;
    // compensate HF rolloff
    float out = a0 * osc + a1 * last;
    last = osc;
    out = out + DC;     // compensate DC offset
    return out * norm;  // store normalized result
  }

  // process loop for creating a bandlimited PWM pulse
  float pulse() {
    step();

    // calculate saw1
    osc = (osc + sinf(2 * M_PI * (phase + osc * scaling))) * 0.5;
    // calculate saw2
    osc2 =
        (osc2 + sinf(2 * M_PI * (phase + osc2 * scaling + PulseWidth))) * 0.5;
    float out = osc - osc2;  // subtract two saw waves
    // compensate HF rolloff
    out = a0 * out + a1 * last;
    last = osc;
    return out * norm;  // store normalized result
  }
};

struct Saw : QuasiBandlimited {
  float operator()() { return saw(); }
};

struct Rect : QuasiBandlimited {
  float operator()() { return pulse(); }
};

}  // namespace diy
   //#endif  // __240C_SYNTHS__
