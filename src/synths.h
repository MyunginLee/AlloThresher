#ifndef __240C_SYNTHS__
#define __240C_SYNTHS__

#include <chrono>
#include <cmath>
#include <cstdio>

// #define die_unless(message, ...)

#define die(message, ...)                                               \
  do {                                                                  \
    fprintf(stderr, "died in %s at line %d with ", __FILE__, __LINE__); \
    fprintf(stderr, message, ##__VA_ARGS__);                            \
    fprintf(stderr, "\n");                                              \
    exit(-1);                                                           \
  } while (0);

#define info(message, ...)                                          \
  do {                                                              \
    fprintf(stderr, "info in %s at line %d: ", __FILE__, __LINE__); \
    fprintf(stderr, message, ##__VA_ARGS__);                        \
    fprintf(stderr, "\n");                                          \
  } while (0);

namespace diy {

const int SAMPLE_RATE = 44100;
const int BLOCK_SIZE = 512;
const int OUTPUT_CHANNELS = 2;
const int INPUT_CHANNELS = 2;

float mtof(float m) { return 8.175799f * powf(2.0f, m / 12.0f); }
float ftom(float f) { return 12.0f * log2f(f / 8.175799f); }
float dbtoa(float db) { return 1.0f * powf(10.0f, db / 20.0f); }
float atodb(float a) { return 20.0f * log10f(a / 1.0f); }

struct Phasor {
  float phase = 0.0;        // on the interval [0, 1)
  float increment = 0.001;  // led to an low F

  void frequency(float hertz) {
    // XXX check for INSANE frequencies
    if (hertz > SAMPLE_RATE) {
      printf("hertz > SAMPLE_RATE\n");
      exit(1);
    }
    if (hertz < -SAMPLE_RATE) {
      printf("hertz < -SAMPLE_RATE\n");
      exit(1);
    }
    increment = hertz / SAMPLE_RATE;
  }

  float operator()() {
    phase += increment;

    // phase wrap.. this only works for frequencies less than the sample rate
    if (phase > 1) phase -= 1;
    if (phase < 0) phase += 1;
    return phase;
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

  // XXX: make this one work...
  float tri() { return 0; }
};

struct Saw : QuasiBandlimited {
  float operator()() { return saw(); }
};

struct SawAlias : Phasor {
  float operator()() { return Phasor::operator()() * 2 - 1; }
};

struct Rect : QuasiBandlimited {
  float operator()() { return pulse(); }
};

struct RectAlias : Phasor {
  float dutyCycle = 0.5;
  float operator()() { return (Phasor::operator()() < dutyCycle) ? -1 : 1; }
};

struct TriAlias : Phasor {
  float operator()() {
    float f = Phasor::operator()();
    return ((f < 0.5) ? f : 1 - f) * 4 - 1;
  }
};

struct DCblock {
  float x1 = 0, y1 = 0;
  float operator()(float in1) {
    float y = in1 - x1 + y1 * 0.9997;
    x1 = in1;
    y1 = y;
    return y;
  }
};

/*
 * this one is not ready
struct Tri : QuasiBandlimited {
  DCblock block, block2;
  float value = 0;
  float operator()() {
    value += block(pulse());
    return block2(value);
  }
};
*/

struct History {
  float _value = 0;
  float operator()(float value) {
    float returnValue = _value;
    _value = value;
    return returnValue;
  }
};

class Biquad {
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
    float w0 = 2 * M_PI * f0 / SAMPLE_RATE;
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
    float w0 = 2 * M_PI * f0 / SAMPLE_RATE;
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
    float w0 = 2 * M_PI * f0 / SAMPLE_RATE;
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
    float w0 = 2 * M_PI * f0 / SAMPLE_RATE;
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
    float w0 = 2 * M_PI * f0 / SAMPLE_RATE;
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

struct Edge {
  float phase = 0.0;        // on the interval [0, 1)
  float increment = 0.001;  // led to an low F

  void frequency(float hertz) { increment = hertz / SAMPLE_RATE; }
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

struct Array {
  float* data = nullptr;
  unsigned size = 0;

  virtual ~Array() {
    printf("Array deleted.\n");
    fflush(stdout);
    if (data) delete[] data;
  }

  // deep-copy copy constructor
  // Array(const Array& other);
  // also do assignment

  float& operator[](unsigned index) { return data[index]; }
  float operator[](const float index) const { return get(index); }

  void resize(unsigned n) {
    size = n;
    if (data) delete[] data;  // or your have a memory leak
    if (n == 0) {
      data = nullptr;
    } else {
      data = new float[n];
      for (unsigned i = 0; i < n; ++i) data[i] = 0.0f;
    }
  }

  float get(float index) const {
    // allow for sloppy indexing (e.g., negative, huge) by fixing the index to
    // within the bounds of the array
    if (index < 0) index += size;  // -21221488559881683402437427200.000000
    if (index > size) index -= size;

    // defer to our method without bounds checking
    return raw(index);
  }

  float raw(const float index) const {
    const unsigned i = floor(index);
    const float x0 = data[i];
    const float x1 = data[(i == (size - 1)) ? 0 : i + 1];  // looping semantics
    const float t = index - i;
    return x1 * t + x0 * (1 - t);
  }

  void add(const float index, const float value) {
    const unsigned i = floor(index);
    const unsigned j = (i == (size - 1)) ? 0 : i + 1;  // looping semantics
    const float t = index - i;
    data[i] += value * (1 - t);
    data[j] += value * t;
  }
};

struct Line {
  float value = 0, target = 0, seconds = 1 / SAMPLE_RATE, increment = 0;

  void set() {
    if (seconds <= 0) seconds = 1 / SAMPLE_RATE;
    // slope per sample
    increment = (target - value) / (seconds * SAMPLE_RATE);
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

/*
// pattern for later
struct Reverb {
  float operator()(float f) { return makeReverbHappenTo(f); }
}
*/

/*
struct Sine : Phasor {
  // how much memory does this cost?
  float data[200000];
  // a float is 4 bytes
  // 200_000 * 4 = 800_000 bytes or 0.8 MB.

  Sine() {
    for (int i = 0; i < 200000; i++) data[i] = sin(2 * M_PI * i / 200000);
  }

  float operator()() {
    float phase = Phasor::operator()();
    return data[int(phase * 200000)];
  }
};
*/

struct Table : Phasor, Array {
  Table(unsigned size = 4096) { resize(size); }

  virtual float operator()() {
    const float index = phase * size;
    const float v = get(index);
    Phasor::operator()();
    return v;
  }
};

struct Noise : Table {
  Noise(unsigned size = 20 * 44100) {
    resize(size);
    for (unsigned i = 0; i < size; ++i) data[i] = rnd::uniformS();
  }
};

struct Normal : Table {
  Normal(unsigned size = 20 * 44100) {
    resize(size);
    for (unsigned i = 0; i < size; ++i) data[i] = rnd::normal();
  }
};

struct Sine : Table {
  Sine(unsigned size = 10000) {
    const float pi2 = M_PI * 2;
    resize(size);
    for (unsigned i = 0; i < size; ++i) data[i] = sinf(i * pi2 / size);
  }
};
//
// RAII
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

}  // namespace diy
#endif  // __240C_SYNTHS__
