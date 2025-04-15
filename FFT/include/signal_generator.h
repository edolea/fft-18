#ifndef SIGNAL_GENERATOR_H
#define SIGNAL_GENERATOR_H

#include <vector>
#include <complex>
#include <random>
#include <functional>

// Type alias for convenience
using Complex = std::complex<double>;
using Signal = std::vector<Complex>;

/**
 * Class for generating test signals
 */
class SignalGenerator {
public:
    // Constructor initializes random number generator
    SignalGenerator(unsigned int seed = std::random_device{}());

    // Generate sine wave: A*sin(2π*f*t + φ)
    Signal generateSineWave(size_t length, double frequency, double amplitude = 1.0,
                            double phase = 0.0, double sample_rate = 1.0);

    // Generate superposition of sine waves
    Signal generateMultiSineWave(size_t length, const std::vector<double>& frequencies,
                                 const std::vector<double>& amplitudes = {});

    // Generate white noise
    Signal generateWhiteNoise(size_t length, double amplitude = 1.0);

    // Generate impulse signal (delta function)
    Signal generateImpulse(size_t length, size_t position = 0, double amplitude = 1.0);

    // Generate step function
    Signal generateStep(size_t length, size_t position = 0, double amplitude = 1.0);

    // Generate chirp signal (frequency sweep)
    Signal generateChirp(size_t length, double start_freq, double end_freq,
                         double amplitude = 1.0);

    // Generate signal from arbitrary function
    Signal generateFromFunction(size_t length, std::function<double(double)> func);

    // Generate composite test signal with various components for transform testing
    Signal generateTestSignal(size_t length);

private:
    std::mt19937 rng_;
    std::normal_distribution<double> normal_dist_;
    std::uniform_real_distribution<double> uniform_dist_;
};

#endif // SIGNAL_GENERATOR_H