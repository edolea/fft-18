#include "signal_generator.h"
#include <cmath>

const double PI = 3.14159265358979323846;

SignalGenerator::SignalGenerator(unsigned int seed)
        : rng_(seed),
          normal_dist_(0.0, 1.0),
          uniform_dist_(-1.0, 1.0) {
}

Signal SignalGenerator::generateSineWave(size_t length, double frequency,
                                         double amplitude, double phase, double sample_rate) {
    Signal signal(length);
    for (size_t i = 0; i < length; ++i) {
        double t = static_cast<double>(i) / sample_rate;
        double value = amplitude * std::sin(2.0 * PI * frequency * t + phase);
        signal[i] = Complex(value, 0.0);
    }
    return signal;
}

Signal SignalGenerator::generateMultiSineWave(size_t length,
                                              const std::vector<double>& frequencies,
                                              const std::vector<double>& amplitudes) {
    Signal signal(length, Complex(0.0, 0.0));

    std::vector<double> amps;
    if (amplitudes.empty()) {
        // Default: all amplitudes are 1.0
        amps.resize(frequencies.size(), 1.0);
    } else if (amplitudes.size() >= frequencies.size()) {
        amps = amplitudes;
    } else {
        // If fewer amplitudes than frequencies, use what's available and fill with 1.0
        amps = amplitudes;
        amps.resize(frequencies.size(), 1.0);
    }

    for (size_t f = 0; f < frequencies.size(); ++f) {
        for (size_t i = 0; i < length; ++i) {
            double t = static_cast<double>(i);
            double value = amps[f] * std::sin(2.0 * PI * frequencies[f] * t / length);
            signal[i] += Complex(value, 0.0);
        }
    }

    return signal;
}

Signal SignalGenerator::generateWhiteNoise(size_t length, double amplitude) {
    Signal signal(length);
    for (size_t i = 0; i < length; ++i) {
        double value = normal_dist_(rng_) * amplitude;
        signal[i] = Complex(value, 0.0);
    }
    return signal;
}

Signal SignalGenerator::generateImpulse(size_t length, size_t position, double amplitude) {
    Signal signal(length, Complex(0.0, 0.0));
    if (position < length) {
        signal[position] = Complex(amplitude, 0.0);
    }
    return signal;
}

Signal SignalGenerator::generateStep(size_t length, size_t position, double amplitude) {
    Signal signal(length, Complex(0.0, 0.0));
    for (size_t i = position; i < length; ++i) {
        signal[i] = Complex(amplitude, 0.0);
    }
    return signal;
}

Signal SignalGenerator::generateChirp(size_t length, double start_freq, double end_freq, double amplitude) {
    Signal signal(length);
    double freq_rate = (end_freq - start_freq) / length;

    for (size_t i = 0; i < length; ++i) {
        // Current instantaneous frequency
        double freq = start_freq + freq_rate * i;
        // Phase is integral of frequency
        double phase = 2.0 * PI * (start_freq * i + 0.5 * freq_rate * i * i) / length;
        double value = amplitude * std::sin(phase);
        signal[i] = Complex(value, 0.0);
    }

    return signal;
}

Signal SignalGenerator::generateFromFunction(size_t length, std::function<double(double)> func) {
    Signal signal(length);
    for (size_t i = 0; i < length; ++i) {
        double t = static_cast<double>(i) / length;
        double value = func(t);
        signal[i] = Complex(value, 0.0);
    }
    return signal;
}

Signal SignalGenerator::generateTestSignal(size_t length) {
    // Create a complex test signal with multiple components

    // 1. Base sine wave at 1/8 of Nyquist
    Signal signal = generateSineWave(length, length/16.0, 1.0);

    // 2. Add higher frequency component
    Signal high_freq = generateSineWave(length, length/4.0, 0.5);
    for (size_t i = 0; i < length; ++i) {
        signal[i] += high_freq[i];
    }

    // 3. Add an impulse
    signal[length/4] += Complex(2.0, 0.0);

    // 4. Add some noise
    Signal noise = generateWhiteNoise(length, 0.1);
    for (size_t i = 0; i < length; ++i) {
        signal[i] += noise[i];
    }

    // 5. Add a localized burst of high frequency (good for wavelet testing)
    size_t burst_start = length * 3 / 4;
    size_t burst_length = length / 8;
    Signal burst = generateSineWave(burst_length, length/8.0, 0.8);
    for (size_t i = 0; i < burst_length && (burst_start + i) < length; ++i) {
        signal[burst_start + i] += burst[i];
    }

    return signal;
}