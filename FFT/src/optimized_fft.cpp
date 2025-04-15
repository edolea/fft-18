#include "optimized_fft.h"
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <complex>

const double PI = 3.14159265358979323846;

OptimizedFFT::OptimizedFFT() : config_() {}

OptimizedFFT::OptimizedFFT(const Config& config) : config_(config) {}

Signal OptimizedFFT::forward(const Signal& signal) {
    if (!canProcess(signal)) {
        throw std::invalid_argument("Signal length must be a power of 2 for FFT");
    }

    // Apply windowing if configured
    Signal windowed_signal;
    if (config_.window_type != NONE) {
        windowed_signal = applyWindow(signal);
    } else {
        windowed_signal = signal;
    }

    // Make a copy of the input signal
    Signal result = windowed_signal;
    size_t n = result.size();

    // Initialize twiddle factors if needed
    if (twiddle_factors_.empty() || twiddle_factors_[0].size() != n) {
        initTwiddleFactors(n);
    }

    // Bit-reverse permutation
    bitReversePermutation(result);

    // Cooley-Tukey FFT algorithm (iterative, decimation-in-time)
    size_t log2n = log2(n);
    for (size_t s = 1; s <= log2n; ++s) {
        size_t m = 1 << s;  // 2^s
        size_t m_half = m >> 1;  // m/2

        // Use precomputed twiddle factors for this stage
        const auto& stage_twiddles = twiddle_factors_[s-1];

        for (size_t k = 0; k < n; k += m) {
            for (size_t j = 0; j < m_half; ++j) {
                Complex t = stage_twiddles[j] * result[k + j + m_half];
                Complex u = result[k + j];

                // Butterfly operation
                result[k + j] = u + t;
                result[k + j + m_half] = u - t;
            }
        }
    }

    return result;
}

Signal OptimizedFFT::inverse(const Signal& transformed_signal) {
    if (!canProcess(transformed_signal)) {
        throw std::invalid_argument("Transform length must be a power of 2 for IFFT");
    }

    // Take complex conjugate for inverse
    Signal conjugate(transformed_signal.size());
    for (size_t i = 0; i < transformed_signal.size(); ++i) {
        conjugate[i] = std::conj(transformed_signal[i]);
    }

    // Perform forward transform on conjugate
    Signal result = forward(conjugate);

    // Take conjugate again and scale
    for (size_t i = 0; i < result.size(); ++i) {
        result[i] = std::conj(result[i]);

        // Normalize if configured
        if (config_.normalize_output) {
            result[i] /= static_cast<double>(result.size());
        }
    }

    return result;
}

bool OptimizedFFT::canProcess(const Signal& signal) const {
    return signal.size() > 0 && isPowerOfTwo(signal.size());
}

std::string OptimizedFFT::getName() const {
    return "Optimized Fast Fourier Transform";
}

void OptimizedFFT::setWindowType(WindowType window_type) {
    config_.window_type = window_type;
}

OptimizedFFT::WindowType OptimizedFFT::getWindowType() const {
    return config_.window_type;
}

void OptimizedFFT::setNormalize(bool normalize) {
    config_.normalize_output = normalize;
}

void OptimizedFFT::initTwiddleFactors(size_t size) {
    size_t log2n = log2(size);
    twiddle_factors_.resize(log2n);

    for (size_t s = 1; s <= log2n; ++s) {
        size_t m = 1 << s;  // 2^s
        size_t m_half = m >> 1;  // m/2

        twiddle_factors_[s-1].resize(m_half);
        for (size_t j = 0; j < m_half; ++j) {
            double angle = -2.0 * PI * j / m;
            twiddle_factors_[s-1][j] = std::polar(1.0, angle);
        }
    }
}

void OptimizedFFT::bitReversePermutation(Signal& signal) {
    size_t n = signal.size();
    size_t log2n = log2(n);

    for (size_t i = 0; i < n; ++i) {
        size_t rev = reverseBits(i, log2n);
        if (i < rev) {
            std::swap(signal[i], signal[rev]);
        }
    }
}

size_t OptimizedFFT::reverseBits(size_t x, size_t log2n) const {
    size_t result = 0;
    for (size_t i = 0; i < log2n; ++i) {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    return result;
}

Signal OptimizedFFT::applyWindow(const Signal& signal) const {
    Signal windowed(signal.size());

    for (size_t i = 0; i < signal.size(); ++i) {
        double window_coef = 1.0;
        double n = static_cast<double>(signal.size());

        switch (config_.window_type) {
            case HANNING:
                window_coef = 0.5 * (1.0 - std::cos(2.0 * PI * i / (n - 1)));
                break;
            case HAMMING:
                window_coef = 0.54 - 0.46 * std::cos(2.0 * PI * i / (n - 1));
                break;
            case BLACKMAN:
                window_coef = 0.42 - 0.5 * std::cos(2.0 * PI * i / (n - 1)) +
                              0.08 * std::cos(4.0 * PI * i / (n - 1));
                break;
            case GAUSSIAN: {
                double sigma = config_.window_param;
                double arg = (i - n/2) / (sigma * n/2);
                window_coef = std::exp(-0.5 * arg * arg);
                break;
            }
            case NONE:
            default:
                window_coef = 1.0;
                break;
        }

        windowed[i] = signal[i] * window_coef;
    }

    return windowed;
}

bool OptimizedFFT::isPowerOfTwo(size_t n) const {
    return (n != 0) && ((n & (n - 1)) == 0);
}

size_t OptimizedFFT::log2(size_t n) const {
    size_t result = 0;
    while (n > 1) {
        n >>= 1;
        ++result;
    }
    return result;
}