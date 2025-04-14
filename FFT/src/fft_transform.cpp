#include "fft_transform.h"
#include <stdexcept>
#include <cmath>
#include <algorithm>

const double PI = 3.14159265358979323846;

FFTTransform::FFTTransform() : config_() {}

FFTTransform::FFTTransform(const Config& config) : config_(config) {}

Signal FFTTransform::forward(const Signal& signal) {
    if (!canProcess(signal)) {
        throw std::invalid_argument("Signal length must be a power of 2 for FFT");
    }

    return fft_recursive(signal, false);
}

Signal FFTTransform::inverse(const Signal& transformed_signal) {
    if (!canProcess(transformed_signal)) {
        throw std::invalid_argument("Transform length must be a power of 2 for IFFT");
    }

    Signal result = fft_recursive(transformed_signal, true);

    // Normalize the result if configured to do so
    if (config_.normalize_output) {
        double n = static_cast<double>(transformed_signal.size());
        for (auto& val : result) {
            val /= n;
        }
    }

    return result;
}

Signal FFTTransform::fft_recursive(const Signal& signal, bool inverse) {
    size_t n = signal.size();

    // Base case
    if (n <= 1) {
        return signal;
    }

    // Split into even and odd indices
    Signal even(n/2);
    Signal odd(n/2);

    for (size_t i = 0; i < n/2; ++i) {
        even[i] = signal[2*i];
        odd[i] = signal[2*i + 1];
    }

    // Recursive calls
    Signal even_result = fft_recursive(even, inverse);
    Signal odd_result = fft_recursive(odd, inverse);

    // Combine results
    Signal result(n);
    double angle_factor = 2 * PI / n * (inverse ? -1.0 : 1.0);

    for (size_t k = 0; k < n/2; ++k) {
        Complex twiddle = std::polar(1.0, angle_factor * k);
        Complex p = twiddle * odd_result[k];
        result[k] = even_result[k] + p;
        result[k + n/2] = even_result[k] - p;
    }

    return result;
}

bool FFTTransform::canProcess(const Signal& signal) const {
    return signal.size() > 0 && isPowerOfTwo(signal.size());
}

std::string FFTTransform::getName() const {
    return "Fast Fourier Transform (FFT)";
}

bool FFTTransform::isPowerOfTwo(size_t n) const {
    return (n != 0) && ((n & (n - 1)) == 0);
}