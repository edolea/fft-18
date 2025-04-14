#include "signal_utils.h"
#include <cmath>

const double PI = 3.14159265358979323846;

Signal SignalUtils::padToPowerOfTwo(const Signal& signal) {
    size_t size = signal.size();

    if (size == 0) return Signal(1);

    // Find the next power of 2
    size_t power = 1;
    while (power < size) {
        power *= 2;
    }

    Signal result = signal;
    result.resize(power, Complex(0, 0));
    return result;
}

Signal SignalUtils::applyWindow(const Signal& signal, const std::string& window_type) {
    std::vector<double> window;

    if (window_type == "hanning") {
        window = hanningWindow(signal.size());
    } else if (window_type == "hamming") {
        window = hammingWindow(signal.size());
    } else if (window_type == "blackman") {
        window = blackmanWindow(signal.size());
    } else {
        // Default to rectangular window (no change)
        return signal;
    }

    Signal result(signal.size());
    for (size_t i = 0; i < signal.size(); ++i) {
        result[i] = signal[i] * window[i];
    }

    return result;
}

std::vector<double> SignalUtils::hanningWindow(size_t size) {
    std::vector<double> window(size);
    for (size_t i = 0; i < size; ++i) {
        window[i] = 0.5 * (1 - std::cos(2 * PI * i / (size - 1)));
    }
    return window;
}

std::vector<double> SignalUtils::hammingWindow(size_t size) {
    std::vector<double> window(size);
    for (size_t i = 0; i < size; ++i) {
        window[i] = 0.54 - 0.46 * std::cos(2 * PI * i / (size - 1));
    }
    return window;
}

std::vector<double> SignalUtils::blackmanWindow(size_t size) {
    std::vector<double> window(size);
    for (size_t i = 0; i < size; ++i) {
        window[i] = 0.42 - 0.5 * std::cos(2 * PI * i / (size - 1)) + 0.08 * std::cos(4 * PI * i / (size - 1));
    }
    return window;
}

Signal SignalUtils::realToComplex(const std::vector<double>& real_signal) {
    Signal complex_signal(real_signal.size());
    for (size_t i = 0; i < real_signal.size(); ++i) {
        complex_signal[i] = Complex(real_signal[i], 0);
    }
    return complex_signal;
}

std::vector<double> SignalUtils::complexToReal(const Signal& complex_signal) {
    std::vector<double> real_signal(complex_signal.size());
    for (size_t i = 0; i < complex_signal.size(); ++i) {
        real_signal[i] = complex_signal[i].real();
    }
    return real_signal;
}

bool SignalUtils::isPowerOfTwo(size_t n) {
    return (n != 0) && ((n & (n - 1)) == 0);
}