#ifndef SIGNAL_UTILS_H
#define SIGNAL_UTILS_H

#include <vector>
#include <complex>
#include <string>

// Type alias for convenience
using Complex = std::complex<double>;
using Signal = std::vector<Complex>;

/**
 * Utility class for signal processing operations
 */
class SignalUtils {
public:
    // Pad signal to next power of two
    static Signal padToPowerOfTwo(const Signal& signal);

    // Apply various window functions
    static Signal applyWindow(const Signal& signal, const std::string& window_type = "hanning");

    // Convert real signal to complex
    static Signal realToComplex(const std::vector<double>& real_signal);

    // Extract real part from complex signal
    static std::vector<double> complexToReal(const Signal& complex_signal);

    // Check if size is power of two
    static bool isPowerOfTwo(size_t n);

private:
    // Window function implementations
    static std::vector<double> hanningWindow(size_t size);
    static std::vector<double> hammingWindow(size_t size);
    static std::vector<double> blackmanWindow(size_t size);
};

#endif // SIGNAL_UTILS_H