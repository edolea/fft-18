#ifndef OPTIMIZED_FFT_H
#define OPTIMIZED_FFT_H

#include "transform.h"
#include <vector>
#include <complex>

/**
 * Optimized Fast Fourier Transform implementation
 * Uses iterative algorithm to avoid recursion overhead
 */
class OptimizedFFT : public TransformBase {
public:
    enum WindowType {
        NONE,
        HANNING,
        HAMMING,
        BLACKMAN,
        GAUSSIAN
    };

    // Configuration options for FFT
    struct Config {
        bool normalize_output = true;
        WindowType window_type = NONE;
        double window_param = 0.5;  // Parameter for certain windows (e.g., Gaussian sigma)
    };

    OptimizedFFT();
    explicit OptimizedFFT(const Config& config);

    Signal forward(const Signal& signal) override;
    Signal inverse(const Signal& transformed_signal) override;
    bool canProcess(const Signal& signal) const override;
    std::string getName() const override;

    // Getters and setters
    void setWindowType(WindowType window_type);
    WindowType getWindowType() const;
    void setNormalize(bool normalize);

private:
    Config config_;

    // Precomputed twiddle factors for efficiency
    std::vector<std::vector<Complex>> twiddle_factors_;

    // Initialize twiddle factors for given size
    void initTwiddleFactors(size_t size);

    // Bit-reverse permutation
    void bitReversePermutation(Signal& signal);

    // Calculate bit-reversed index
    size_t reverseBits(size_t x, size_t log2n) const;

    // Apply window function
    Signal applyWindow(const Signal& signal) const;

    // Check if input size is valid (power of 2)
    bool isPowerOfTwo(size_t n) const;

    // Get log2 of integer (for power of 2 values)
    size_t log2(size_t n) const;
};

#endif // OPTIMIZED_FFT_H