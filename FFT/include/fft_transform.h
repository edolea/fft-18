#ifndef FFT_TRANSFORM_H
#define FFT_TRANSFORM_H

#include "transform.h"
#include <string>

/**
 * Fast Fourier Transform implementation
 */
class FFTTransform : public TransformBase {
public:
    // Configuration options for FFT
    struct Config {
        bool normalize_output = true;
        bool use_recursive_algorithm = true;
    };

    FFTTransform();
    explicit FFTTransform(const Config& config);

    Signal forward(const Signal& signal) override;
    Signal inverse(const Signal& transformed_signal) override;
    bool canProcess(const Signal& signal) const override;
    std::string getName() const override;

private:
    Config config_;

    // Helper method for recursive FFT implementation
    Signal fft_recursive(const Signal& signal, bool inverse = false);

    // Helper to check if number is power of two
    bool isPowerOfTwo(size_t n) const;
};

#endif // FFT_TRANSFORM_H