#ifndef DWT_TRANSFORM_H
#define DWT_TRANSFORM_H

#include "transform.h"
#include <string>

/**
 * Discrete Wavelet Transform implementation
 */
class DWTTransform : public TransformBase {
public:
    enum WaveletType {
        HAAR,
        DAUBECHIES4,
        DAUBECHIES6,
        DAUBECHIES8
    };

    // Constructor with configurable wavelet type and decomposition level
    explicit DWTTransform(WaveletType type = DAUBECHIES4, int level = -1);

    Signal forward(const Signal& signal) override;
    Signal inverse(const Signal& coefficients) override;
    bool canProcess(const Signal& signal) const override;
    std::string getName() const override;

    // Getter/setter for wavelet type
    DWTTransform::WaveletType getWaveletType() const;
    void setWaveletType(WaveletType type);

    // Getter/setter for decomposition level
    int getDecompositionLevel() const;
    void setDecompositionLevel(int level);

private:
    WaveletType wavelet_type_;
    int level_;  // Decomposition level, -1 means full decomposition

    // Filter coefficients for the selected wavelet
    std::vector<double> lowPassDecompose() const;
    std::vector<double> highPassDecompose() const;
    std::vector<double> lowPassReconstruct() const;
    std::vector<double> highPassReconstruct() const;

    // Helper to check if number is power of two
    bool isPowerOfTwo(size_t n) const;

    // Core DWT implementation methods
    Signal decompose(const Signal& signal, int level);
    Signal reconstruct(const Signal& coefficients, int level);
};

#endif // DWT_TRANSFORM_H