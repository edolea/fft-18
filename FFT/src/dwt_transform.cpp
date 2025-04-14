#include "dwt_transform.h"
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <sstream>

DWTTransform::DWTTransform(WaveletType type, int level)
        : wavelet_type_(type), level_(level) {}

Signal DWTTransform::forward(const Signal& signal) {
    if (!canProcess(signal)) {
        throw std::invalid_argument("Signal length must be a power of 2 and >= 2 for DWT");
    }

    int max_level = static_cast<int>(std::log2(signal.size()));
    int actual_level = (level_ < 0 || level_ > max_level) ? max_level : level_;

    return decompose(signal, actual_level);
}

Signal DWTTransform::inverse(const Signal& coefficients) {
    if (!canProcess(coefficients)) {
        throw std::invalid_argument("Coefficients length must be a power of 2 and >= 2 for IDWT");
    }

    int max_level = static_cast<int>(std::log2(coefficients.size()));
    int actual_level = (level_ < 0 || level_ > max_level) ? max_level : level_;

    return reconstruct(coefficients, actual_level);
}

bool DWTTransform::canProcess(const Signal& signal) const {
    return signal.size() >= 2 && isPowerOfTwo(signal.size());
}

std::string DWTTransform::getName() const {
    std::string wavelet_name;

    switch (wavelet_type_) {
        case HAAR:
            wavelet_name = "Haar";
            break;
        case DAUBECHIES4:
            wavelet_name = "Daubechies 4";
            break;
        case DAUBECHIES6:
            wavelet_name = "Daubechies 6";
            break;
        case DAUBECHIES8:
            wavelet_name = "Daubechies 8";
            break;
    }

    std::stringstream ss;
    ss << "Discrete Wavelet Transform (" << wavelet_name << ")";
    return ss.str();
}

DWTTransform::WaveletType DWTTransform::getWaveletType() const {
    return wavelet_type_;
}

void DWTTransform::setWaveletType(WaveletType type) {
    wavelet_type_ = type;
}

int DWTTransform::getDecompositionLevel() const {
    return level_;
}

void DWTTransform::setDecompositionLevel(int level) {
    level_ = level;
}

std::vector<double> DWTTransform::lowPassDecompose() const {
    switch (wavelet_type_) {
        case HAAR:
            return {0.7071067811865475, 0.7071067811865475};
        case DAUBECHIES4:
            return {0.4829629131445341, 0.8365163037378077,
                    0.2241438680420134, -0.1294095225512603};
        case DAUBECHIES6:
            return {0.3326705529500826, 0.8068915093110925,
                    0.4598775021184915, -0.1350110200102546,
                    -0.0854412738820267, 0.0352262918857096};
        case DAUBECHIES8:
            return {0.2303778133088964, 0.7148465705529154,
                    0.6308807679298587, -0.0279837694168599,
                    -0.1870348117190931, 0.0308413818355607,
                    0.0328830116668852, -0.0105974017850690};
        default:
            return {0.7071067811865475, 0.7071067811865475}; // Default to Haar
    }
}

std::vector<double> DWTTransform::highPassDecompose() const {
    std::vector<double> lowpass = lowPassDecompose();
    size_t n = lowpass.size();
    std::vector<double> highpass(n);

    for (size_t i = 0; i < n; ++i) {
        highpass[i] = lowpass[n - 1 - i] * (i % 2 == 0 ? 1 : -1);
    }

    return highpass;
}

std::vector<double> DWTTransform::lowPassReconstruct() const {
    return highPassDecompose();
}

std::vector<double> DWTTransform::highPassReconstruct() const {
    std::vector<double> highpass = highPassDecompose();
    for (auto& val : highpass) {
        val = -val;
    }
    return highpass;
}

Signal DWTTransform::decompose(const Signal& signal, int level) {
    // For demonstration purposes - real implementation would be more complex
    // This would involve convolution with filters and downsampling

    // Placeholder simplified implementation
    Signal result = signal;  // Make a copy to start with

    // In a real implementation:
    // 1. Apply low and high pass filters to the signal
    // 2. Downsample both results by 2
    // 3. Keep the high-pass (detail) coefficients
    // 4. Recursively process the low-pass (approximation) coefficients

    return result;
}

Signal DWTTransform::reconstruct(const Signal& coefficients, int level) {
    // For demonstration purposes - real implementation would be more complex
    // This would involve upsampling and convolution with reconstruction filters

    // Placeholder simplified implementation
    Signal result = coefficients;  // Make a copy to start with

    // In a real implementation:
    // 1. Start from the coarsest level
    // 2. Upsample approximation and detail coefficients
    // 3. Apply reconstruction filters
    // 4. Add the results together
    // 5. Repeat for all levels

    return result;
}

bool DWTTransform::isPowerOfTwo(size_t n) const {
    return (n != 0) && ((n & (n - 1)) == 0);
}