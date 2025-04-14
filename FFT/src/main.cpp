#include <iostream>
#include <vector>
#include <memory>
#include "transform.h"
#include "fft_transform.h"
#include "dwt_transform.h"
#include "signal_utils.h"

// Function to print complex signal
void printSignal(const Signal& signal, const std::string& label) {
    std::cout << label << std::endl;
    for (size_t i = 0; i < std::min(signal.size(), size_t(10)); ++i) {
        std::cout << i << ": " << signal[i].real() << " + " << signal[i].imag() << "i" << std::endl;
    }
    if (signal.size() > 10) {
        std::cout << "... (" << signal.size() - 10 << " more values)" << std::endl;
    }
    std::cout << std::endl;
}

// Function to demonstrate a transform
void demonstrateTransform(TransformBase& transform, const Signal& signal) {
    std::cout << "===== " << transform.getName() << " =====" << std::endl;

    try {
        if (!transform.canProcess(signal)) {
            std::cout << "Signal can't be processed by this transform. Padding to power of two..." << std::endl;
            Signal padded = SignalUtils::padToPowerOfTwo(signal);

            Signal result = transform.forward(padded);
            printSignal(result, "Transform Result:");

            Signal reconstructed = transform.inverse(result);
            printSignal(reconstructed, "Reconstructed Signal:");
        } else {
            Signal result = transform.forward(signal);
            printSignal(result, "Transform Result:");

            Signal reconstructed = transform.inverse(result);
            printSignal(reconstructed, "Reconstructed Signal:");
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    std::cout << std::endl;
}

int main() {
    // Create a simple test signal (a sine wave with 8 samples)
    std::vector<double> real_signal_data = {0.0, 0.7071, 1.0, 0.7071, 0.0, -0.7071, -1.0, -0.7071};

    // Convert to complex for our transforms
    Signal signal = SignalUtils::realToComplex(real_signal_data);

    // Print original signal
    printSignal(signal, "Original Signal:");

    // Configure and demonstrate FFT
    FFTTransform::Config fft_config;
    fft_config.normalize_output = true;
    FFTTransform fft_transform(fft_config);
    demonstrateTransform(fft_transform, signal);

    // Configure and demonstrate DWT
    DWTTransform dwt_transform(DWTTransform::HAAR);
    demonstrateTransform(dwt_transform, signal);

    return 0;
}