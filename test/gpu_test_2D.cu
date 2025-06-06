#include <gtest/gtest.h>
#include "vector_generator_gpu.hpp"
#include "parallel_fourier.hpp"
#include "parameter_class.hpp"
#include "recursive_fourier.hpp" // Add this for CPU reference
#include <cmath>
int ParameterClass::N = 0;
double ParameterClass::frequency = 0.0;
double ParameterClass::amplitude = 0.0;

constexpr double TOLERANCE = 1e-5;

// Utility function to compare two 2D complex matrices with tolerance
bool areEqual2D(const doubleMatrix &a, const doubleMatrix &b, double tol = TOLERANCE) {
    if (a.size() != b.size() || a[0].size() != b[0].size()) return false;

    for (size_t i = 0; i < a.size(); ++i) {
        for (size_t j = 0; j < a[0].size(); ++j) {
            if (std::abs(a[i][j].real() - b[i][j].real()) > tol ||
                std::abs(a[i][j].imag() - b[i][j].imag()) > tol) {
                return false;
                }
        }
    }
    return true;
}

// Test fixture
class ParallelFourier2DGPUTest : public ::testing::Test {
protected:
    void SetUp() override {
        size = 32;
        ParameterClass::initializeParameters(size, 2.0, 1.0);
        input = RandomVectorGenerator::generate<doubleMatrix>(size, ParameterClass::frequency, ParameterClass::amplitude);
    }

    int size;
    doubleMatrix input;
};

// ✅ Test: Compare GPU 2D FFT output with CPU reference
TEST_F(ParallelFourier2DGPUTest, CompareGPUWithCPU2DFFT) {
    // CPU
    RecursiveFourier<doubleMatrix> recursiveFft;
    doubleMatrix cpu_output;
    recursiveFft.compute(input, cpu_output, true);

    // GPU
    ParallelFourier<doubleMatrix> gpuFft(input);
    gpuFft.compute(input, true);
    auto gpu_output = gpuFft.getOutput();

    EXPECT_TRUE(areEqual2D(cpu_output, gpu_output))
        << "Mismatch between CPU and GPU 2D FFT output!";
}

// ✅ Test: Ensure inverse GPU FFT restores original input
TEST_F(ParallelFourier2DGPUTest, InverseRestoresOriginal2D) {
    // GPU forward FFT
    ParallelFourier<doubleMatrix> gpuFft(input);
    gpuFft.compute(input, true);
    auto fft_output = gpuFft.getOutput();

    // GPU inverse FFT
    gpuFft.compute(fft_output, false);
    auto inverse_output = gpuFft.getOutput();

    EXPECT_TRUE(areEqual2D(input, inverse_output))
        << "GPU 2D FFT + IFFT did not restore original input!";
}