#include <gtest/gtest.h>
#include "vector_generator_gpu.hpp"
#include "parallel_fourier.hpp"
#include "parameter_class.hpp"
#include "recursive_fourier.hpp"
#include <cmath>

int ParameterClass::N = 0;
double ParameterClass::frequency = 0.0;
double ParameterClass::amplitude = 0.0;

constexpr double TOLERANCE = 1e-4;

bool areEqual(const doubleVector &v1, const doubleVector &v2, double tolerance = 1e-10)
{
    if (v1.size() != v2.size()) return false;
    for (size_t i = 0; i < v1.size(); i++) {
        double real_diff = std::abs(v1[i].real() - v2[i].real());
        double imag_diff = std::abs(v1[i].imag() - v2[i].imag());
        if (real_diff > tolerance || imag_diff > tolerance) return false;
    }
    return true;
}

bool areEqual2D(const doubleMatrix &a, const doubleMatrix &b, double tol = TOLERANCE) {
    if (a.size() != b.size() || a[0].size() != b[0].size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        for (size_t j = 0; j < a[0].size(); ++j) {
            if (std::abs(a[i][j].real() - b[i][j].real()) > tol ||
                std::abs(a[i][j].imag() - b[i][j].imag()) > tol)
                return false;
        }
    }
    return true;
}

class ParallelFourierGPUTest : public ::testing::Test {
protected:
    void SetUp() override {
        ParameterClass::initializeParameters(1024, 0.5, 1.0);
    }
};

// ✅ Basic call check
TEST_F(ParallelFourierGPUTest, Compute1D_DoesNotThrow) {
    const auto vec = RandomVectorGenerator::generate<doubleVector>(ParameterClass::N);
    ParallelFourier<doubleVector> fft(vec);
    EXPECT_NO_THROW(fft.compute(vec));
}

// ✅ CPU vs GPU FFT match
TEST_F(ParallelFourierGPUTest, CompareCPUtoGPU_1DFFT) {
    const auto input = RandomVectorGenerator::generate<doubleVector>(ParameterClass::N);
    RecursiveFourier<doubleVector> cpuFft;
    doubleVector cpu_output;
    cpuFft.compute(input, cpu_output, true);

    ParallelFourier<doubleVector> gpuFft(input);
    gpuFft.compute(input);
    auto gpu_output = gpuFft.getOutput();

    EXPECT_TRUE(areEqual(cpu_output, gpu_output))
        << "Mismatch between CPU and GPU 1D FFT!";
}

// ✅ Inverse GPU FFT restores input
TEST_F(ParallelFourierGPUTest, InverseRestoresInput_1D) {
    const auto input = RandomVectorGenerator::generate<doubleVector>(ParameterClass::N);
    ParallelFourier<doubleVector> fft(input);
    fft.compute(input);  // Forward
    auto forward = fft.getOutput();

    fft.compute(forward, false);  // Inverse
    auto restored = fft.getOutput();

    EXPECT_TRUE(areEqual(input, restored, 1e-5))
        << "Inverse 1D FFT did not restore original!";
}

// ✅ Performance print (no assertion)
TEST_F(ParallelFourierGPUTest, PrintPerformance_1D) {
    std::vector<int> sizes = {512, 1024, 2048, 4096};
    for (int N : sizes) {
        ParameterClass::initializeParameters(N, 0.5, 1.0);
        const auto input = RandomVectorGenerator::generate<doubleVector>(N);

        RecursiveFourier<doubleVector> cpuFft;
        doubleVector cpu_output;
        cpuFft.compute(input, cpu_output);

        ParallelFourier<doubleVector> gpuFft(input);
        gpuFft.compute(input);

        std::cout << "N = " << N
                  << " | CPU time: " << cpuFft.getTime()
                  << " | GPU time: " << gpuFft.getTime() << "\n";
    }
}

// ✅ GPU vs CPU 2D FFT match
TEST_F(ParallelFourierGPUTest, CompareCPUtoGPU_2DFFT) {
    const int size = 2048;
    ParameterClass::initializeParameters(size, 2.0, 1.0);
    const auto input = RandomVectorGenerator::generate2D<doubleMatrix>(size, ParameterClass::frequency, ParameterClass::amplitude);

    RecursiveFourier<doubleMatrix> cpuFft;
    doubleMatrix cpu_output;
    cpuFft.compute(input, cpu_output, true);

    ParallelFourier<doubleMatrix> gpuFft(input);
    gpuFft.compute(input, true);
    auto gpu_output = gpuFft.getOutput();

    EXPECT_TRUE(areEqual2D(cpu_output, gpu_output))
        << "Mismatch between CPU and GPU 2D FFT!";
}

// ✅ 2D Inverse FFT
TEST_F(ParallelFourierGPUTest, InverseRestoresInput_2D) {
    const int size = 2048;
    ParameterClass::initializeParameters(size, 2.0, 1.0);
    const auto input = RandomVectorGenerator::generate2D<doubleMatrix>(size, ParameterClass::frequency, ParameterClass::amplitude);

    ParallelFourier<doubleMatrix> fft(input);
    fft.compute(input, true);
    auto fft_output = fft.getOutput();

    fft.compute(fft_output, false);
    auto restored = fft.getOutput();

    EXPECT_TRUE(areEqual2D(input, restored))
        << "2D inverse FFT did not restore original!";
}
