#include <gtest/gtest.h>
#include "vector_generator_gpu.hpp"
#include "parallel_fourier.hpp"
#include "parameter_class.hpp"
#include "recursive_fourier.hpp" // Add this for CPU reference
#include <cmath>
int ParameterClass::N = 0;
double ParameterClass::frequency = 0.0;
double ParameterClass::amplitude = 0.0;
// Utility function to compare complex vectors with tolerance
bool areEqual(const doubleVector &v1, const doubleVector &v2, double tolerance = 1e-10)
{
    if (v1.size() != v2.size())
        return false;

    for (size_t i = 0; i < v1.size(); i++)
    {
        double real_diff = std::abs(v1[i].real() - v2[i].real());
        double imag_diff = std::abs(v1[i].imag() - v2[i].imag());
        if (real_diff > tolerance || imag_diff > tolerance)
            return false;
    }
    return true;
}

class ParallelFourierGPUTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        ParameterClass::initializeParameters(1024, 0.5, 1.0);
    }
};

TEST_F(ParallelFourierGPUTest, ComputeDoesNotThrow)
{
    const auto vec = RandomVectorGenerator::generate<doubleVector>(ParameterClass::N);
    ParallelFourier<doubleVector> parallelFourier(vec);

    EXPECT_NO_THROW(parallelFourier.compute());
}

TEST(GPUvsCPU, CompareCPUandGPU)
{
    ParameterClass::initializeParameters(1024, 0.5, 1.0);
    const auto input = RandomVectorGenerator::generate<doubleVector>(ParameterClass::N);

    // CPU computation
    RecursiveFourier<doubleVector> recursiveFft;
    doubleVector cpu_output;
    recursiveFft.computeDir(input, cpu_output);

    // GPU computation
    ParallelFourier<doubleVector> gpuFft(input);
    gpuFft.compute();

    // Try to get the result: adjust this line if your class uses a different member
    const auto &gpu_output = gpuFft.output; // or gpuFft.getResult() if that's correct

    EXPECT_TRUE(areEqual(cpu_output, gpu_output))
        << "CPU and GPU FFT results differ!";
}
TEST(GPUvsCPU, ComparePerformanceSmallSizes)
{
    std::vector<int> sizes = {512, 1024, 2048, 4096};
    for (int N : sizes)
    {
        ParameterClass::initializeParameters(N, 0.5, 1.0);
        const auto input = RandomVectorGenerator::generate<doubleVector>(N);

        // CPU computation timing
        RecursiveFourier<doubleVector> recursiveFft;
        doubleVector cpu_output;
        recursiveFft.computeDir(input, cpu_output);

        // GPU computation timing
        ParallelFourier<doubleVector> gpuFft(input);
        gpuFft.compute();

        // Print performance results (not a test assertion)
        std::cout << "N = " << N
                  << " | CPU time: " << recursiveFft.time.count() << " ms"
                  << " | GPU time: " << gpuFft.execution_time.count() << " ms" << std::endl;
    }
}
