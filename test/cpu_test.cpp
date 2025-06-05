#include <gtest/gtest.h>
#include "abstract_transform.hpp"
#include "recursive_fourier.hpp"
#include "iterative_fourier.hpp"
#include "vector_generator.hpp"
#include <cmath>

// Utility function to compare complex vectors with tolerance
bool areEqual(const doubleVector &v1, const doubleVector &v2, double tolerance = 1e-10) {
    if (v1.size() != v2.size())
        return false;

    for (size_t i = 0; i < v1.size(); i++)
    {
        double real_diff = std::abs(v1[i].real() - v2[i].real());
        double imag_diff = std::abs(v1[i].imag() - v2[i].imag());
        if (real_diff > tolerance || imag_diff > tolerance)
        {
            std::cout << "Difference at index " << i << ": "
                      << v1[i] << " vs " << v2[i] << std::endl;
            return false;
        }
    }
    return true;
}

// Utility function to compare 2D complex matrices with tolerance
bool areEqual2D(const std::vector<std::vector<complexDouble>> &m1, const std::vector<std::vector<complexDouble>> &m2, double tolerance = 1e-3) {
    if (m1.size() != m2.size())
        return false;

    for (size_t i = 0; i < m1.size(); i++) {
        if (m1[i].size() != m2[i].size())
            return false;

        for (size_t j = 0; j < m1[i].size(); j++) {
            double real_diff = std::abs(m1[i][j].real() - m2[i][j].real());
            double imag_diff = std::abs(m1[i][j].imag() - m2[i][j].imag());
            if (real_diff > tolerance || imag_diff > tolerance) {
                std::cout << "Difference at position [" << i << "][" << j << "]: "
                          << m1[i][j] << " vs " << m2[i][j] << std::endl;
                return false;
            }
        }
    }
    return true;
}

class FourierTransformTest : public ::testing::Test {
protected:
    RecursiveFourier<doubleVector> recursiveFft;
    IterativeFourier<doubleVector> iterativeFft;

    RecursiveFourier<doubleMatrix> recursiveFft2D;
    IterativeFourier<doubleMatrix> iterativeFft2D;
};

TEST_F(FourierTransformTest, SameOutputForRandomInput_Direct) {
    // Test for different power-of-two sizes
    for (size_t size : {2, 4, 8, 16, 32, 128, 256, 1024})
    {
        const auto input = RandomVectorGenerator::generate<doubleVector>(size);
        doubleVector recursive_output;
        doubleVector iterative_output;

        recursiveFft.compute(input, recursive_output);
        iterativeFft.compute(input, iterative_output);

        EXPECT_TRUE(areEqual(recursive_output, iterative_output))
            << "FFT implementations produced different results for size " << size;
    }
}

TEST_F(FourierTransformTest, SameOutputForRandomInput_Inverse)
{
    // Test for different power-of-two sizes
    for (size_t size : {2, 4, 8, 16, 32, 128, 256, 1024})
    {
        const auto input = RandomVectorGenerator::generate<doubleVector>(size);
        doubleVector recursive_output;
        doubleVector iterative_output;

        recursiveFft.compute(input, recursive_output, false);
        iterativeFft.compute(input, iterative_output, false);

        EXPECT_TRUE(areEqual(recursive_output, iterative_output))
            << "FFT implementations produced different results for size " << size;
    }
}

// Test with simple input where we know the expected FFT result
TEST_F(FourierTransformTest, KnownInput) {
    // Simple case: [1, 0, 0, 0] -> [1, 1, 1, 1]
    doubleVector input = {complexDouble(1, 0), complexDouble(0, 0),
                          complexDouble(0, 0), complexDouble(0, 0)};

    doubleVector recursive_output;
    doubleVector iterative_output;

    recursiveFft.compute(input, recursive_output);
    iterativeFft.compute(input, iterative_output);

    // Both should equal [1, 1, 1, 1]
    doubleVector expected = {complexDouble(1, 0), complexDouble(1, 0),
                             complexDouble(1, 0), complexDouble(1, 0)};

    EXPECT_TRUE(areEqual(recursive_output, expected));
    EXPECT_TRUE(areEqual(iterative_output, expected));
}

// Test that applying FFT followed by inverse FFT returns the original input (within tolerance) for N = 64
TEST_F(FourierTransformTest, InverseFftRestoresInput_N64) {
    const size_t size = 64;
    const auto input = RandomVectorGenerator::generate<doubleVector>(size);

    doubleVector fft_output, ifft_output;

    // Recursive FFT and inverse FFT
    recursiveFft.compute(input, fft_output);
    recursiveFft.compute(fft_output, ifft_output, false); // Already normalized

    EXPECT_TRUE(areEqual(input, ifft_output))
        << "Recursive FFT+IFFT did not restore input for size " << size;

    // Iterative FFT and inverse FFT
    iterativeFft.compute(input, fft_output);
    iterativeFft.compute(fft_output, ifft_output, false); // Already normalized

    EXPECT_TRUE(areEqual(input, ifft_output))
        << "Iterative FFT+IFFT did not restore input for size " << size;
}

// Test inverse FFT on a known frequency domain vector for N = 64
TEST_F(FourierTransformTest, InverseFftKnownInput_N64){
    const size_t size = 64;
    // Frequency domain: [64, 0, 0, ..., 0] (should yield time domain: [1, 1, ..., 1])
    doubleVector freq_input(size, complexDouble(0, 0));
    freq_input[0] = complexDouble(static_cast<double>(size), 0);

    doubleVector expected(size, complexDouble(1, 0));
    doubleVector output;

    // Recursive
    recursiveFft.compute(freq_input, output, false); // Already normalized
    EXPECT_TRUE(areEqual(output, expected)) << "Recursive IFFT failed on known input";

    // Iterative
    iterativeFft.compute(freq_input, output, false); // Already normalized
    EXPECT_TRUE(areEqual(output, expected)) << "Iterative IFFT failed on known input";
}

TEST_F(FourierTransformTest, Test2DFft) {
    // Test with different sizes
    for (int size : {8, 16, 32}) {
        // Generate a 2D sinusoidal pattern
        const double frequency = 2.0;
        const double amplitude = 1.0;

        // Generate the 2D test matrix using the specified generator
        const auto input = RandomVectorGenerator::generate<doubleMatrix>(size, frequency, amplitude);

        doubleMatrix recursive_output;
        doubleMatrix iterative_output;

        std::cout << "\n===== Testing 2D FFT of size " << size << "x" << size << " =====\n";

        // Direct transform
        recursiveFft2D.compute(input, recursive_output);
        iterativeFft2D.compute(input, iterative_output);

        std::cout << "******* recursive 2D " << size << "x" << size << " TIME: "
                  << recursiveFft2D.getTime() << "*******" << std::endl;
        std::cout << "******* iterative 2D " << size << "x" << size << " TIME: "
                  << iterativeFft2D.getTime() << "*******" << std::endl;

        // Compare outputs from both implementations
        EXPECT_TRUE(areEqual2D(recursive_output, iterative_output))
            << "2D FFT implementations produced different results for size " << size << "x" << size;

        // Test inverse transform
        doubleMatrix recursive_inverse;
        doubleMatrix iterative_inverse;

        //recursiveFft2D.compute(recursive_output, recursive_inverse, false);
        iterativeFft2D.compute(iterative_output, iterative_inverse, false);

        std::cout << "******* recursive inverse 2D " << size << "x" << size << " TIME: "
                  << recursiveFft2D.getTime() << "*******" << std::endl;
        std::cout << "******* iterative inverse 2D " << size << "x" << size << " TIME: "
                  << iterativeFft2D.getTime() << "*******" << std::endl;

        // Check if applying FFT and then inverse FFT returns the original input
        EXPECT_TRUE(areEqual2D(recursive_inverse, input))
            << "Recursive 2D FFT+IFFT did not restore input for size " << size << "x" << size;
        EXPECT_TRUE(areEqual2D(iterative_inverse, input))
            << "Iterative 2D FFT+IFFT did not restore input for size " << size << "x" << size;
    }
}

TEST_F(FourierTransformTest, ComparePerformance) {
    const auto input_10 = RandomVectorGenerator::generate<doubleVector>(1024);
    const auto input_15 = RandomVectorGenerator::generate<doubleVector>(32768);
    const auto input_20 = RandomVectorGenerator::generate<doubleVector>(1048576);

    doubleVector recursive_output_10, recursive_output_15, recursive_output_20;
    doubleVector iterative_output_10, iterative_output_15, iterative_output_20;

    std::cout << "\n===== Testing size 2^10 =====\n";
    recursiveFft.compute(input_10, recursive_output_10);
    iterativeFft.compute(input_10, iterative_output_10);

    std::cout << "******* recursive 2^10 TIME: " << recursiveFft.getTime() << "*******" << std::endl;
    std::cout << "******* iterative 2^10 TIME: " << iterativeFft.getTime() << "*******" << std::endl;

    EXPECT_TRUE(areEqual(recursive_output_10, iterative_output_10));

    std::cout << "\n===== Testing size 2^15 =====\n";
    recursiveFft.compute(input_15, recursive_output_15);
    iterativeFft.compute(input_15, iterative_output_15);

    std::cout << "******* recursive 2^15 TIME: " << recursiveFft.getTime() << "*******" << std::endl;
    std::cout << "******* iterative 2^15 TIME: " << iterativeFft.getTime() << "*******" << std::endl;

    EXPECT_TRUE(areEqual(recursive_output_15, iterative_output_15));

    std::cout << "\n===== Testing size 2^20 =====\n";
    recursiveFft.compute(input_20, recursive_output_20);
    iterativeFft.compute(input_20, iterative_output_20);

    std::cout << "******* recursive 2^20 TIME: " << recursiveFft.getTime() << "*******" << std::endl;
    std::cout << "******* iterative 2^20 TIME: " << iterativeFft.getTime() << "*******\n\n" << std::endl;

    EXPECT_TRUE(areEqual(recursive_output_20, iterative_output_20));
}