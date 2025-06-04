#include <gtest/gtest.h>
#include "abstract_transform.hpp"
#include "recursive_fourier.hpp"
#include "iterative_fourier.hpp"
#include "vector_generator.hpp"
#include <cmath>

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
        {
            std::cout << "Difference at index " << i << ": "
                      << v1[i] << " vs " << v2[i] << std::endl;
            return false;
        }
    }
    return true;
}

class FourierTransformTest : public ::testing::Test
{
protected:
    RecursiveFourier<doubleVector> recursiveFft;
    IterativeFourier<doubleVector> iterativeFft;
};

TEST_F(FourierTransformTest, SameOutputForRandomInput)
{
    // Test for different power-of-two sizes
    for (size_t size : {2, 4, 8, 16, 32})
    {
        const auto input = RandomVectorGenerator::generate<doubleVector>(size);
        doubleVector recursive_output;
        doubleVector iterative_output;

        recursiveFft.computeDir(input, recursive_output);
        iterativeFft.computeDir(input, iterative_output);

        EXPECT_TRUE(areEqual(recursive_output, iterative_output))
            << "FFT implementations produced different results for size " << size;
    }
}

// Test with simple input where we know the expected FFT result
TEST_F(FourierTransformTest, KnownInput)
{
    // Simple case: [1, 0, 0, 0] -> [1, 1, 1, 1]
    doubleVector input = {complexDouble(1, 0), complexDouble(0, 0),
                          complexDouble(0, 0), complexDouble(0, 0)};

    doubleVector recursive_output;
    doubleVector iterative_output;

    recursiveFft.computeDir(input, recursive_output);
    iterativeFft.computeDir(input, iterative_output);

    // Both should equal [1, 1, 1, 1]
    doubleVector expected = {complexDouble(1, 0), complexDouble(1, 0),
                             complexDouble(1, 0), complexDouble(1, 0)};

    EXPECT_TRUE(areEqual(recursive_output, expected));
    EXPECT_TRUE(areEqual(iterative_output, expected));
}

TEST_F(FourierTransformTest, ComparePerformance)
{
    // Just for observation, not a pass/fail test
    const auto input = RandomVectorGenerator::generate<doubleVector>(1024);
    doubleVector recursive_output;
    doubleVector iterative_output;

    recursiveFft.computeDir(input, recursive_output);
    iterativeFft.computeDir(input, iterative_output);

    recursiveFft.executionTime();
    iterativeFft.executionTime();

    EXPECT_TRUE(areEqual(recursive_output, iterative_output));
}
