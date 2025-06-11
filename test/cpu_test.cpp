#include <gtest/gtest.h>
#include "abstract_transform.hpp"
#include "recursive_fourier.hpp"
#include "iterative_fourier.hpp"
#include "vector_generator.hpp"
#include <cmath>
#include "mpi_wrapper.hpp"


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

// it always run with 1 process since cmake not configured here,
// good to check if compiler links OpenMP correctly
TEST(MPIWrapperTest, BasicFunctionality) {
    // Initialize the MPI wrapper
    MPIWrapper mpi;

    // Basic info test
    std::cout << "Process " << mpi.rank() << " out of " << mpi.size() << " started" << std::endl;

    // Test broadcast
    std::vector<int> data(10);
    if (mpi.isRoot()) {
        // Root process initializes the data
        for (int i = 0; i < 10; i++) {
            data[i] = i * 10;
        }
        std::cout << "Root initialized data" << std::endl;
    }

    // Broadcast data from root to all processes
    mpi.broadcast(data.data(), data.size());

    // Verify broadcast worked
    for (int i = 0; i < 10; i++) {
        EXPECT_EQ(data[i], i * 10) << "Broadcast data mismatch at index " << i;
    }
    std::cout << "Process " << mpi.rank() << ": Broadcast test passed" << std::endl;

    // Test allGather
    std::vector<int> localData(2);
    localData[0] = mpi.rank() * 100;
    localData[1] = mpi.rank() * 100 + 50;

    std::vector<int> allData(2 * mpi.size());
    mpi.allGather(localData.data(), sizeof(int) * 2,
                  allData.data(), sizeof(int) * 2);

    // Verify allGather worked
    for (int i = 0; i < mpi.size(); i++) {
        EXPECT_EQ(allData[i*2], i * 100) << "AllGather data mismatch for rank " << i;
        EXPECT_EQ(allData[i*2 + 1], i * 100 + 50) << "AllGather data mismatch for rank " << i;
    }
    std::cout << "Process " << mpi.rank() << ": AllGather test passed" << std::endl;

    // Test send/recv
    if (mpi.size() > 1) {
        int testValue = 42;

        if (mpi.rank() == 0) {
            // Send to process 1
            mpi.send(&testValue, 1, 1, 123);
            std::cout << "Process 0: Sent value " << testValue << " to process 1" << std::endl;
        }
        else if (mpi.rank() == 1) {
            int receivedValue = 0;
            mpi.recv(&receivedValue, 1, 0, 123);
            EXPECT_EQ(receivedValue, 42) << "Received wrong value in point-to-point communication";
            std::cout << "Process 1: Received value " << receivedValue << " from process 0" << std::endl;
        }
    }

    // Barrier to synchronize all processes
    mpi.barrier();
    if (mpi.isRoot()) {
        std::cout << "All MPI tests completed successfully!" << std::endl;
    }
}

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

TEST_F(FourierTransformTest, Compare2DFftImplementations) {
    for (int size : {8, 16, 32}) {
        const double frequency = 2.0;
        const double amplitude = 1.0;

        // Generate the 2D test matrix
        const auto input = RandomVectorGenerator::generate<doubleMatrix>(size, frequency, amplitude);

        doubleMatrix recursive_output;
        doubleMatrix iterative_output;

        std::cout << "\n===== Testing 2D FFT Comparison for size " << size << "x" << size << " =====\n";

        // Direct FFT
        recursiveFft2D.compute(input, recursive_output);
        iterativeFft2D.compute(input, iterative_output);

        std::cout << "Recursive 2D FFT time:  " << recursiveFft2D.getTime() << " seconds\n";
        std::cout << "Iterative 2D FFT time:  " << iterativeFft2D.getTime() << " seconds\n";

        // Compare both FFT outputs
        EXPECT_TRUE(areEqual2D(recursive_output, iterative_output))
            << "Mismatch between recursive and iterative 2D FFT for size " << size << "x" << size;
    }
}


TEST_F(FourierTransformTest, Inverse2DFftRestoresInput) {
    for (int size : {8, 16, 32}) {
        const double frequency = 2.0;
        const double amplitude = 1.0;

        // Generate the 2D test matrix
        const auto input = RandomVectorGenerator::generate<doubleMatrix>(size, frequency, amplitude);

        doubleMatrix fft_output;
        doubleMatrix inverse_recursive, inverse_iterative;

        std::cout << "\n===== Testing 2D FFT + IFFT Round-Trip for size " << size << "x" << size << " =====\n";

        // FFT
        recursiveFft2D.compute(input, fft_output);

        // IFFT
        recursiveFft2D.compute(fft_output, inverse_recursive, false);
        iterativeFft2D.compute(fft_output, inverse_iterative, false);

        std::cout << "Recursive IFFT time:  " << recursiveFft2D.getTime() << " seconds\n";
        std::cout << "Iterative IFFT time:  " << iterativeFft2D.getTime() << " seconds\n";

        // Compare restored signals to original input
        EXPECT_TRUE(areEqual2D(inverse_recursive, input))
            << "Recursive 2D FFT+IFFT did not restore input for size " << size << "x" << size;

        EXPECT_TRUE(areEqual2D(inverse_iterative, input))
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

    /*
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
    */
}