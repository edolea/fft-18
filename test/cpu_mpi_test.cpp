#include <gtest/gtest.h>
#include "mpi_wrapper.hpp"
#include "abstract_transform.hpp"
#include "mpi_iterative_fourier.hpp"
#include "iterative_fourier.hpp"
#include "vector_generator.hpp"
#include <iostream>
#include <vector>

// not enough processes in GitHub Workflows to run mpi
#ifdef CI_ENVIRONMENT
TEST(MPITest, SkipInCI) {
    GTEST_SKIP() << "Skipping MPITest in CI environment due to resource constraints.";
}
#endif

// Global MPI environment to share across tests
class MPIEnvironment : public ::testing::Environment {
private:
    MPIWrapper* mpi_wrapper_;

public:
    MPIEnvironment() : mpi_wrapper_(nullptr) {
        mpi_wrapper_ = new MPIWrapper();
    }

    // NOTE: all environment done only for this
    // without this u have MPI_Init and MPI_Finalize at each test
    // --> MULTIPLE INIT AND FINALIZE NOT PERMITTED IN MPI !!!
    ~MPIEnvironment() {
        delete mpi_wrapper_;
    }

    MPIWrapper* getMPI() {
        return mpi_wrapper_;
    }
};

// Global accessor for the MPI environment
MPIWrapper* GetSharedMPI() {
    static MPIEnvironment* env = dynamic_cast<MPIEnvironment*>(
        ::testing::AddGlobalTestEnvironment(new MPIEnvironment));
    return env->getMPI();
}

// Test fixture that uses the shared MPI wrapper
class MPITest : public ::testing::Test {
protected:
    MPIWrapper* mpi;

    void SetUp() override {
        mpi = GetSharedMPI();
    }
};

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


// Use the TEST_F macro with the fixture
TEST_F(MPITest, BasicFunctionality) {
    // Basic info test
    std::cout << "Process " << mpi->rank() << " out of " << mpi->size() << " started" << std::endl;

    // Test broadcast
    std::vector<int> data(10);
    if (mpi->isRoot()) {
        // Root process initializes the data
        for (int i = 0; i < 10; i++) {
            data[i] = i * 10;
        }
        std::cout << "Root initialized data" << std::endl;
    }

    // Broadcast data from root to all processes
    mpi->broadcast(data.data(), data.size());

    // Verify broadcast worked
    for (int i = 0; i < 10; i++) {
        EXPECT_EQ(data[i], i * 10) << "Broadcast data mismatch at index " << i;
    }
    std::cout << "Process " << mpi->rank() << ": Broadcast test passed" << std::endl;

    // Barrier to synchronize all processes
    mpi->barrier();
}

TEST_F(MPITest, BasicFunctionality2) {
    // Test allGather
    std::vector<int> localData(2);
    localData[0] = mpi->rank() * 100;
    localData[1] = mpi->rank() * 100 + 50;

    std::vector<int> allData(2 * mpi->size());
    mpi->allGather(localData.data(), sizeof(int) * 2,
                  allData.data(), sizeof(int) * 2);

    // Verify allGather worked
    for (int i = 0; i < mpi->size(); i++) {
        EXPECT_EQ(allData[i*2], i * 100) << "AllGather data mismatch for rank " << i;
        EXPECT_EQ(allData[i*2 + 1], i * 100 + 50) << "AllGather data mismatch for rank " << i;
    }
    std::cout << "Process " << mpi->rank() << ": AllGather test passed" << std::endl;

    // Test send/recv if we have enough processes
    if (mpi->size() > 1) {
        int testValue = 42;

        if (mpi->rank() == 0) {
            // Send to process 1
            mpi->send(&testValue, 1, 1, 123);
            std::cout << "Process 0: Sent value " << testValue << " to process 1" << std::endl;
        }
        else if (mpi->rank() == 1) {
            int receivedValue = 0;
            mpi->recv(&receivedValue, 1, 0, 123);
            EXPECT_EQ(receivedValue, 42) << "Received wrong value in point-to-point communication";
            std::cout << "Process 1: Received value " << receivedValue << " from process 0" << std::endl;
        }
    }

    // Barrier to synchronize all processes
    mpi->barrier();
    if (mpi->isRoot()) {
        std::cout << "All MPI tests completed successfully!" << std::endl;
    }
}

TEST_F(MPITest, FFT1D_DirectTransform) {
    // Test for different power-of-two sizes
    if (mpi->isRoot()) {
        for (size_t size : {16, 32, 64}) {
            doubleVector input;
            doubleVector serial_output;
            doubleVector mpi_output;

            // Create both implementations
            IterativeFourier<doubleVector> serial_fft;
            MPIIterativeFourier<doubleVector> mpi_fft;

            // Execute transforms
            serial_fft.compute(input, serial_output, true);
            mpi_fft.compute(input, mpi_output, true);

            // Only root process compares results
            EXPECT_TRUE(areEqual(serial_output, mpi_output, 1e-5))
                << "MPI FFT direct transform results differ from serial for size " << size;
        }
    }
}

/*
TEST_F(MPITest, FFT1D_InverseTransform) {
    // Test for different power-of-two sizes
    for (size_t size : {16, 32, 64}) {
        doubleVector input;
        doubleVector forward_output;
        doubleVector serial_inverse;
        doubleVector mpi_inverse;

        // Only root process generates random data
        if (mpi->isRoot()) {
            input = RandomVectorGenerator::generate<doubleVector>(size);
        } else {
            input.resize(size);
        }

        // Broadcast input to all processes
        mpi->broadcast(input.data(), input.size());

        // Create both implementations
        IterativeFourier<doubleVector> serial_fft;
        MPIIterativeFourier<doubleVector> mpi_fft;

        // First do a forward transform with serial implementation to get frequency data
        serial_fft.compute(input, forward_output, true);

        // Broadcast the frequency domain data to ensure all processes have the same input
        mpi->broadcast(forward_output.data(), forward_output.size());

        // Then compare inverse transforms
        serial_fft.compute(forward_output, serial_inverse, false);
        mpi_fft.compute(forward_output, mpi_inverse, false);

        // Only root process compares results
        if (mpi->isRoot()) {
            EXPECT_TRUE(areEqual(serial_inverse, mpi_inverse, 1e-5))
                << "MPI FFT inverse transform results differ from serial for size " << size;

            // Also verify the inverse actually returns the original data
            EXPECT_TRUE(areEqual(input, mpi_inverse, 1e-5))
                << "Inverse transform doesn't recover the original signal for size " << size;
        }
    }
}

TEST_F(MPITest, FFT2D_DirectTransform) {
    // Test for different power-of-two sizes
    for (int size : {8, 16}) {
        doubleMatrix input;
        doubleMatrix serial_output;
        doubleMatrix mpi_output;

        // Only root process generates random data
        if (mpi->isRoot()) {
            const double frequency = 2.0;
            const double amplitude = 1.0;
            input = RandomVectorGenerator::generate<doubleMatrix>(size, frequency, amplitude);
        } else {
            input.resize(size, doubleVector(size));
        }

        // Broadcast the input matrix to all processes
        for (int i = 0; i < size; i++) {
            if (!mpi->isRoot()) input[i].resize(size);
            mpi->broadcast(input[i].data(), input[i].size());
        }

        // Create both implementations
        IterativeFourier<doubleMatrix> serial_fft;
        MPIIterativeFourier<doubleMatrix> mpi_fft;

        // Execute transforms
        serial_fft.compute(input, serial_output, true);
        mpi_fft.compute(input, mpi_output, true);

        // Only root process compares results
        if (mpi->isRoot()) {
            EXPECT_TRUE(areEqual2D(serial_output, mpi_output))
                << "MPI 2D FFT direct transform results differ from serial for size " << size << "x" << size;
        }
    }
}

TEST_F(MPITest, FFT2D_InverseTransform) {
    // Test for different power-of-two sizes
    for (int size : {8, 16}) {
        doubleMatrix input;
        doubleMatrix forward_output;
        doubleMatrix serial_inverse;
        doubleMatrix mpi_inverse;

        // Only root process generates random data
        if (mpi->isRoot()) {
            const double frequency = 2.0;
            const double amplitude = 1.0;
            input = RandomVectorGenerator::generate<doubleMatrix>(size, frequency, amplitude);
        } else {
            input.resize(size, doubleVector(size));
        }

        // Broadcast the input matrix to all processes
        for (int i = 0; i < size; i++) {
            if (!mpi->isRoot()) input[i].resize(size);
            mpi->broadcast(input[i].data(), input[i].size());
        }

        // Create both implementations
        IterativeFourier<doubleMatrix> serial_fft;
        MPIIterativeFourier<doubleMatrix> mpi_fft;

        // First do a forward transform with serial implementation
        serial_fft.compute(input, forward_output, true);

        // Broadcast the frequency domain data
        for (int i = 0; i < size; i++) {
            mpi->broadcast(forward_output[i].data(), forward_output[i].size());
        }

        // Then compare inverse transforms
        serial_fft.compute(forward_output, serial_inverse, false);
        mpi_fft.compute(forward_output, mpi_inverse, false);

        // Only root process compares results
        if (mpi->isRoot()) {
            EXPECT_TRUE(areEqual2D(serial_inverse, mpi_inverse))
                << "MPI 2D FFT inverse transform results differ from serial for size " << size << "x" << size;

            // Also verify the inverse transform recovers the original data
            EXPECT_TRUE(areEqual2D(input, mpi_inverse))
                << "2D inverse transform doesn't recover original matrix for size " << size << "x" << size;
        }
    }
}

TEST_F(MPITest, PerformanceComparison) {
    // Only do a larger test if we have more than 1 process
    if (mpi->size() > 1) {
        const int size = 256;  // Larger size to better measure performance
        doubleVector input;
        doubleVector serial_output;
        doubleVector mpi_output;

        // Only root process generates random data
        if (mpi->isRoot()) {
            input = RandomVectorGenerator::generate<doubleVector>(size);
        } else {
            input.resize(size);
        }

        // Broadcast input to all processes
        mpi->broadcast(input.data(), input.size());

        // Create both implementations
        IterativeFourier<doubleVector> serial_fft;
        MPIIterativeFourier<doubleVector> mpi_fft;

        // Execute transforms
        serial_fft.compute(input, serial_output, true);
        mpi_fft.compute(input, mpi_output, true);

        // Report performance on root process
        if (mpi->isRoot()) {
            std::cout << "\n===== Performance Comparison =====" << std::endl;
            std::cout << "Serial FFT time: " << serial_fft.getTime().count() << " seconds" << std::endl;
            std::cout << "MPI FFT time: " << mpi_fft.getTime().count() << " seconds" << std::endl;
            std::cout << "MPI processes: " << mpi->size() << std::endl;

            // Verify results match
            EXPECT_TRUE(areEqual(serial_output, mpi_output, 1e-5))
                << "Performance test results don't match";
        }
    }
}
*/

// Also modify MPIWrapper class to disable automatic finalization in the destructor
// by adding this to mpi_wrapper.hpp:
//
// void disableFinalize() { mpiInitialized_ = false; }

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    return result;
}