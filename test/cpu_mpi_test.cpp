#include <gtest/gtest.h>
#include "mpi_wrapper.hpp"
#include <iostream>
#include <vector>

// Global MPI environment to share across tests
class MPIEnvironment : public ::testing::Environment {
private:
    MPIWrapper* mpi_wrapper_;

public:
    MPIEnvironment() : mpi_wrapper_(nullptr) {
        mpi_wrapper_ = new MPIWrapper();
    }

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

// Also modify MPIWrapper class to disable automatic finalization in the destructor
// by adding this to mpi_wrapper.hpp:
//
// void disableFinalize() { mpiInitialized_ = false; }

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    return result;
}