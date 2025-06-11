//
// Created by Edoardo Leali on 11/06/25.
//
#include <gtest/gtest.h>
#include "mpi_wrapper.hpp"
#include "mpi_iterative_fourier.hpp"
#include "iterative_fourier.hpp"
#include "vector_generator.hpp"

TEST(MPI_FFT_Simple, OneDimensional) {
    std::cout << "Process " << " started" << std::endl;



    MPIWrapper mpi;
    size_t size = 16;
    doubleVector input, serial_output, mpi_output;
    std::cout << "Process " << mpi.rank() + 1 << " out of " << mpi.size() << " started" << std::endl;
    mpi.barrier();



    if (mpi.isRoot()) {
        input = RandomVectorGenerator::generate<doubleVector>(size);
    } else {
        input.resize(size);
    }

    mpi.broadcast(input.data(), input.size());

    IterativeFourier<doubleVector> serial_fft;
    MPIIterativeFourier<doubleVector> mpi_fft;

    if (mpi.isRoot()) {
        serial_fft.compute(input, serial_output, true);
    }

    mpi_fft.compute(input, mpi_output, true);

    mpi.barrier();

    if (mpi.isRoot()) {
        ASSERT_EQ(serial_output.size(), mpi_output.size());
        for (size_t i = 0; i < serial_output.size(); ++i) {
            ASSERT_NEAR(serial_output[i].real(), mpi_output[i].real(), 1e-5);
            ASSERT_NEAR(serial_output[i].imag(), mpi_output[i].imag(), 1e-5);
        }
    }
}

// since mpi01 test is called without the main from gtest --> manually include main
int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}

