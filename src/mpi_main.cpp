#include <mpi.h>
#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include "iterative_fourier.hpp"
#include "mpi_iterative_fourier.hpp"
#include "vector_generator.hpp"

using complexDouble = std::complex<double>;
using doubleVector = std::vector<complexDouble>;
using doubleMatrix = std::vector<doubleVector>;

// Generate test data for FFT
template<typename T>
void generateTestData(T& data, int n) {
    if constexpr (ComplexVector<T>) {
        data.resize(n);
        for (int i = 0; i < n; i++) {
            double real = std::cos(2 * M_PI * i / n);
            double imag = std::sin(2 * M_PI * i / n);
            data[i] = {real, imag};
        }
    } else if constexpr (ComplexVectorMatrix<T>) {
        data.resize(n, doubleVector(n));
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double real = std::cos(2 * M_PI * (i + j) / n);
                double imag = std::sin(2 * M_PI * (i + j) / n);
                data[i][j] = {real, imag};
            }
        }
    }
}

// Validate FFT results
template<typename T>
bool compareResults(const T& parallel, const T& sequential, double tolerance = 1e-10) {
    if constexpr (ComplexVector<T>) {
        for (size_t i = 0; i < parallel.size(); i++) {
            if (std::abs(parallel[i] - sequential[i]) > tolerance) {
                std::cout << "Mismatch at index " << i << ": parallel = " << parallel[i] << ", sequential = " << sequential[i] << std::endl;
                return false;
            }
        }
    } else if constexpr (ComplexVectorMatrix<T>) {
        for (size_t i = 0; i < parallel.size(); i++) {
            for (size_t j = 0; j < parallel[i].size(); j++) {
                if (std::abs(parallel[i][j] - sequential[i][j]) > tolerance)
                    return false;
            }
        }
    }
    return true;
}

int main(int argc, char** argv) {
    // Initialize MPI
    int provided;

    int err = MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (err != MPI_SUCCESS) {
        std::cerr << "MPI_Init_thread failed with error code " << err << std::endl;
        MPI_Abort(MPI_COMM_WORLD, err);
        return 1;
    }

    if (provided != MPI_THREAD_FUNNELED) {
        std::cerr << "MPI_THREAD_FUNNELED not implemented" << std::endl;
        MPI_Finalize();
        return 1;
    }

    int rank, world_size;
    err = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (err != MPI_SUCCESS) {
        std::cerr << "MPI_Comm_rank failed with error code " << err << std::endl;
        MPI_Abort(MPI_COMM_WORLD, err);
    }

    err = MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (err != MPI_SUCCESS) {
        std::cerr << "MPI_Comm_size failed with error code " << err << std::endl;
        MPI_Abort(MPI_COMM_WORLD, err);
    }

    // Parse command line arguments
    if (argc < 3) {
        if (rank == 0) std::cerr << "Usage: " << argv[0] << " <dimension (1|2)> <size>" << std::endl;
        MPI_Finalize();
        return 1;
    }

    int dimension = std::atoi(argv[1]);
    int n = std::atoi(argv[2]);

    // Validate inputs
    if (!isPowerOfTwo(n)) {
        if (rank == 0) std::cerr << "Error: Size must be a power of 2" << std::endl;
        MPI_Finalize();
        return 1;
    }

    if (n % world_size != 0) {
        if (rank == 0) std::cerr << "Error: Size must be divisible by number of processes" << std::endl;
        MPI_Finalize();
        return 1;
    }

    // 1D FFT
    if (dimension == 1) {
        doubleVector input, parallel_output, sequential_output;

        if (rank == 0) {
            generateTestData(input, n);
            parallel_output.resize(n);
            sequential_output.resize(n);
        }

        MpiIterativeFourier<doubleVector> parallelFFT(MPI_COMM_WORLD);

        parallelFFT.compute(input, parallel_output, true);

        if (rank == 0) {
            IterativeFourier<doubleVector> sequentialFFT;
            sequentialFFT.compute(input, sequential_output, true);

            std::cout << n << "  " << sequentialFFT.getTime().count() << "  " << parallelFFT.getTime().count()
                      << "  " << sequentialFFT.getTime().count() / parallelFFT.getTime().count() << std::endl;
        }
    }
    // 2D FFT
    else if (dimension == 2) {
        doubleMatrix input, parallel_output, sequential_output;

        if (rank == 0) {
            generateTestData(input, n);
            parallel_output.resize(n, doubleVector(n));
            sequential_output.resize(n, doubleVector(n));
        }

        MpiIterativeFourier<doubleMatrix> parallelFFT(MPI_COMM_WORLD);
        parallelFFT.compute(input, parallel_output, true);

        if (rank == 0) {
            IterativeFourier<doubleMatrix> sequentialFFT;
            sequentialFFT.compute(input, sequential_output, true);

            std::cout << n << "  " << sequentialFFT.getTime().count() << "  " << parallelFFT.getTime().count()
                      << "  " << sequentialFFT.getTime().count() / parallelFFT.getTime().count() << std::endl;
        }
    } else {
        if (rank == 0) std::cerr << "Error: Dimension must be 1 or 2" << std::endl;
    }

    MPI_Finalize();
    return 0;
}