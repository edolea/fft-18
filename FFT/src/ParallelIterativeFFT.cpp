//
// Created by Edoardo Leali on 26/03/25.
//
#include <mpi.h>
#include <omp.h>
#include <vector>
#include <complex>
#include <cmath>
#include "../include/Cooley-Tukey-parallel.hpp"

std::vector<std::complex<double>> ParallelIterativeFFT::findFFT(std::vector<std::complex<double>> input) {
    int n = input.size();
    int m = log2(n);
    std::vector<std::complex<double>> y(n);

    // Bit-reversal permutation
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        int j = 0;
        for (int k = 0; k < m; k++) {
            if (i & (1 << k)) {
                j |= (1 << (m - 1 - k));
            }
        }
        y[j] = input[i];
    }

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::cout << "Hello from process " << rank << " out of " << size << "!" << std::endl;

    // Iterative FFT
    for (int j = 1; j <= m; j++) {
        int d = 1 << j;
        std::complex<double> wd(std::cos(2 * M_PI / d), std::sin(2 * M_PI / d));

        #pragma omp parallel for schedule(static)
        for (int k = 0; k < d / 2; k++) {
            std::complex<double> w = std::pow(wd, k); // Precompute the value of w for this task
            for (int i = k + rank; i < n; i += d * size) {
                std::complex<double> t = w * y[i + d / 2];
                std::complex<double> x = y[i];
                y[i] = x + t;
                y[i + d / 2] = x - t;
            }
        }

        // Synchronize results across all processes
        MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, y.data(), n / size, MPI_DOUBLE_COMPLEX, MPI_COMM_WORLD);
    }

    return y;
}

/*
 * // TODO: check if this is faster
 // Optimize the MPI communication pattern
std::vector<std::complex<double>> ParallelIterativeFFT::findFFT(std::vector<std::complex<double>> input) {
    int n = input.size();
    int m = log2(n);
    std::vector<std::complex<double>> y(n);

    // Bit-reversal permutation (already parallelized correctly)
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        // ... (existing code)
    }

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Precompute twiddle factors for all stages
    std::vector<std::complex<double>> twiddle_factors[m+1];
    for (int j = 1; j <= m; j++) {
        int d = 1 << j;
        std::complex<double> wd(std::cos(2 * M_PI / d), std::sin(2 * M_PI / d));
        twiddle_factors[j].resize(d/2);
        for (int k = 0; k < d/2; k++) {
            twiddle_factors[j][k] = std::pow(wd, k);
        }
    }

    // Create MPI datatype for complex<double>
    MPI_Datatype complex_type;
    MPI_Type_contiguous(2, MPI_DOUBLE, &complex_type);
    MPI_Type_commit(&complex_type);

    // Calculate chunk size for each process
    int chunk_size = n / size;
    std::vector<std::complex<double>> local_chunk(chunk_size);

    // Iterative FFT with improved communication
    for (int j = 1; j <= m; j++) {
        int d = 1 << j;

        #pragma omp parallel for schedule(dynamic, 64)
        for (int k = 0; k < d / 2; k++) {
            std::complex<double> w = twiddle_factors[j][k]; // Use precomputed value
            for (int i = k + rank; i < n; i += d * size) {
                std::complex<double> t = w * y[i + d / 2];
                std::complex<double> x = y[i];
                y[i] = x + t;
                y[i + d / 2] = x - t;
            }
        }

        // Correct MPI_Allgather usage
        MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                     y.data(), n, complex_type, MPI_COMM_WORLD);
    }

    MPI_Type_free(&complex_type);
    return y;
}
 */