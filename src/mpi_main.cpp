// main.cpp
#include <mpi.h>
#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <cassert>
#include "mpi_iterative_fourier.hpp"
#include "iterative_fourier.hpp"  // iterative FFT

// Helper comparison for complex vectors/matrices
template<typename T>
bool approxEqual(const T &a, const T &b, double eps = 1e-6) {
    if constexpr (std::is_same_v<T, std::vector<std::complex<double>>>) {
        if (a.size() != b.size()) {
            std::cerr << "Error in MPIParallelFourier: size mismatch 1D" << std::endl;
            return false;
        }
        for (size_t i = 0; i < a.size(); ++i)
            if (std::abs(a[i] - b[i]) > eps) {
                std::cerr << "ERROOOOOR: a[" << i << "] = "<< a[i] << " - b[" << i << "] = " << b[i] << "< toll " << eps << std::endl;
                return false;
            }
        return true;
    } else {
        // matrix
        if (a.size() != b.size()) {
            std::cerr << "Error in MPIParallelFourier: size mismatch 2D" << std::endl;
            return false;
        }
        for (size_t i = 0; i < a.size(); ++i) {
            if (a[i].size() != b[i].size()) return false;
            for (size_t j = 0; j < a[i].size(); ++j)
                if (std::abs(a[i][j] - b[i][j]) > eps) return false;
        }
        return true;
    }
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (argc < 2) {
        if (rank == 0)
            std::cerr << "Usage: " << argv[0] << " <dim (1 or 2)> [N]" << std::endl;
        MPI_Finalize();
        return 1;
    }

    int dim = std::stoi(argv[1]);
    bool direct = true;

    if (dim == 1) {
        int N = (argc >= 3 ? std::stoi(argv[2]) : 8);
        using C = std::complex<double>;
        std::vector<C> data;
        if (rank == 0) {
            data.resize(N);
            for (int i = 0; i < N; ++i)
                data[i] = C(static_cast<double>(i), 0.0);
        }
        MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (rank != 0) data.resize(N);
        MPI_Bcast(data.data(), N, MPI_CXX_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);

        // Iterative FFT
        IterativeFourier<std::vector<C>> iterative;
        std::vector<C> iterResult;
        iterative.compute(data, iterResult, direct);

        // MPI Parallel FFT
        MPIParallelFourier<std::vector<C>> parallel;
        std::vector<C> parResult;
        parallel.compute(data, parResult, direct);

        if (rank == 0) {
            std::cout << "1D FFT Iterative result:" << std::endl;
            for (const auto &c : iterResult) std::cout << c << std::endl;
            std::cout << "1D FFT MPI-Parallel result:" << std::endl;
            for (const auto &c : parResult) std::cout << c << std::endl;

            bool same = approxEqual(iterResult, parResult);
            std::cout << "Outputs are " << (same ? "consistent" : "different") << std::endl;
        }
        iterative.executionTime();
        parallel.executionTime();

    } else if (dim == 2) {
        int N = (argc >= 3 ? std::stoi(argv[2]) : 4);
        using C = std::complex<double>;
        std::vector<C> flat;
        if (rank == 0) {
            flat.resize(N * N);
            for (int i = 0; i < N; ++i)
                for (int j = 0; j < N; ++j)
                    flat[i * N + j] = C(static_cast<double>(i), static_cast<double>(j));
        }
        MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (rank != 0) flat.resize(N * N);
        MPI_Bcast(flat.data(), N * N, MPI_CXX_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);

        std::vector<std::vector<C>> data(N, std::vector<C>(N));
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                data[i][j] = flat[i * N + j];

        // Iterative 2D FFT
        IterativeFourier<std::vector<std::vector<C>>> iterative2;
        std::vector<std::vector<C>> iterMat;
        iterative2.compute(data, iterMat, direct);

        // MPI Parallel 2D FFT
        MPIParallelFourier<std::vector<std::vector<C>>> parallel2;
        std::vector<std::vector<C>> parMat;
        parallel2.compute(data, parMat, direct);

        if (rank == 0) {
            std::cout << "2D FFT Iterative result:" << std::endl;
            for (const auto &row : iterMat) {
                for (const auto &c : row) std::cout << c << " ";
                std::cout << std::endl;
            }
            std::cout << "2D FFT MPI-Parallel result:" << std::endl;
            for (const auto &row : parMat) {
                for (const auto &c : row) std::cout << c << " ";
                std::cout << std::endl;
            }

            bool same = approxEqual(iterMat, parMat);
            std::cout << "Outputs are " << (same ? "consistent" : "different") << std::endl;
        }
        iterative2.executionTime();
        parallel2.executionTime();

    } else {
        if (rank == 0)
            std::cerr << "Invalid dimension: use 1 or 2" << std::endl;
    }

    MPI_Finalize();
    return 0;
}