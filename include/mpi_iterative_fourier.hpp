#ifndef FFT_MPI_PARALLEL_FOURIER_HPP
#define FFT_MPI_PARALLEL_FOURIER_HPP

#include <mpi.h>
#include "abstract_transform.hpp"
#include <vector>
#include <complex>
#include <cmath>
#include <cassert>

// MPI-based iterative FFT implementation for 1D and 2D complex data
// Implements the BaseTransform interface (computeImpl method)
// Supports power-of-two sizes and assumes the number of processes divides the dimensions.

template <typename T>
requires ComplexVector<T> || ComplexVectorMatrix<T>
class MPIParallelFourier : public BaseTransform<T> {
protected:
    void computeImpl(const T &input, T &output, const bool &direct) override {
        if constexpr (ComplexVector<T>) {
            mpi_fft1D(input, output, direct);
        } else if constexpr (ComplexVectorMatrix<T>) {
            mpi_fft2D(input, output, direct);
        }
    }

private:
    using C = std::complex<double>;
    static constexpr double PI = std::acos(-1);

    void mpi_fft1D(const std::vector<C> &in, std::vector<C> &out, bool direct) {
        int rank, size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        size_t N = in.size();
        assert(N % size == 0);
        size_t chunk = N / size;
        size_t start = rank * chunk;

        double sign = direct ? 1.0 : -1.0;
        double norm = direct ? 1.0 : 1.0 / static_cast<double>(N);
        out.resize(N);

        std::vector<C> localBuf(chunk);
        for (size_t k = 0; k < chunk; ++k) {
            C sum{0.0, 0.0};
            size_t globalK = start + k;
            for (size_t n = 0; n < N; ++n) {
                double angle = sign * 2.0 * PI * static_cast<double>(globalK) * static_cast<double>(n) / static_cast<double>(N);
                sum += in[n] * C(std::cos(angle), std::sin(angle));
            }
            localBuf[k] = sum * norm;
        }

        MPI_Allgather(
            localBuf.data(), static_cast<int>(chunk), MPI_CXX_DOUBLE_COMPLEX,
            out.data(),      static_cast<int>(chunk), MPI_CXX_DOUBLE_COMPLEX,
            MPI_COMM_WORLD
        );
    }

    void mpi_fft2D(const std::vector<std::vector<C>> &in, std::vector<std::vector<C>> &out, bool direct) {
        int rank, size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        size_t N = in.size();
        assert(N % size == 0);
        size_t rowsPerProc = N / size;

        double sign = direct ? -1.0 : 1.0;
        double norm = direct ? 1.0 : 1.0 / static_cast<double>(N);

        // 1) Row-wise FFT
        std::vector<C> localRowBuf(rowsPerProc * N);
        for (size_t i = 0; i < rowsPerProc; ++i) {
            size_t globalRow = rank * rowsPerProc + i;
            for (size_t k = 0; k < N; ++k) {
                C sum{0.0, 0.0};
                for (size_t n = 0; n < N; ++n) {
                    double angle = sign * 2.0 * PI * static_cast<double>(k) * static_cast<double>(n) / static_cast<double>(N);
                    sum += in[globalRow][n] * C(std::cos(angle), std::sin(angle));
                }
                localRowBuf[i * N + k] = sum * norm;
            }
        }

        std::vector<C> allRowBuf(N * N);
        MPI_Allgather(
            localRowBuf.data(), static_cast<int>(rowsPerProc * N), MPI_CXX_DOUBLE_COMPLEX,
            allRowBuf.data(),   static_cast<int>(rowsPerProc * N), MPI_CXX_DOUBLE_COMPLEX,
            MPI_COMM_WORLD
        );

        // Reconstruct full row-transformed matrix
        std::vector<std::vector<C>> rowMat(N, std::vector<C>(N));
        for (size_t i = 0; i < N; ++i)
            for (size_t k = 0; k < N; ++k)
                rowMat[i][k] = allRowBuf[i * N + k];

        // Transpose for column FFT
        auto transposed = this->transpose2D(rowMat);

        // 2) Column-wise FFT (on transposed rows)
        std::vector<C> localColBuf(rowsPerProc * N);
        for (size_t i = 0; i < rowsPerProc; ++i) {
            size_t globalRow = rank * rowsPerProc + i;
            for (size_t k = 0; k < N; ++k) {
                C sum{0.0, 0.0};
                for (size_t n = 0; n < N; ++n) {
                    double angle = sign * 2.0 * PI * static_cast<double>(k) * static_cast<double>(n) / static_cast<double>(N);
                    sum += transposed[globalRow][n] * C(std::cos(angle), std::sin(angle));
                }
                localColBuf[i * N + k] = sum * norm;
            }
        }

        std::vector<C> allColBuf(N * N);
        MPI_Allgather(
            localColBuf.data(), static_cast<int>(rowsPerProc * N), MPI_CXX_DOUBLE_COMPLEX,
            allColBuf.data(),   static_cast<int>(rowsPerProc * N), MPI_CXX_DOUBLE_COMPLEX,
            MPI_COMM_WORLD
        );

        // Reconstruct and transpose back
        std::vector<std::vector<C>> colMat(N, std::vector<C>(N));
        for (size_t i = 0; i < N; ++i)
            for (size_t k = 0; k < N; ++k)
                colMat[i][k] = allColBuf[i * N + k];

        out = this->transpose2D(colMat);
    }
};

#endif // FFT_MPI_PARALLEL_FOURIER_HPP