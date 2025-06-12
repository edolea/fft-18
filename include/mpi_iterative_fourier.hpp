#ifndef MPI_ITERATIVE_FOURIER_HPP
#define MPI_ITERATIVE_FOURIER_HPP

#include "abstract_transform.hpp"
#include <mpi.h>

template <typename T>
class MpiIterativeFourier final : public BaseTransform<T> {
    bool direct{};  // only used for printing execution time

    int size;
    int rank;
    MPI_Comm comm;

    // Helper method for 1D FFT computation
    template <ComplexVector VectorType>
    void compute1D(const VectorType &input, VectorType &output, bool isDirect) {
        int n = input.size();
        int m = static_cast<int>(log2(n));
        output.resize(n);

        // Bit-reversal permutation
        for (int i = 0; i < n; i++) {
            int j = 0;
            for (int k = 0; k < m; k++)
                if (i & (1 << k))
                    j |= (1 << (m - 1 - k));
            output[j] = input[i];
        }

        // Iterative Cooley-Tukey FFT
        for (int j = 1; j <= m; j++) {
            int d{1 << j}; // 2^j
            typename VectorType::value_type wn{std::cos(2 * M_PI / d), (isDirect ? 1.0 : -1.0) * std::sin(2 * M_PI / d)};

            for (int k = 0; k < n; k += d) {
                typename VectorType::value_type w{1, 0};
                for (int i = 0; i < d / 2; i++) {
                    typename VectorType::value_type t = w * output[k + i + d / 2];
                    typename VectorType::value_type u = output[k + i];
                    output[k + i] = u + t;
                    output[k + i + d / 2] = u - t;
                    w = w * wn;
                }
            }
        }

        // Normalize inverse output (only for 1D case - 2D case is handled separately)
        if (!isDirect && std::is_same_v<VectorType, T>)
            for (auto &val : output)
                val /= n;
    }

    void compute_mpi_1d_(const T &input, T &output, const bool &isDirect) {
        direct = isDirect;

        // Get input size and validate it's a power of 2
        int global_n;
        if (rank == 0)
            global_n = input.size();
        MPI_Bcast(&global_n, 1, MPI_INT, 0, comm);

        const int m = static_cast<int>(log2(global_n));

        // Calculate local data size
        const int local_n = global_n / size;

        // Ensure output is properly sized on rank 0
        if (rank == 0) {
            output.resize(global_n);
        }

        // Local storage
        T local_output(local_n);
        T permuted_data;

        // Only rank 0 performs bit-reversal permutation
        if (rank == 0) {
            permuted_data.resize(global_n);

            // Bit-reversal permutation - sequential on rank 0
            for (int i = 0; i < global_n; i++) {
                int j = 0;
                for (int k = 0; k < m; k++)
                    if (i & (1 << k))
                        j |= (1 << (m - 1 - k));
                permuted_data[j] = input[i];
            }
        }

        // Scatter the permuted data to all processes
        MPI_Scatter(rank == 0 ? permuted_data.data() : nullptr,
                   local_n, MPI_DOUBLE_COMPLEX,
                   local_output.data(), local_n, MPI_DOUBLE_COMPLEX,
                   0, comm);

        // Parallel Cooley-Tukey FFT implementation
        for (int j = 1; j <= m; j++) {
            int d = 1 << j;  // 2^j
            typename T::value_type wn{std::cos(2 * M_PI / d),
                                     (isDirect ? 1.0 : -1.0) * std::sin(2 * M_PI / d)};

            // Handle stages where butterfly operations are within local blocks
            if (d <= local_n) {
                // All butterfly operations are local - no communication needed
                for (int k = 0; k < local_n; k += d) {
                    typename T::value_type w{1, 0};
                    for (int i = 0; i < d/2; i++) {
                        typename T::value_type t = w * local_output[k + i + d/2];
                        typename T::value_type u = local_output[k + i];
                        local_output[k + i] = u + t;
                        local_output[k + i + d/2] = u - t;
                        w = w * wn;
                    }
                }
            }
            // Handle stages where butterfly operations cross process boundaries
            else {
                // Inter-process butterfly operations require communication
                T full_data;
                if (rank == 0) {
                    full_data.resize(global_n);
                }

                // Gather all data to rank 0
                MPI_Gather(local_output.data(), local_n, MPI_DOUBLE_COMPLEX,
                          rank == 0 ? full_data.data() : nullptr,
                          local_n, MPI_DOUBLE_COMPLEX,
                          0, comm);

                // Rank 0 performs the computation for large butterfly stages
                if (rank == 0) {
                    for (int k = 0; k < global_n; k += d) {
                        typename T::value_type w{1, 0};
                        for (int i = 0; i < d/2; i++) {
                            typename T::value_type t = w * full_data[k + i + d/2];
                            typename T::value_type u = full_data[k + i];
                            full_data[k + i] = u + t;
                            full_data[k + i + d/2] = u - t;
                            w = w * wn;
                        }
                    }
                }

                // Scatter results back to all processes
                MPI_Scatter(rank == 0 ? full_data.data() : nullptr,
                           local_n, MPI_DOUBLE_COMPLEX,
                           local_output.data(), local_n, MPI_DOUBLE_COMPLEX,
                           0, comm);
            }
        }

        // Normalize for inverse transform
        if (!isDirect) {
            for (auto &val : local_output)
                val /= global_n;
        }

        // Gather final results back to root process
        MPI_Gather(local_output.data(), local_n, MPI_DOUBLE_COMPLEX,
                  output.data(), local_n, MPI_DOUBLE_COMPLEX,
                  0, comm);
    }

    void computeImpl(const T &input, T &output, const bool& isDirect) override {
        direct = isDirect;

        // 1D FFT implementation
        if constexpr (ComplexVector<T>) {
            compute_mpi_1d_(input, output, isDirect);
        }
        // 2D FFT implementation using row-column decomposition
        else if constexpr (ComplexVectorMatrix<T>) {
            int rows = input.size();
            if (rows == 0) return;

            int cols = input[0].size();

            // Step 1: Apply FFT to each row
            T row_fft(rows);
            for (int i = 0; i < rows; ++i)
                compute1D(input[i], row_fft[i], isDirect);

            // Step 2: Transpose
            //T transposed = this->transpose2D(row_fft);
            this->transpose2D_more_efficient(row_fft);

            // Step 3: Apply FFT to each (now transposed) row == original columns
            T col_fft(row_fft.size());
            for (size_t i = 0; i < row_fft.size(); ++i)
                compute1D(row_fft[i], col_fft[i], isDirect);

            // Step 4: Transpose back
            // TODO: used more efficient transpose here also
            output = this->transpose2D(col_fft);

            // Step 5: Normalize for inverse FFT
            if (!isDirect) {
                for (auto &row : output)
                    for (auto &val : row)
                        val /= static_cast<double>(rows * cols);
            }
        }

    }

public:
    MpiIterativeFourier(MPI_Comm communicator) : comm(communicator) {
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &size);
    }

    void compute(const T &input, T &output, const bool& direct=true) {
        const auto start = std::chrono::high_resolution_clock::now();
        computeImpl(input, output, direct);
        const auto end = std::chrono::high_resolution_clock::now();
        this->time = end - start;
    }

    void executionTime() const override {
        if (rank == 0) {
            std::cout << "MPI " << (direct ? "Direct" : "Inverse") << " ";

            if constexpr (ComplexVector<T>) {
                std::cout << "1D";
            } else if constexpr (ComplexVectorMatrix<T>) {
                std::cout << "2D";
            }

            std::cout << " FFT time: " << this->time.count() << " seconds with " << size << " processors" << std::endl;
        }
    }
};

#endif //MPI_ITERATIVE_FOURIER_HPP