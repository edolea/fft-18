#ifndef MPI_ITERATIVE_FOURIER_HPP
#define MPI_ITERATIVE_FOURIER_HPP

#include "abstract_transform.hpp"
#include <mpi.h>
#include <algorithm>

template <typename T>
requires ComplexVector<T> || ComplexVectorMatrix<T>
class MpiIterativeFourier final : public BaseTransform<T> {
    bool direct{};  // only used for printing execution time

    int size;
    int rank;
    MPI_Comm comm;

    // Convert a flattened 1D buffer back to a 2D matrix
    void unflattenBuffer(const typename T::value_type& flatBuffer, T& matrix,
                        int numRows, int numCols) {
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < numRows; i++)
            std::copy(flatBuffer.begin() + i * numCols,
                     flatBuffer.begin() + (i + 1) * numCols,
                     matrix[i].begin());
    }

    // Convert a 2D matrix to a flattened 1D buffer
    void flattenBuffer(const T& matrix, typename T::value_type& flatBuffer,
                      int numRows, int numCols) {
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < numRows; i++)
            std::copy(matrix[i].begin(), matrix[i].end(),
                     flatBuffer.begin() + i * numCols);
    }

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
                if (i & 1 << k) // checks if k-th bit is 1
                    j |= (1 << (m - 1 - k));
            output[j] = input[i];
        }

        // Iterative Cooley-Tukey FFT
        for (int j = 1; j <= m; j++) {
            int d{1 << j}; // 2^j
            typename VectorType::value_type wn{std::cos(2 * M_PI / d), (isDirect ? 1.0 : -1.0) * std::sin(2 * M_PI / d)};

            // #pragma omp parallel for collapse(2) schedule(static, d/2)
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
    }

    // Transpose function but for mpi
    void transpose2D_more_efficient(T &input) {
        if (input.empty() || input.size() != input[0].size()) return; // Ensure square matrix

        const size_t n = input.size();
        #pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < n; ++i)
            for (size_t j = i + 1; j < n; ++j)
                std::swap(input[i][j], input[j][i]);
    }

    // MPI 1D
    void compute_mpi_1d_(const T &input, T &output, const bool &isDirect) {
        int global_n;
        if (rank == 0)
            global_n = input.size();
        MPI_Bcast(&global_n, 1, MPI_INT, 0, comm);

        const int m = static_cast<int>(log2(global_n));

        // Calculate local data size
        const int local_n = global_n / size;

        // Ensure output is properly sized on rank 0
        if (rank == 0)
            output.resize(global_n);

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
                #pragma omp parallel for schedule(static)
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
                    #pragma omp parallel for
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
        // 1D FFT implementation
        if constexpr (ComplexVector<T>) {
            compute_mpi_1d_(input, output, isDirect);
        }
        // 2D FFT implementation using row-column decomposition
        else if constexpr (ComplexVectorMatrix<T>) {
            int rows, cols;
            if (rank == 0) {
                rows = input.size();
                cols = input[0].size();
            }
            MPI_Bcast(&rows, 1, MPI_INT, 0, comm);
            MPI_Bcast(&cols, 1, MPI_INT, 0, comm);

            // local row count for each process -- for now both are power of 2
            int local_rows = rows / size; // --> local_rows_pre
            // int remainder = rows % size;
            // int local_rows = local_rows_pre + (rank < remainder ? 1 : 0);

            // Local storage for assigned rows
            T local_input(local_rows, typename T::value_type(cols));
            T local_output(local_rows, typename T::value_type(cols));

            // Local buffers for all ranks
            typename T::value_type recBuff(local_rows * cols);
            // typename T::value_type sendBuff_notRoot(local_rows * cols);
            typename T::value_type sendBuff(rows * cols);

            // Step 0: Scatter rows to processes
            if (rank == 0)
                flattenBuffer(input, sendBuff, rows, cols);
            MPI_Scatter(rank == 0 ? sendBuff.data() : nullptr, rank == 0 ? local_rows * cols : 0, MPI_DOUBLE_COMPLEX,
                recBuff.data(), local_rows * cols, MPI_DOUBLE_COMPLEX, 0, comm);

            // Step 0b: unflatten rec
            unflattenBuffer(recBuff, local_input, local_rows, cols);

            // Step 1: Apply FFT to each row
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < local_rows; ++i)
                compute1D(local_input[i], local_output[i], isDirect);

            // Step 1.5: Gather results back to root
            T row_fft;
            if (rank == 0) {
                row_fft.resize(rows);
                #pragma omp parallel for schedule(static)
                for (auto& row : row_fft)
                    row.resize(cols);
            }

            flattenBuffer(local_output, sendBuff, local_rows, cols);
            typename T::value_type recvbuf(rank == 0 ? rows * cols : 0);
            MPI_Gather(sendBuff.data(), local_rows * cols, MPI_DOUBLE_COMPLEX,
                      rank == 0 ? recvbuf.data() : nullptr, rank == 0 ? local_rows * cols : 0, MPI_DOUBLE_COMPLEX,
                      0, comm);

            // Step 1.5b: Rank 0 reshapes gathered data
            if (rank == 0) {
                unflattenBuffer(recvbuf, row_fft, rows, cols);

                // Step 2: Transpose on root
                transpose2D_more_efficient(row_fft);
            }

            // Step 2.5: scatter again
            // typename T::value_type sendBuff(rows * cols);
            if (rank == 0)
                flattenBuffer(row_fft, sendBuff, rows, cols);
            MPI_Scatter(rank == 0 ? sendBuff.data() : nullptr, rank == 0 ? local_rows * cols : 0, MPI_DOUBLE_COMPLEX,
            recBuff.data(), local_rows * cols, MPI_DOUBLE_COMPLEX, 0, comm);

            unflattenBuffer(recBuff, local_input, local_rows, cols);

            // Step 3: Apply FFT to each (now transposed) row == original columns
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < local_rows; ++i)
                compute1D(local_input[i], local_output[i], isDirect);

            // Step 3.5: Gather back to root again
            output.clear();
            if (rank == 0) {
                output.resize(rows);

                #pragma omp parallel for schedule(static)
                for (auto& row : output)
                    row.resize(cols);
            }

            flattenBuffer(local_output, sendBuff, local_rows, cols);

            // Step 5: Normalize for inverse FFT
            if (!isDirect) {
                #pragma omp parallel for schedule(static)
                for (auto& val : sendBuff)
                    val /= static_cast<typename T::value_type::value_type::value_type>(local_rows * cols);

                // NB: normalizing on buffer is better bc all the vector inside contiguous heap memory block
                //      with matrix each row would be on different heap --> worse threads performance
            }

            MPI_Gather(sendBuff.data(), local_rows * cols, MPI_DOUBLE_COMPLEX,
                      rank == 0 ? recvbuf.data() : nullptr, local_rows * cols, MPI_DOUBLE_COMPLEX,
                      0, comm);

            // Step 3.5b: reshape on root again
            if (rank == 0) {
                // Reshape gathered data into output
                unflattenBuffer(recvbuf, output, rows, cols);

                // Step 4: Transpose back on root
                transpose2D_more_efficient(output);
            }
        }
    }

public:
    explicit MpiIterativeFourier(MPI_Comm communicator) : comm(communicator) {
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &size);
    }

    void compute(const T &input, T &output, const bool& direct=true) {
        this->direct = direct;
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