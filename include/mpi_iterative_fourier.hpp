#ifndef FFT_MPI_ITERATIVE_FOURIER_HPP
#define FFT_MPI_ITERATIVE_FOURIER_HPP

#include <abstract_transform.hpp>
#include <mpi_wrapper.hpp>
#include <cmath>

template <typename T>
class MPIIterativeFourier final : public BaseTransform<T> {
    bool direct{};  // only used for printing execution time
    MPIWrapper mpi_; // MPI wrapper instance

    // Helper method for 1D FFT computation with MPI
    template <ComplexVector VectorType>
    void compute1D(const VectorType &input, VectorType &output, bool isDirect) {
        int n = input.size();
        int m = static_cast<int>(log2(n));

        // Resize output on all processes
        output.resize(n);

        // Calculate local workload for each process
        int chunk_size = n / mpi_.size();
        int start_idx = mpi_.rank() * chunk_size;
        int end_idx = (mpi_.rank() == mpi_.size() - 1) ? n : (mpi_.rank() + 1) * chunk_size;
        int local_n = end_idx - start_idx;

        // Local storage for computation
        VectorType local_output(local_n);

        // Distribute data from root to all processes
        mpi_.broadcast(const_cast<typename VectorType::value_type*>(input.data()), n);

        // Local bit-reversal permutation
        for (int i = start_idx; i < end_idx; i++) {
            int j = 0;
            for (int k = 0; k < m; k++) {
                if (i & (1 << k))
                    j |= (1 << (m - 1 - k));
            }
            if (j < n) {
                local_output[i - start_idx] = input[j];
            }
        }

        // Gather results from all processes
        mpi_.allGather(
            local_output.data(),
            local_n * sizeof(typename VectorType::value_type),
            output.data(),
            chunk_size * sizeof(typename VectorType::value_type)
        );

        // Iterative Cooley-Tukey FFT - executed by all processes on distributed data
        for (int s = 1; s <= m; s++) {
            int m_s = 1 << s; // 2^s
            typename VectorType::value_type wm{cos(2 * M_PI / m_s), (isDirect ? -1.0 : 1.0) * sin(2 * M_PI / m_s)};

            for (int k = start_idx; k < end_idx; k += m_s) {
                typename VectorType::value_type w{1, 0};
                for (int j = 0; j < m_s/2; j++) {
                    if (k + j >= n || k + j + m_s/2 >= n) continue;

                    typename VectorType::value_type t = w * output[k + j + m_s/2];
                    typename VectorType::value_type u = output[k + j];
                    output[k + j] = u + t;
                    output[k + j + m_s/2] = u - t;
                    w = w * wm;
                }
            }

            // Synchronize all processes after each stage
            mpi_.allGather(
                output.data() + start_idx,
                local_n * sizeof(typename VectorType::value_type),
                output.data(),
                chunk_size * sizeof(typename VectorType::value_type)
            );
        }

        // Normalize inverse output
        if (!isDirect && std::is_same_v<VectorType, T>) {
            for (int i = start_idx; i < end_idx; i++) {
                output[i] /= n;
            }

            // Ensure all processes have normalized data
            mpi_.allGather(
                output.data() + start_idx,
                local_n * sizeof(typename VectorType::value_type),
                output.data(),
                chunk_size * sizeof(typename VectorType::value_type)
            );
        }
    }

    void computeImpl(const T &input, T &output, const bool& isDirect) override {
        direct = isDirect;

        // 1D FFT implementation
        if constexpr (ComplexVector<T>) {
            compute1D(input, output, isDirect);
        }
        // 2D FFT implementation using row-column decomposition
        else if constexpr (ComplexVectorMatrix<T>) {
            int rows = input.size();
            if (rows == 0) return;
            int cols = input[0].size();

            output.resize(rows);
            for (auto& row : output) {
                row.resize(cols);
            }

            // Step 1: Apply FFT to each row in parallel
            T row_fft(rows);
            for (int i = 0; i < rows; ++i) {
                if (i % mpi_.size() == mpi_.rank()) {
                    row_fft[i].resize(cols);
                    compute1D(input[i], row_fft[i], isDirect);
                }
            }

            // Gather all row FFT results
            for (int i = 0; i < rows; i++) {
                int owner = i % mpi_.size();
                if (mpi_.rank() == owner) {
                    for (int r = 0; r < mpi_.size(); r++) {
                        if (r != mpi_.rank()) {
                            mpi_.send(row_fft[i].data(), cols * sizeof(typename T::value_type::value_type), r, i);
                        }
                    }
                } else {
                    row_fft[i].resize(cols);
                    mpi_.recv(row_fft[i].data(), cols * sizeof(typename T::value_type::value_type), owner, i);
                }
            }

            // Step 2: Transpose the matrix
            T transposed = this->transpose2D(row_fft);

            // Step 3: Apply FFT to each column (now rows in transposed matrix)
            T col_fft(cols);
            for (int i = 0; i < cols; ++i) {
                if (i % mpi_.size() == mpi_.rank()) {
                    col_fft[i].resize(rows);
                    compute1D(transposed[i], col_fft[i], isDirect);
                }
            }

            // Gather all column FFT results
            for (int i = 0; i < cols; i++) {
                int owner = i % mpi_.size();
                if (mpi_.rank() == owner) {
                    for (int r = 0; r < mpi_.size(); r++) {
                        if (r != mpi_.rank()) {
                            mpi_.send(col_fft[i].data(), rows * sizeof(typename T::value_type::value_type), r, i);
                        }
                    }
                } else {
                    col_fft[i].resize(rows);
                    mpi_.recv(col_fft[i].data(), rows * sizeof(typename T::value_type::value_type), owner, i);
                }
            }

            // Step 4: Transpose back
            output = this->transpose2D(col_fft);

            // Step 5: Normalize for inverse FFT
            if (!isDirect) {
                double factor = static_cast<double>(rows * cols);
                // Distribute normalization across processes
                for (int i = 0; i < rows; i++) {
                    if (i % mpi_.size() == mpi_.rank()) {
                        for (auto &val : output[i]) {
                            val /= factor;
                        }
                    }
                }

                // Share normalized data
                for (int i = 0; i < rows; i++) {
                    int owner = i % mpi_.size();
                    if (mpi_.rank() == owner) {
                        for (int r = 0; r < mpi_.size(); r++) {
                            if (r != mpi_.rank()) {
                                mpi_.send(output[i].data(), cols * sizeof(typename T::value_type::value_type), r, i);
                            }
                        }
                    } else {
                        mpi_.recv(output[i].data(), cols * sizeof(typename T::value_type::value_type), owner, i);
                    }
                }
            }
        }
    }

public:
    MPIIterativeFourier() = default;

    void executionTime() const override {
        if (mpi_.isRoot()) {  // Only root process reports time
            std::cout << "MPI Iterative " << (direct ? "Direct" : "Inverse") << " ";

            if constexpr (ComplexVector<T>) {
                std::cout << "1D";
            } else if constexpr (ComplexVectorMatrix<T>) {
                std::cout << "2D";
            }

            std::cout << " FFT time: " << this->time.count() << " seconds"
                      << " (using " << mpi_.size() << " processes)" << std::endl;
        }
    }
};

#endif // FFT_MPI_ITERATIVE_FOURIER_HPP