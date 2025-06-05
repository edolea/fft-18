#ifndef FFT_ITERATIVE_FOURIER_HPP
#define FFT_ITERATIVE_FOURIER_HPP

#include <abstract_transform.hpp>

template <typename T>
class IterativeFourier final : public BaseTransform<T> {
    bool direct{};  // only used for printing execution time

    // Helper to check if T is a vector of vectors (matrix)
    template <typename U>
    static constexpr bool is_complex_matrix =
        std::is_same_v<U, std::vector<std::vector<std::complex<double>>>> ||
        std::is_same_v<U, std::vector<std::vector<std::complex<float>>>>;

    // Helper to check if T is a vector (1D array)
    template <typename U>
    static constexpr bool is_complex_vector =
        std::is_same_v<U, std::vector<std::complex<double>>> ||
        std::is_same_v<U, std::vector<std::complex<float>>>;

    // Helper method for 1D FFT computation
    template <typename VectorType>
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
            int d{1 << j};
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

    void computeImpl(const T &input, T &output, const bool& isDirect) override {
        direct = isDirect;

        // 1D FFT implementation
        if constexpr (is_complex_vector<T>) {
            compute1D(input, output, isDirect);
        }
        // 2D FFT implementation using row-column decomposition
        else if constexpr (is_complex_matrix<T>) {
            int rows = input.size();
            if (rows == 0) return;

            int cols = input[0].size();
            //output.resize(rows * cols);
            output = input; // Initialize output with input

            // Process rows first
            for (int i = 0; i < rows; ++i) {
                typename T::value_type row_output;
                compute1D(output[i], row_output, isDirect);
                output[i] = row_output;
            }

            // Process columns
            for (int j = 0; j < cols; ++j) {
                // Extract column
                typename T::value_type col_input(rows);
                for (int i = 0; i < rows; ++i) {
                    col_input[i] = output[i][j];
                }

                // FFT on column
                typename T::value_type col_output;
                compute1D(col_input, col_output, isDirect);

                // Place column back
                for (int i = 0; i < rows; ++i) {
                    output[i][j] = col_output[i];
                }
            }

            // Normalize for inverse FFT - for 2D, we need to divide by total number of elements
            if (!isDirect) {
                for (auto &row : output) {
                    for (auto &val : row) {
                        val /= (rows * cols);
                    }
                }
            }
        }
    }

public:
    void executionTime() const override {
        std::cout << "Iterative " << (direct ? "Direct" : "Inverse") << " ";

        if constexpr (is_complex_vector<T>) {
            std::cout << "1D";
        } else if constexpr (is_complex_matrix<T>) {
            std::cout << "2D";
        }

        std::cout << " FFT time: " << this->time.count() << " seconds" << std::endl;
    }
};

#endif // FFT_ITERATIVE_FOURIER_HPP
