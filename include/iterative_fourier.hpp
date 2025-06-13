#ifndef FFT_ITERATIVE_FOURIER_HPP
#define FFT_ITERATIVE_FOURIER_HPP

#include <abstract_transform.hpp>

template <typename T>
class IterativeFourier final : public BaseTransform<T> {
    bool direct{};  // only used for printing execution time

    // Helper method for 1D FFT computation
    template <ComplexVector VectorType>
    void compute1D(const VectorType &input, VectorType &output, bool isDirect) {
        const int n = input.size();
        const int m = static_cast<int>(log2(n));
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
            const int d{1 << j}; // 2^j
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
        if constexpr (ComplexVector<T>) {
            compute1D(input, output, isDirect);
        }
        // 2D FFT implementation using row-column decomposition
        else if constexpr (ComplexVectorMatrix<T>) {
            int rows = input.size();
            int cols = input[0].size();

            // Step 1: Apply FFT to each row
            T row_fft(rows);
            for (int i = 0; i < rows; ++i)
                compute1D(input[i], row_fft[i], isDirect);

            // Step 2: Transpose
            this->transpose2D_more_efficient(row_fft);

            // Step 3: Apply FFT to each (now transposed) row == original columns
            output.resize(cols);
            for (size_t i = 0; i < row_fft.size(); ++i)
                compute1D(row_fft[i], output[i], isDirect);

            // Step 4: Transpose back
            this->transpose2D_more_efficient(output);

            // Step 5: Normalize for inverse FFT
            if (!isDirect) {
                for (auto &row : output)
                    for (auto &val : row)
                        val /= static_cast<double>(rows * cols);
            }
        }
    }

public:
    void compute(const T &input, T &output, const bool& direct=true) {
        const auto start = std::chrono::high_resolution_clock::now();
        computeImpl(input, output, direct);
        const auto end = std::chrono::high_resolution_clock::now();
        this->time = end - start;
    }

    void executionTime() const override {
        std::cout << "Iterative " << (direct ? "Direct" : "Inverse") << " ";

        if constexpr (ComplexVector<T>) {
            std::cout << "1D";
        } else if constexpr (ComplexVectorMatrix<T>) {
            std::cout << "2D";
        }

        std::cout << " FFT time: " << this->time.count() << " seconds" << std::endl;
    }
};

#endif // FFT_ITERATIVE_FOURIER_HPP
