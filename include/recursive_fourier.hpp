#ifndef FFT_RECURSIVE_FOURIER_HPP
#define FFT_RECURSIVE_FOURIER_HPP

#include "abstract_transform.hpp"

template <typename T>
class RecursiveFourier final : public BaseTransform<T> {
    bool direct{};

    void computeImpl(const T &input, T &output, const bool &isDirect) override {
        direct = isDirect;

        if constexpr (ComplexVector<T>) {
            // 1D FFT
            output = algorithm1D(input, isDirect);

            // Normalize the output for inverse transform
            if (!isDirect) {
                for (auto &val : output)
                    val /= input.size();
            }
        }
        else if constexpr (ComplexVectorMatrix<T>) {
            int rows = input.size();
            if (rows == 0) return;
            int cols = input[0].size();

            // Step 1: Row-wise FFT
            T row_fft(rows);
            for (int i = 0; i < rows; ++i)
                row_fft[i] = algorithm1D(input[i], isDirect);

            // Step 2: Transpose
            this->transpose2D(row_fft);

            // Step 3: Column-wise FFT (as row-wise on transposed)
            output.resize(cols);
            for (size_t i = 0; i < row_fft.size(); ++i)
                output[i] = algorithm1D(row_fft[i], isDirect);

            // Step 4: Transpose back
            this->transpose2D(output);

            // Step 5: Normalize for inverse FFT
            if (!isDirect) {
                for (auto &row : output)
                    for (auto &val : row)
                        val /= static_cast<double>(rows * cols);
            }
        }
    }

    // Renamed the original algorithm function to algorithm1D for clarity
    template <ComplexVector Vector>
    Vector algorithm1D(const Vector &x, const bool& isDirect) {
        if (x.size() == 1)
            return x;

        int n = x.size();

        typename Vector::value_type wn{std::cos(2 * M_PI / n), (isDirect ? 1.0 : -1.0) * std::sin(2 * M_PI / n)};
        typename Vector::value_type w{1.0, 0.0};

        Vector x_even, x_odd;
        for (int i = 0; i < n; i++) {
            if (i % 2 == 0)
                x_even.push_back(x[i]);
            else
                x_odd.push_back(x[i]);
        }

        Vector y_even = algorithm1D(x_even, isDirect);
        Vector y_odd = algorithm1D(x_odd, isDirect);

        Vector y(n);
        for (int i = 0; i < n / 2; i++) {
            y[i] = y_even[i] + w * y_odd[i];
            y[i + n / 2] = y_even[i] - w * y_odd[i];
            w = w * wn;
        }

        return y;
    }

public:
    void executionTime() const override {
        std::cout << "Recursive " << (direct ? "Direct" : "Inverse") << " ";

        if constexpr (ComplexVector<T>) {
            std::cout << "1D";
        } else if constexpr (ComplexVectorMatrix<T>) {
            std::cout << "2D";
        }

        std::cout << " FFT time: " << this->time.count() << " seconds" << std::endl;
    }
};

#endif // FFT_RECURSIVE_FOURIER_HPP