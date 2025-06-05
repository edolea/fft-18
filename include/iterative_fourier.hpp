#ifndef FFT_ITERATIVE_FOURIER_HPP
#define FFT_ITERATIVE_FOURIER_HPP

#include <abstract_transform.hpp>

template <typename T>
class IterativeFourier final : public BaseTransform<T> {
    bool direct{};

    void computeImpl(const T &input, T &output, const bool& isDirect) override{
        direct = isDirect;
        int n = input.size();
        int m = static_cast<int>(log2(n));
        T y(n);

        // Bit-reversal permutation
        for (int i = 0; i < n; i++) {
            int j = 0;
            for (int k = 0; k < m; k++)
                if (i & (1 << k))
                    j |= (1 << (m - 1 - k));
            y[j] = input[i];
        }

        // Iterative Cooley-Tukey FFT
        for (int j = 1; j <= m; j++) {
            int d{1 << j};
            typename T::value_type wn{std::cos(2 * M_PI / d), (direct ? 1.0 : -1.0) * std::sin(2 * M_PI / d)};

            for (int k = 0; k < n; k += d) {
                typename T::value_type w{1, 0};
                for (int i = 0; i < d / 2; i++) {
                    typename T::value_type t = w * y[k + i + d / 2];
                    typename T::value_type u = y[k + i];
                    y[k + i] = u + t;
                    y[k + i + d / 2] = u - t;
                    w = w * wn;
                }
            }
        }

        output = std::move(y);

        // Normalize inverse output
        if (!isDirect)
            for (auto &val : output)
                val /= input.size();
    }

public:
    void executionTime() const override {
        std::cout << "Iterative " << (direct ? "Direct" : "Inverse") << " FFT time: " << this->time.count() << " seconds" << std::endl;
    }
};

#endif // FFT_ITERATIVE_FOURIER_HPP
