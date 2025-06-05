#ifndef FFT_RECURSIVE_FOURIER_HPP
#define FFT_RECURSIVE_FOURIER_HPP

#include "abstract_transform.hpp"

template <typename T>
class RecursiveFourier final : public BaseTransform<T> {
    bool direct{};

    void computeImpl(const T &input, T &output, const bool &isDirect) override {
        direct = isDirect;
        output = algorithm(input,isDirect);

        // Normalize the output
        if (!isDirect)
            for (auto &val : output)
                val /= input.size();
    }

    T algorithm(const T &x, const bool& isDirect) {
        if (x.size() == 1)
            return x;

        int n = x.size();

        typename T::value_type wn{std::cos(2 * M_PI / n), (direct ? 1.0 : -1.0) * std::sin(2 * M_PI / n)};
        typename T::value_type w{1.0, 0.0};

        T x_even, x_odd;
        for (int i = 0; i < n; i++) {
            if (i % 2 == 0)
                x_even.push_back(x[i]);
            else
                x_odd.push_back(x[i]);
        }

        T y_even = algorithm(x_even, isDirect);
        T y_odd = algorithm(x_odd, isDirect);

        T y(n);
        for (int i = 0; i < n / 2; i++) {
            y[i] = y_even[i] + w * y_odd[i];
            y[i + n / 2] = y_even[i] - w * y_odd[i];
            w = w * wn;
        }

        return y;
    }

public:
    void executionTime() const override {
        std::cout << "Recursive " << (direct ? "Direct" : "Inverse") << " FFT time: " << this->time.count() << " seconds" << std::endl;
    }
};

#endif // FFT_RECURSIVE_FOURIER_HPP
