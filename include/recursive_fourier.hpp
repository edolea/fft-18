#ifndef FFT_RECURSIVE_FOURIER_HPP
#define FFT_RECURSIVE_FOURIER_HPP

#include "abstract_transform.hpp"

template <typename T>
class RecursiveFourier final : public BaseTransform<T> {
    bool direct{};

    void computeDirect(const T &input, T &output) override
    {
        direct = true;
        output = algorithm(input);
    }

    void computeInverse(const T &input, T &output) override
    {
        direct = false;
        output = algorithm(input);

        // Normalize the output
        for (auto &val : output)
            val /= input.size();
    }

    T algorithm(const T &x)
    {
        if (x.size() == 1)
            return x;

        int n = x.size();

        typename T::value_type wn;
        if (direct)
        {
            wn = typename T::value_type{std::cos(2 * M_PI / n), std::sin(2 * M_PI / n)};
        }
        else
        {
            wn = typename T::value_type{std::cos(2 * M_PI / n), -std::sin(2 * M_PI / n)};
        }

        typename T::value_type w{1.0, 0.0};

        T x_even, x_odd;
        for (int i = 0; i < n; i++)
        {
            if (i % 2 == 0)
                x_even.push_back(x[i]);
            else
                x_odd.push_back(x[i]);
        }

        T y_even = algorithm(x_even);
        T y_odd = algorithm(x_odd);

        T y(n);
        for (int i = 0; i < n / 2; i++)
        {
            y[i] = y_even[i] + w * y_odd[i];
            y[i + n / 2] = y_even[i] - w * y_odd[i];
            w = w * wn;
        }

        return y;
    }

public:
    void executionTime() const override
    {
        std::cout << "Recursive FFT time: " << this->time.count() << " seconds" << std::endl;
    }
};

#endif // FFT_RECURSIVE_FOURIER_HPP
