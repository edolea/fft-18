#ifndef FFT_ITERATIVE_FOURIER_HPP
#define FFT_ITERATIVE_FOURIER_HPP

#include <abstract_transform.hpp>


template <typename T>
class IterativeFourier : public BaseTransform<T> {

public:
    explicit IterativeFourier(const T& input) : BaseTransform<T>(input) {}

    void compute(const T& input) override {
        this->input = input;
        this->output = computation(input);
    }

    void compute() override {
        this->output = computation(this->input);
    }

private:
    doubleVector computation(const T& input) {
        int n = input.size();
        int m = log2(n);
        std::vector<std::complex<double>> y(n); // Must a power of 2

        // Bit-reversal permutation
        for (int i = 0; i < n; i++)
        {
            int j = 0;
            for (int k = 0; k < m; k++)
            {
                if (i & (1 << k))
                {
                    j |= (1 << (m - 1 - k));
                }
            }
            y[j] = input[i];
        }
        // Iterative FFT
        for (int j = 1; j <= m; j++)
        {
            int d = 1 << j;
            std::complex<double> w(1, 0);
            std::complex<double> wd(std::cos(2 * M_PI / d), std::sin(2 * M_PI / d));
            for (int k = 0; k < d / 2; k++)
            {
                for (int m = k; m < n; m += d)
                {
                    std::complex<double> t = w * y[m + d / 2];
                    std::complex<double> x = y[m];
                    y[m] = x + t;
                    y[m + d / 2] = x - t;
                }
                w = w * wd;
            }
        }
        return y;
    }
};


#endif //FFT_ITERATIVE_FOURIER_HPP
