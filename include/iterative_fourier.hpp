#ifndef FFT_ITERATIVE_FOURIER_HPP
#define FFT_ITERATIVE_FOURIER_HPP

#include <abstract_transform.hpp>


template <typename T>
class IterativeFourier : public BaseTransform<T> {

public:
    explicit IterativeFourier(const T& input) : BaseTransform<T>(input) {}

    void compute() override {
        this->output = computation(this->input);
    }

private:
    T computation(const T& input) const {
        int n = input.size();
        int m = static_cast<int>(log2(n));
        T y(n); // Must a power of 2

        // Bit-reversal permutation
        for (int i = 0; i < n; i++) {
            int j = 0;
            for (int k = 0; k < m; k++)
                if (i & (1 << k))
                    j |= (1 << (m - 1 - k));
            y[j] = input[i];
        }
        // Iterative FFT
        for (int j = 1; j <= m; j++)
        {
            int d = 1 << j;
            typename T::value_type w{1, 0};
            typename T::value_type wd{std::cos(2 * M_PI / d), std::sin(2 * M_PI / d)};

            for (int k = 0; k < d / 2; k++) {
                for (int m = k; m < n; m += d) {
                    typename T::value_type t{w * y[m + d / 2]};
                    typename T::value_type x{y[m]};
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
