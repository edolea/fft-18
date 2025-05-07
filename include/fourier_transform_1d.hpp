#ifndef FFT_FOURIER_TRANSFORM_1D_HPP
#define FFT_FOURIER_TRANSFORM_1D_HPP

#include "abstract_transform.hpp"

template <ComplexVector T>
class FourierTransform1D : BaseTransform<T>{


public:
    explicit FourierTransform1D(const T& input) : BaseTransform<T>(input) {}

    // todo: overload left for backward compatibility. Check if to remove at the very end
    void compute(const T &input) override = 0;

    void compute() override = 0;

private:
    virtual void fft_computation() = 0;
};


#endif //FFT_FOURIER_TRANSFORM_1D_HPP
