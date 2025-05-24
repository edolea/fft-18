#ifndef FFT_FOURIER_TRANSFORM_1D_HPP
#define FFT_FOURIER_TRANSFORM_1D_HPP

#include "abstract_transform.hpp"

template <ComplexVector T>
class FourierTransform1D : BaseTransform<T>{


public:
    explicit FourierTransform1D(const T& input) : BaseTransform<T>(input) {}

    void compute() override = 0;

private:
    virtual void fft_computation() = 0;
};


#endif //FFT_FOURIER_TRANSFORM_1D_HPP
