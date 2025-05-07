#ifndef FFT_FOURIER_TRANSFORM_HPP
#define FFT_FOURIER_TRANSFORM_HPP

#include "abstract_transform.hpp"

template <ComplexVector T>
class FourierTransform1D : BaseTransform<T>{


public:
    explicit FourierTransform1D(const T& input) : BaseTransform<T>(input) {}

    // todo: overload left for backward compatibility. Check if to remove at the very end
    void compute(const T &input) override {

    }

    void compute() override {

    }

private:
    void fft_computation(){

    }
};


#endif //FFT_FOURIER_TRANSFORM_HPP
