#include "fourier_transform.hpp"
#include <vector>
#include <complex>

using floatV = std::vector<std::complex<float>>;
using doubleV = std::vector<std::complex<double>>;

int main(){
    const doubleV vec = { {10, 2}, {5, 4} };

    FourierTransform1D<doubleV> fft(vec);
    fft.compute();

}