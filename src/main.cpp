#include <vector>
#include <complex>
#include <iostream>
#include "vector_generator.hpp"
#include "recursive_fourier"

void vector_print(const doubleVector& result);

using floatVector = std::vector<std::complex<float>>;
using doubleVector = std::vector<std::complex<double>>;

int main(){
    const doubleVector vec = {{10, 2}, {5, 4} };
    const auto vec2 = RandomVectorGenerator::generate<doubleVector>(8);

    RecursiveFft<doubleVector> fft(vec);
    RecursiveFft<doubleVector> fft2(vec2);
    fft.compute();
    fft2.compute();

    // here ComplexVector is redundant, but still a nice double check for output type
    ComplexVector auto result = fft.getOutput();
    ComplexVector auto result2 = fft2.getOutput();

    std::cout << "result" << std::endl;
    vector_print(result);
    std::cout << "\nresult2" << std::endl;
    vector_print(result2);


}

void vector_print(const doubleVector& result) {
    // std::cout << "result" << std::endl;
    for (auto i : result){
        std::cout << i << std::endl;
    }
}
