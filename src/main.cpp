#include <vector>
#include <complex>
#include <iostream>
#include "vector_generator.hpp"
#include "recursive_fourier.hpp"
#include "iterative_fourier.hpp"

void vector_print(const doubleVector& result);

using floatVector = std::vector<std::complex<float>>;
using doubleVector = std::vector<std::complex<double>>;

int main(){
    const doubleVector vec = {{10, 2}, {5, 4} };
    const auto vec2 = RandomVectorGenerator::generate<doubleVector>(8);

    RecursiveFft<doubleVector> fft(vec2);
    IterativeFourier<doubleVector> iterativeFourier(vec2);
    fft.compute();
    iterativeFourier.compute();

    // here ComplexVector is redundant, but still a nice double check for output type
    ComplexVector auto result = fft.getOutput();
    ComplexVector auto result_iter = iterativeFourier.getOutput();

    std::cout << "result recursive" << std::endl;
    vector_print(result);
    std::cout << "\nresult iterative" << std::endl;
    vector_print(result_iter);


}

void vector_print(const doubleVector& result) {
    for (auto i : result){
        std::cout << i << std::endl;
    }
}
