#include <iostream>
#include <thread>
#include "vector_generator.hpp"
#include "recursive_fourier.hpp"
#include "iterative_fourier.hpp"

void vector_print(const doubleVector& result);

int main(){
    const auto vec = RandomVectorGenerator::generate<doubleVector>(8);
    doubleVector output;
    doubleVector output2;

    RecursiveFourier<doubleVector> recursiveFourier;
    IterativeFourier<doubleVector> iterativeFourier;

    recursiveFourier.compute(vec, output);
    iterativeFourier.compute(vec, output2);

    std::cout << "result recursive" << std::endl;
    vector_print(output);
    std::cout << "\nresult iterative" << std::endl;
    vector_print(output2);
}

void vector_print(const doubleVector& result) {
    for (auto i : result){
        std::cout << i << std::endl;
    }
}
