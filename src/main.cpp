#include <iostream>
#include <unistd.h>

#include "vector_generator.hpp"
#include "recursive_fourier.hpp"
#include "iterative_fourier.hpp"
#include "parameter_class.hpp"
void vector_print(const doubleVector &result);

int ParameterClass::N = 0;
double ParameterClass::frequency = 0.0;
double ParameterClass::amplitude = 0.0;

int main()
{
    ParameterClass::initializeParameters(1024, 0.5, 1.0);

    const auto input = RandomVectorGenerator::generate<doubleVector>(8);
    doubleVector output;
    doubleVector output2;
    doubleVector output3;
    doubleVector output4;

    RecursiveFourier<doubleVector> recursiveFourier;
    IterativeFourier<doubleVector> iterativeFourier;

    recursiveFourier.compute(input, output);
    iterativeFourier.compute(input, output2);

    std::cout << "result recursive" << std::endl;
    vector_print(output);
    std::cout << "\nresult iterative" << std::endl;
    vector_print(output2);

    recursiveFourier.executionTime();
    iterativeFourier.executionTime();

    recursiveFourier.compute(input, output3, false);
    iterativeFourier.compute(input, output4, false);


    std::cout << "\n\nresult inverse recursive" << std::endl;
    vector_print(output3);
    std::cout << "\nresult inverse iterative" << std::endl;
    vector_print(output4);

    recursiveFourier.executionTime();
    iterativeFourier.executionTime();
}

void vector_print(const doubleVector &result)
{
    for (auto i : result)
        std::cout << i << std::endl;
}
