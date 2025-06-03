#include <iostream>
#include <thread>
#include "vector_generator_gpu.hpp"
#include "parallel_fourier.hpp"
#include "parameter_class.hpp"

int ParameterClass::N = 0;
double ParameterClass::frequency = 0.0;
double ParameterClass::amplitude = 0.0;

int main()
{
    ParameterClass::initializeParameters(1024, 0.5, 1.0);
    const auto vec = RandomVectorGenerator::generate<doubleVector>(ParameterClass::N);
    ParallelFourier<doubleVector> parallelFourier(vec);
    parallelFourier.compute();

    return 0;
}