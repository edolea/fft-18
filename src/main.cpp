#include <iostream>
#include <thread>
#include "vector_generator.hpp"
#include "recursive_fourier.hpp"
#include "iterative_fourier.hpp"
#include "parameter_class.hpp"
void vector_print(const doubleVector &result);

// already in abstract, here to remember them
// using floatVector = std::vector<std::complex<float>>;
// using doubleVector = std::vector<std::complex<double>>;

int ParameterClass::N = 0;
double ParameterClass::frequency = 0.0;
double ParameterClass::amplitude = 0.0;

int main()
{
    ParameterClass::initializeParameters(1024, 0.5, 1.0);
    const auto vec = RandomVectorGenerator::generate<doubleVector>(ParameterClass::N);

    // RecursiveFft<doubleVector> fft(vec);
    IterativeFourier<doubleVector> iterativeFourier(vec);
    // RecursiveFft fft(vec);
    // fft.compute();
    iterativeFourier.compute();

    // Use polymorphism to call the desired transform
    std::unique_ptr<BaseTransform<doubleVector>> fft;

    // Assign RecursiveFft or IterativeFft
    fft = std::make_unique<RecursiveFourier<doubleVector>>(vec);
    fft->compute();
    ComplexVector auto result = fft->getOutput();

    fft = std::make_unique<IterativeFourier<doubleVector>>(vec);
    fft->compute();
    ComplexVector auto result_iter = fft->getOutput();
    // here ComplexVector is redundant, but still a nice double check for output type

    std::cout << "result recursive" << std::endl;
    vector_print(result);
    std::cout << "\nresult iterative" << std::endl;
    vector_print(result_iter);

    // std::cout << '\n' << std::thread::hardware_concurrency() << '\n';
}

void vector_print(const doubleVector &result)
{
    for (auto i : result)
    {
        std::cout << i << std::endl;
    }
}
