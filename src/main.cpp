#include <iostream>
#include "vector_generator.hpp"
#include "recursive_fourier.hpp"
#include "iterative_fourier.hpp"
#include "parameter_class.hpp"


void vector_print(const doubleVector &result);
void matrix_print(const doubleMatrix &matrix);

int size = 8;
double frequency = 2.0;
double amplitude = 1.0;

int main()
{
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

    const auto input2D = RandomVectorGenerator::generate<doubleMatrix>(size, frequency, amplitude);
    doubleMatrix output2D;

    IterativeFourier<doubleMatrix> iterativeFourier2D;
    iterativeFourier2D.compute(input2D, output2D);
    std::cout << "\n\nresult 2D direct iterative" << std::endl;
    matrix_print(output2D);
}

void vector_print(const doubleVector &result) {
    for (auto i : result)
        std::cout << i << std::endl;
}

void matrix_print(const doubleMatrix &matrix) {
    if (matrix.empty()) {
        std::cout << "Empty matrix" << std::endl;
        return;
    }

    // Get dimensions for reference
    size_t rows = matrix.size();
    size_t cols = matrix[0].size();

    std::cout << "Matrix " << rows << "x" << cols << ":" << std::endl;

    for (const auto &row : matrix) {
        for (const auto &elem : row)
            std::cout << elem << " ";
        std::cout << std::endl;
    }
}
