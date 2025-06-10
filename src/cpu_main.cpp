#include <iostream>
#include <vector>
#include "abstract_transform.hpp"
#include "recursive_fourier.hpp"
#include "iterative_fourier.hpp"
#include "vector_generator.hpp"
#include "timing_saver.h"
#include <cmath>

void vector_print(const doubleVector &result);
void matrix_print(const doubleMatrix &matrix);

int size = 1024; // Maximum size of the vector for testing
double frequency = 2.0;
double amplitude = 1.0;

std::vector<size_t> initialize_vector(size_t max_size) {
    std::vector<size_t> n_list;
    for (size_t i = 1; i <= max_size; i = i * 2) {
        n_list.push_back(i);
    }
    return n_list;
}
int main()
{
    std::vector<std::pair<int, double>> time_recursive_fft,time_iterative_fft;
    for (const std::vector<size_t> n_list = initialize_vector(size); size_t n : n_list) {
        const auto input = RandomVectorGenerator::generate<doubleVector>(n);
        doubleVector output_recursive;
        doubleVector output_iterative;
        RecursiveFourier<doubleVector> recursiveFourier;
        IterativeFourier<doubleVector> iterativeFourier;
        recursiveFourier.compute(input, output_recursive);
        iterativeFourier.compute(input, output_iterative);

        time_iterative_fft.emplace_back(n, iterativeFourier.getTime().count());
        time_recursive_fft.emplace_back(n, recursiveFourier.getTime().count());
    }

    TimingSaver::saveFFTTimings(time_recursive_fft, "recursive_fft_timings.txt");
    TimingSaver::saveFFTTimings(time_iterative_fft, "iterative_fft_timings.txt");





}
