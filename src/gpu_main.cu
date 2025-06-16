#include "vector_generator_gpu.hpp"
#include "parallel_fourier.hpp"
#include "timing_saver.h"
#include <cmath>

int ParameterClass::N = 0;
double ParameterClass::frequency = 0.0;
double ParameterClass::amplitude = 0.0;


int max_vector_size = 2048; // Maximum size of the vector for testing
double frequency = 2.0;
double amplitude = 1.0;

std::vector<size_t> initialize_vector(const size_t max_size) {
    std::vector<size_t> n_list;
    for (size_t i = 1; i <= max_size; i = i * 2) {
        n_list.push_back(i);
    }
    return n_list;
}


int main()
{
    std::vector<std::tuple<int, double, int>> timings_parallel_fft;
    for (const std::vector<size_t> n_list = initialize_vector(max_vector_size); size_t n : n_list) {
        const auto input = RandomVectorGenerator::generate<doubleVector>(n);

        ParallelFourier<doubleVector> gpuFft(input);
        gpuFft.compute(input);
        timings_parallel_fft.emplace_back(n, gpuFft.getTime().count(), THREAD_PER_BLOCK);
    }

    TimingSaver::saveParallelFFTTimings(timings_parallel_fft, "timings_parallel_fft.txt");

    return 0;
}