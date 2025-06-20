#include "vector_generator_gpu.hpp"
#include "parallel_fourier.hpp"
#include "timing_saver.h"
#include <cmath>
#include <iostream>
#include <vector>
#include <tuple>
#include <string>

int ParameterClass::N = 0;
double ParameterClass::frequency = 0.0;
double ParameterClass::amplitude = 0.0;

int max_vector_size = 4096; // Maximum size of the vector for testing
double frequency = 2.0;
double amplitude = 1.0;

std::vector<size_t> initialize_vector(const size_t max_size) {
    std::vector<size_t> n_list;
    for (size_t i = 1; i <= max_size; i *= 2) {
        n_list.push_back(i);
    }
    return n_list;
}

int main(int argc, char* argv[]) {

    // === Prompt for 1D or 2D ===
    std::string dimension_choice;
    bool is_1d = false, is_2d = false;

    while (true) {
        std::cout << "Do you want to process a 1D or 2D FFT? (Enter '1D' or '2D'): ";
        std::cin >> dimension_choice;

        if (dimension_choice == "1D" || dimension_choice == "1d") {
            is_1d = true;
            break;
        } else if (dimension_choice == "2D" || dimension_choice == "2d") {
            is_2d = true;
            break;
        } else {
            std::cerr << "Invalid input. Please enter '1D' or '2D'." << std::endl;
        }
    }

    // Ask user for input type
    bool use_sin_function = false;
    std::string user_choice;

    std::cout << "Do you want to process a random function or a sin/cos function? (Enter 'random' or 'sin'): ";
    std::cin >> user_choice;

    if (user_choice == "sin" || user_choice == "sinus" || user_choice == "cos") {
        use_sin_function = true;

        std::cout << "Enter the frequency: ";
        std::cin >> frequency;

        std::cout << "Enter the amplitude: ";
        std::cin >> amplitude;
    } else if (user_choice != "random") {
        std::cerr << "Invalid choice. Please enter 'random' or 'sin'." << std::endl;
        return 1;
    }

    const std::string outputFilename = "timings_parallel_fft_" + std::to_string(THREAD_PER_BLOCK) + ".txt";
    const auto n_list = initialize_vector(max_vector_size);

    if (is_1d) {
        std::vector<std::tuple<int, double, int>> timings_parallel_fft;

        for (size_t n : n_list) {
            doubleVector input;

            if (use_sin_function) {
                input = RandomVectorGenerator::generate<doubleVector>(n, frequency, amplitude);
            } else {
                input = RandomVectorGenerator::generate<doubleVector>(n);
            }

            ParallelFourier<doubleVector> gpuFft(input);
            gpuFft.compute(input);
            timings_parallel_fft.emplace_back(n, gpuFft.getTime().count(), THREAD_PER_BLOCK);
        }
        TimingSaver::saveParallelFFTTimings(timings_parallel_fft, outputFilename);
    }

    if (is_2d) {
        std::vector<std::tuple<int, double, int>> timings_parallel_fft;

        for (size_t n : n_list) {
            ParameterClass::initializeParameters(n, frequency, amplitude);
            doubleMatrix input = RandomVectorGenerator::generate2D<doubleMatrix>(n, frequency, amplitude);

            ParallelFourier<doubleMatrix> gpuFft(input);
            gpuFft.compute(input);
            timings_parallel_fft.emplace_back(n, gpuFft.getTime().count(), THREAD_PER_BLOCK);
        }

        TimingSaver::saveParallelFFTTimings(timings_parallel_fft, outputFilename);
    }

    return 0;
}

