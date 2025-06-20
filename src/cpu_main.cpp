#include <iostream>
#include <vector>
#include <cmath>
#include <string>

#include "abstract_transform.hpp"
#include "recursive_fourier.hpp"
#include "iterative_fourier.hpp"
#include "vector_generator.hpp"
#include "timing_saver.h"
#include "save_result.hpp"

void vector_print(const doubleVector &result);
void matrix_print(const doubleMatrix &matrix);

int size = 1024;
double frequency = 2.0;
double amplitude = 1.0;

std::vector<size_t> initialize_vector(size_t max_size) {
    std::vector<size_t> n_list;
    for (size_t i = 1; i <= max_size; i *= 2) {
        n_list.push_back(i);
    }
    return n_list;
}

int main() {
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

    // === Prompt for function type ===
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

    const auto n_list = initialize_vector(size);

    // === 1D FFT Test ===
    if (is_1d) {
        std::vector<std::pair<int, double>> time_recursive_fft, time_iterative_fft;

        for (size_t n : n_list) {
            doubleVector input;
            if (use_sin_function) {
                input = RandomVectorGenerator::generate<doubleVector>(n, frequency, amplitude);
            } else {
                input = RandomVectorGenerator::generate<doubleVector>(n);
            }

            doubleVector recursive_output, iterative_output;

            RecursiveFourier<doubleVector> recursiveFft;
            IterativeFourier<doubleVector> iterativeFft;

            recursiveFft.compute(input, recursive_output);
            iterativeFft.compute(input, iterative_output);

            time_recursive_fft.emplace_back(n, recursiveFft.getTime().count());
            time_iterative_fft.emplace_back(n, iterativeFft.getTime().count());
        }

        TimingSaver::saveFFTTimings(time_recursive_fft, "recursive_fft_timings_1d.txt");
        TimingSaver::saveFFTTimings(time_iterative_fft, "iterative_fft_timings_1d.txt");
    }

    // === 2D FFT Test ===
    if (is_2d) {
        std::vector<std::pair<int, double>> time_recursive_fft, time_iterative_fft;

        RecursiveFourier<doubleMatrix> recursiveFft2D;
        IterativeFourier<doubleMatrix> iterativeFft2D;

        for (size_t n : n_list) {
            doubleMatrix input;
            if (use_sin_function) {
                input = RandomVectorGenerator::generate<doubleMatrix>(n, frequency, amplitude);
            } else {
                input = RandomVectorGenerator::generate<doubleMatrix>(static_cast<int>(n), frequency, amplitude);
            }

            doubleMatrix recursive_output, iterative_output;

            recursiveFft2D.compute(input, recursive_output);
            iterativeFft2D.compute(input, iterative_output);

            time_recursive_fft.emplace_back(n, recursiveFft2D.getTime().count());
            time_iterative_fft.emplace_back(n, iterativeFft2D.getTime().count());
        }

        TimingSaver::saveFFTTimings(time_recursive_fft, "recursive_fft_timings_2d.txt");
        TimingSaver::saveFFTTimings(time_iterative_fft, "iterative_fft_timings_2d.txt");
    }

    return 0;
}
