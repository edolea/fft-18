#include "./header.hpp"
#include "main_class.hpp"
#include "random_generator.hpp"
#include "sinusoidal_generator.hpp"

int MainClass::N = 0;
double MainClass::frequency = 0.0;
double MainClass::amplitude = 0.0;

int main()
{

    srand(95);

    MainClass::initializeParameters(1024, 0.5, 1.0);

    SinusoidalGenerator sinusoidalGen;
    std::vector<std::complex<double>> input = sinusoidalGen.createInput();

    // INITIAL COMPUTATION TO WARM UP THE GPU AND AVOID TIMING DISCREPANCIES
    std::vector<std::complex<double>> cuda_output_vector = kernel_direct_fft(input).first;

    // DIRECT TRANSFORM
    cuda_output_vector = kernel_direct_fft(input).first;
    std::chrono::duration<double> elapsed_seconds = kernel_direct_fft(input).second;
    plot_fft_result(cuda_output_vector);

    // INVERSE TRANSFORM
    std::vector<std::complex<double>> cuda_inverse_output_vector = kernel_inverse_fft(cuda_output_vector);

    return 0;
}