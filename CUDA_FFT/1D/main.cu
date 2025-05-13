#include "../../include/vector_generator.hpp"
#include "cuda_fourier.hpp"

int N = 1024;
double frequency = 0.5;
double amplitude = 1.0;

using complexDouble = std::complex<double>;
using doubleVector = std::vector<complexDouble>;


int main()
{
    /*
    srand(95);

    SinusoidalGenerator sinusoidalGen;
    std::vector<std::complex<double>> input = sinusoidalGen.createInput(); */

    MainClass::initializeParameters(N, frequency, amplitude);
    auto input = RandomVectorGenerator::generate<doubleVector>(N, frequency, amplitude);

    CudaFourier fft(input); // FIXME: CLion non riconosce la classe CudaFourier --> prob perchè non è nel mio cmake setup
    fft.compute();
    auto output = fft.getOutput;

    /*
    // INITIAL COMPUTATION TO WARM UP THE GPU AND AVOID TIMING DISCREPANCIES
    std::vector<std::complex<double>> cuda_output_vector = kernel_direct_fft(input).first;

    // DIRECT TRANSFORM  ---- FIXME: SUPER ERRORE !!!!
    cuda_output_vector = kernel_direct_fft(input).first;
    std::chrono::duration<double> elapsed_seconds = kernel_direct_fft(input).second;
    plot_fft_result(cuda_output_vector);

    // INVERSE TRANSFORM
    std::vector<std::complex<double>> cuda_inverse_output_vector = kernel_inverse_fft(cuda_output_vector);
     */

    return 0;
}