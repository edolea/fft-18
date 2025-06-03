#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <cuComplex.h>
#include <cuda.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <chrono>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <cstdlib>
#define THREAD_PER_BLOCK 1024 // this could be changed depends on GPU
using namespace std;

/**
 * @brief Compares two vectors of complex numbers with a specified tolerance.
 *
 * @param output_iterative Vector of complex numbers representing the first dataset.
 * @param cuda_output_vector Vector of complex numbers representing the second dataset.
 * @param tolerance Tolerance value for comparing real and imaginary parts.
 * @return true if the vectors are the same within the specified tolerance, false otherwise.
 */
bool compareComplexVectors(const std::vector<std::complex<double>> &output_iterative,
                           const std::vector<std::complex<double>> &cuda_output_vector,
                           double tolerance = 1e-3)
{
    if (output_iterative.size() != cuda_output_vector.size())
    {
        std::cerr << "Vectors have different sizes!\n";
        return false;
    }

    for (size_t i = 0; i < output_iterative.size(); i++)
    {
        // Get the real and imaginary parts
        double real_diff = std::abs(output_iterative[i].real() - cuda_output_vector[i].real());
        double imag_diff = std::abs(output_iterative[i].imag() - cuda_output_vector[i].imag());

        // Check if the differences in both real and imaginary parts are within the tolerance
        if (real_diff > tolerance || imag_diff > tolerance)
        {
            std::cout.precision(15);
            std::cout << "output_iterative[" << i << "] = " << output_iterative[i]
                      << ", cuda[" << i << "] = " << cuda_output_vector[i]
                      << ", real diff = " << real_diff
                      << ", imag diff = " << imag_diff << std::endl;
            std::cout << "Different result in line " << i << std::endl;
            return false; // Return false on the first mismatch
        }
    }

    std::cout << "Same result for the methods" << std::endl;
    return true;
}

// Function to convert cuDoubleComplex* to std::vector<std::complex<double>>
std::vector<std::complex<double>> cuDoubleComplexToVector(const cuDoubleComplex *a, size_t size)
{
    std::vector<std::complex<double>> result(size);
    for (size_t i = 0; i < size; ++i)
    {
        result[i] = std::complex<double>(cuCreal(a[i]), cuCimag(a[i]));
    }
    return result;
}

// Funzione per graficare usando gnuplot
void plot_fft_result(const std::vector<std::complex<double>> &fft_result, const std::string &title = "FFT Magnitude")
{
    // Calcola la magnitudine delle frequenze
    std::vector<double> magnitude;
    for (const auto &val : fft_result)
    {
        magnitude.push_back(std::abs(val));
    }

    // Crea l'asse delle frequenze
    int num_points = fft_result.size();
    std::vector<double> frequencies(num_points);
    for (int i = 0; i < num_points; ++i)
    {
        frequencies[i] = i; // Frequenze normalizzate (o usa una scala personalizzata)
    }

    // Scrive i dati in un file temporaneo nella cartella Output_result
    std::ofstream file("fft_output.csv", std::ios::out | std::ios::trunc); // Cambia il nome del file
    for (size_t i = 0; i < frequencies.size(); ++i)
    {
        file << frequencies[i] << "," << magnitude[i] << "\n"; // Usa ',' come delimitatore
    }
    file.close();

    return;
}