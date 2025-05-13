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

std::vector<std::complex<double>> cuda_library_fft(const std::vector<std::complex<double>> &input)
{
    const int N = input.size(); // Assuming N is the size of the input vector

    auto START_CUDA_FFT = std::chrono::high_resolution_clock::now();

    cuDoubleComplex *d_input;
    cudaMalloc((void **)&d_input, sizeof(cuDoubleComplex) * N);

    std::vector<cuDoubleComplex> device_input(N);
    for (int i = 0; i < N; ++i)
    {
        device_input[i].x = input[i].real();
        device_input[i].y = input[i].imag();
    }
    cudaMemcpy(d_input, device_input.data(), sizeof(cuDoubleComplex) * N, cudaMemcpyHostToDevice);

    auto CUDA_MALLOC_COMPLETE = std::chrono::high_resolution_clock::now();

    cufftHandle plan;
    cufftPlan1d(&plan, N, CUFFT_Z2Z, 1);
    cufftExecZ2Z(plan, d_input, d_input, CUFFT_FORWARD);

    cudaMemcpy(device_input.data(), d_input, sizeof(cuDoubleComplex) * N, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    auto END_CUDA_FFT = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_cuda_fft = END_CUDA_FFT - START_CUDA_FFT;
    std::chrono::duration<double> duration_cuda_fft_without_malloc = END_CUDA_FFT - CUDA_MALLOC_COMPLETE;

    std::cout << "CUDA FFT execution time: " << duration_cuda_fft.count() << " seconds" << std::endl;
    std::cout << "CUDA FFT execution time WITHOUT MALLOC: " << duration_cuda_fft_without_malloc.count() << " seconds" << std::endl;

    // Convert the results back to std::complex<double>
    std::vector<std::complex<double>> result(N);
    for (int i = 0; i < N; ++i)
    {
        result[i] = std::complex<double>(device_input[i].x, device_input[i].y);
    }

    // Cleanup
    cudaFree(d_input);
    cufftDestroy(plan);

    return result;
}

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

std::vector<std::complex<double>> iterative_FFT(std::vector<std::complex<double>> input)
{
    int n = input.size();
    int m = log2(n);
    std::vector<std::complex<double>> y(n); // Must a power of 2

    // Bit-reversal permutation
    for (int i = 0; i < n; i++)
    {
        int j = 0;
        for (int k = 0; k < m; k++)
        {
            if (i & (1 << k))
            {
                j |= (1 << (m - 1 - k));
            }
        }
        y[j] = input[i];
    }
    // Iterative FFT
    for (int j = 1; j <= m; j++)
    {
        int d = 1 << j;
        std::complex<double> w(1, 0);
        std::complex<double> wd(std::cos(2 * M_PI / d), std::sin(2 * M_PI / d));
        for (int k = 0; k < d / 2; k++)
        {
            for (int m = k; m < n; m += d)
            {
                std::complex<double> t = w * y[m + d / 2];
                std::complex<double> x = y[m];
                y[m] = x + t;
                y[m + d / 2] = x - t;
            }
            w = w * wd;
        }
    }
    return y;
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