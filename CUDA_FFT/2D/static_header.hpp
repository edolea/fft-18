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
#include "main_class_2d.hpp"
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

bool compareComplexMatrices(const std::vector<std::vector<std::complex<double>>> &output_iterative,
                            const std::vector<std::vector<std::complex<double>>> &cuda_output_matrix,
                            double tolerance = 1e-3)
{
    if (output_iterative.size() != cuda_output_matrix.size() || output_iterative[0].size() != cuda_output_matrix[0].size())
    {
        std::cerr << "Matrices have different sizes!\n";
        return false;
    }

    for (size_t i = 0; i < output_iterative.size(); ++i)
    {
        for (size_t j = 0; j < output_iterative[i].size(); ++j)
        {
            // Get the real and imaginary parts
            double real_diff = std::abs(output_iterative[i][j].real() - cuda_output_matrix[i][j].real());
            double imag_diff = std::abs(output_iterative[i][j].imag() - cuda_output_matrix[i][j].imag());

            // Check if the differences in both real and imaginary parts are within the tolerance
            if (real_diff > tolerance || imag_diff > tolerance)
            {
                std::cout.precision(15);
                std::cout << "output_iterative[" << i << "][" << j << "] = " << output_iterative[i][j]
                          << ", cuda[" << i << "][" << j << "] = " << cuda_output_matrix[i][j]
                          << ", real diff = " << real_diff
                          << ", imag diff = " << imag_diff << std::endl;
                std::cout << "Different result at position (" << i << ", " << j << ")" << std::endl;
                return false; // Return false on the first mismatch
            }
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

std::vector<std::vector<std::complex<double>>> transpose(const std::vector<std::vector<std::complex<double>>> &matrix)
{
    int rows = matrix.size();
    int cols = matrix[0].size();
    std::vector<std::vector<std::complex<double>>> transposed(cols, std::vector<std::complex<double>>(rows));

    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            transposed[j][i] = matrix[i][j];
        }
    }

    return transposed;
}

std::vector<std::vector<std::complex<double>>> iterative_FFT_2D(const std::vector<std::vector<std::complex<double>>> &input)
{
    int rows = input.size();
    int cols = input[0].size();
    auto stasrt_iterative_2d = std::chrono::high_resolution_clock::now();

    // Apply FFT to each row
    std::vector<std::vector<std::complex<double>>> rowTransformed(rows, std::vector<std::complex<double>>(cols));
    for (int i = 0; i < rows; ++i)
    {
        rowTransformed[i] = iterative_FFT(input[i]);
    }

    // Transpose the result
    auto transposed = transpose(rowTransformed);

    // Apply FFT to each column (now row of the transposed matrix)
    std::vector<std::vector<std::complex<double>>> columnTransformed(cols, std::vector<std::complex<double>>(rows));
    for (int i = 0; i < cols; ++i)
    {
        columnTransformed[i] = iterative_FFT(transposed[i]);
    }

    // Transpose back to get the final result
    auto result = transpose(columnTransformed);
    auto end_iterative_2d = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_iterative_2d = end_iterative_2d - stasrt_iterative_2d;
    std::cout << "Iterative 2D FFT execution time: " << duration_iterative_2d.count() << " seconds" << std::endl;
    return result;
}

std::vector<std::vector<std::complex<double>>> cuda_library_fft_2d(const std::vector<std::vector<std::complex<double>>> &input)
{
    int rows = input.size();
    int cols = input[0].size();
    size_t size = rows * cols;

    // Flatten input matrix into a single array
    std::vector<cuDoubleComplex> flat_input(size);
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            flat_input[i * cols + j].x = input[i][j].real();
            flat_input[i * cols + j].y = input[i][j].imag();
        }
    }

    cuDoubleComplex *d_input;
    cudaMalloc((void **)&d_input, sizeof(cuDoubleComplex) * size);
    cudaMemcpy(d_input, flat_input.data(), sizeof(cuDoubleComplex) * size, cudaMemcpyHostToDevice);

    // Create 2D FFT plan
    cufftHandle plan;
    cufftPlan2d(&plan, rows, cols, CUFFT_Z2Z);

    // Perform forward 2D FFT
    auto START_CUDA_FFT = std::chrono::high_resolution_clock::now();
    cufftExecZ2Z(plan, d_input, d_input, CUFFT_FORWARD);
    cudaDeviceSynchronize();
    auto END_CUDA_FFT = std::chrono::high_resolution_clock::now();

    // Copy result back to host
    cudaMemcpy(flat_input.data(), d_input, sizeof(cuDoubleComplex) * size, cudaMemcpyDeviceToHost);

    // Reshape the output back to 2D matrix
    std::vector<std::vector<std::complex<double>>> output(rows, std::vector<std::complex<double>>(cols));
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            output[i][j] = std::complex<double>(flat_input[i * cols + j].x, flat_input[i * cols + j].y);
        }
    }

    std::chrono::duration<double> duration_cuda_fft = END_CUDA_FFT - START_CUDA_FFT;
    std::cout << "CUDA 2D FFT execution time: " << duration_cuda_fft.count() << " seconds" << std::endl;

    // Cleanup
    cufftDestroy(plan);
    cudaFree(d_input);

    return output;
}

// Funzione per graficare usando gnuplot
void save_fft_result_2d(const std::vector<std::vector<std::complex<double>>> &fft_result, const std::string &filename = "fft_output_2d.csv")
{
    // Apri il file in modalità scrittura
    std::ofstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Errore: impossibile aprire il file " << filename << std::endl;
        return;
    }

    // Salva i valori della FFT in formato CSV
    for (const auto &row : fft_result)
    {
        for (size_t j = 0; j < row.size(); ++j)
        {
            // Controlla il segno della parte immaginaria
            if (row[j].imag() >= 0)
            {
                file << row[j].real() << "+" << row[j].imag() << "j"; // Aggiungi '+' se immaginario positivo
            }
            else
            {
                file << row[j].real() << row[j].imag() << "j"; // Nessun '+' se immaginario negativo
            }

            if (j < row.size() - 1)
            {
                file << ","; // Separatore di colonne
            }
        }
        file << "\n"; // Separatore di righe
    }

    file.close();
    std::cout << "Dati FFT 2D salvati su: " << filename << std::endl;
}

std::vector<std::vector<std::complex<double>>> load_matrix_from_txt(const std::string &filename)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        throw std::runtime_error("Could not open file");
    }

    int height, width;
    file >> height >> width; // Read matrix dimensions

    std::vector<std::vector<std::complex<double>>> matrix(height, std::vector<std::complex<double>>(width));

    for (int i = 0; i < height; ++i)
    {
        for (int j = 0; j < width; ++j)
        {
            int value;
            file >> value;
            matrix[i][j] = std::complex<double>(static_cast<double>(value), 0.0); // Convert to complex number
        }
    }

    file.close();
    return matrix;
}