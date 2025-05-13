#include "./header.hpp"
#include "main_class_2d.hpp"
#include "sinusoidal_2d.hpp"
#include "gaussian_2d.hpp"
#include "random_2d.hpp"
#include <vector>
#include <complex>
#include <cmath>

int MainClass2D::rows = 0;
int MainClass2D::cols = 0;

int main()
{
  int *prova;
  cudaMallocManaged((void **)&prova, sizeof(int));
  srand(95);

  MainClass2D::initializeParameters(1024, 1024);

  Sinusoidal2D sinusoidalGen;
  Gaussian2D gaussianGen;
  Random2D randomGen;

  // Example usage
  auto input_sinusoidal = sinusoidalGen.createInput();
  auto input_gaussian = gaussianGen.createInput();
  auto input_random = randomGen.createInput();
  // Image usage
  auto input_image = load_matrix_from_txt("./image_converter/matrix_output.txt");

  auto input = input_image;

  // DIRECT TRANSFORM
  bool direct = true;
  std::pair<std::vector<std::vector<std::complex<double>>>, std::chrono::duration<double>> result = direct_fft_2d(input);

  std::vector<std::vector<std::complex<double>>> cuda_output_vector = result.first;
  std::chrono::duration<double> elapsed_seconds = result.second;

  // CUDA FFT
  cuda_output_vector = cuda_library_fft_2d(input);
  save_fft_result_2d(cuda_output_vector);
  // INVERSE TRANSFORM
  std::vector<std::vector<std::complex<double>>> cuda_output_vector_inverse = inverse_fft_2d(cuda_output_vector);

  return 0;
}