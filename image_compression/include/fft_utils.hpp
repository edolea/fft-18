#ifndef FFT_UTILS_HPP
#define FFT_UTILS_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include <complex>

// Conversione tra cv::Mat e matrice complessa
std::vector<std::vector<std::complex<double>>> matToComplex2D(const cv::Mat& mat);
cv::Mat complex2DToMat(const std::vector<std::vector<std::complex<double>>>& data);

// Calcolo PSNR tra immagine originale e ricostruita
double ComputePSNR(const cv::Mat& original, const cv::Mat& reconstructed);

#endif