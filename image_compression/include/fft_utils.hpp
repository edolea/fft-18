#ifndef FFT_UTILS_HPP
#define FFT_UTILS_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include <complex>

std::vector<std::vector<std::complex<double>>> matToComplex2D(const cv::Mat& mat);
cv::Mat complex2DToMat(const std::vector<std::vector<std::complex<double>>>& data);

#endif