#include "fft_utils.hpp"
#include <opencv2/opencv.hpp>
#include <cmath>
#include <vector>
#include <complex>

std::vector<std::vector<std::complex<double>>> matToComplex2D(const cv::Mat& mat) {
    int n = mat.rows;
    std::vector<std::vector<std::complex<double>>> result(n, std::vector<std::complex<double>>(n));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            result[i][j] = std::complex<double>(mat.at<float>(i, j), 0.0);
    return result;
}

cv::Mat complex2DToMat(const std::vector<std::vector<std::complex<double>>>& data) {
    int n = data.size();
    cv::Mat mat(n, n, CV_32F);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            mat.at<float>(i, j) = std::real(data[i][j]);
    return mat;
}

double ComputePSNR(const cv::Mat& original, const cv::Mat& reconstructed) {
    cv::Mat diff;
    cv::absdiff(original, reconstructed, diff);
    diff.convertTo(diff, CV_32F);
    diff = diff.mul(diff);

    double mse = cv::sum(diff)[0] / (original.total());
    if (mse == 0.0) return INFINITY;  // No error
    return 10.0 * std::log10(255.0 * 255.0 / mse);
}