#include "fft_utils.hpp"

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