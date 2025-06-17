#include "../include/fft_analysis_magnitude.hpp"
#include "../include/fft_utils.hpp"
#include "../../include/iterative_fourier.hpp"


#include <algorithm>
#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>


FFTAnalysisMagnitude::FFTAnalysisMagnitude(int size)
    : n_(size), reconstruction_error_(0.0) {}

void FFTAnalysisMagnitude::LoadImage(const std::string& path) {
    original_image_ = cv::imread(path, cv::IMREAD_GRAYSCALE);
    if (original_image_.empty()) {
        std::cerr << "Failed to load image." << std::endl;
        return;
    }
    cv::resize(original_image_, original_image_, cv::Size(n_, n_));
    original_image_.convertTo(original_float_, CV_32F);
}

void FFTAnalysisMagnitude::ComputeFFT() {
    auto input = matToComplex2D(original_float_);

    IterativeFourier<std::vector<std::vector<std::complex<double>>>> fft2D;
    fft2D.compute(input, fft2d_);

    filtered_fft2d_ = fft2d_;
}

void FFTAnalysisMagnitude::ApplyThresholdPercentage(double percentage) {
    filtered_fft2d_ = fft2d_;

    std::vector<double> magnitudes;
    magnitudes.reserve(n_ * n_);
    for (const auto& row : fft2d_)
        for (const auto& val : row)
            magnitudes.push_back(std::abs(val));

    std::sort(magnitudes.begin(), magnitudes.end());
    size_t index = std::max<size_t>(1, static_cast<size_t>((percentage / 100.0) * magnitudes.size())) - 1;
    index = std::min(index, magnitudes.size() - 1);
    double threshold = magnitudes[index];

    int zeroed = 0;
    for (int i = 0; i < n_; i++) {
        for (int j = 0; j < n_; j++) {
            if (std::abs(filtered_fft2d_[i][j]) < threshold) {
                filtered_fft2d_[i][j] = 0;
                ++zeroed;
            }
        }
    }
}

void FFTAnalysisMagnitude::ComputeIFFT() {
    IterativeFourier<std::vector<std::vector<std::complex<double>>>> fft2D;
    fft2D.compute(filtered_fft2d_, filtered_fft2d_, false); // inverse FFT in-place

    reconstructed_image_ = complex2DToMat(filtered_fft2d_);
}

void FFTAnalysisMagnitude::ComputeReconstructionError() {
    cv::Mat recon_float;
    reconstructed_image_.convertTo(recon_float, CV_32F);

    double error = 0.0;
    for (int i = 0; i < n_; i++)
        for (int j = 0; j < n_; j++)
            error += std::pow(original_float_.at<float>(i, j) - recon_float.at<float>(i, j), 2);

    reconstruction_error_ = std::sqrt(error) / (n_ * n_);
}

void FFTAnalysisMagnitude::SaveFFTToCSV(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }

    file << std::setprecision(12);
    for (const auto& row : fft2d_) {
        for (size_t j = 0; j < row.size(); j++) {
            const auto& val = row[j];
            file << std::real(val);
            if (std::imag(val) >= 0)
                file << "+" << std::imag(val) << "j";
            else
                file << std::imag(val) << "j";
            if (j != row.size() - 1)
                file << ",";
        }
        file << "\n";
    }

    file.close();
    std::cout << "FFT 2D saved in " << filename << std::endl;
}

void FFTAnalysisMagnitude::SaveMagnitudeToCSV(const std::string& filename, bool after_filter) const {
    const auto& data = after_filter ? filtered_fft2d_ : fft2d_;

    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }

    file << std::setprecision(12) << std::fixed;

    for (int i = 0; i < n_; i++) {
        for (int j = 0; j < n_; j++) {
            double magnitude = std::abs(data[i][j]);
            file << magnitude;
            if (j != n_ - 1)
                file << ",";
        }
        file << "\n";
    }

    file.close();
    std::cout << "Magnitude saved to " << filename << std::endl;
}

double FFTAnalysisMagnitude::GetError() const {
    return reconstruction_error_;
}

const std::vector<std::vector<std::complex<double>>>& FFTAnalysisMagnitude::GetOriginalFFT() const {
    return fft2d_;
}

const std::vector<std::vector<std::complex<double>>>& FFTAnalysisMagnitude::GetFilteredFFT() const {
    return filtered_fft2d_;
}

const cv::Mat& FFTAnalysisMagnitude::GetReconstructedImage() const {
    return reconstructed_image_;
}