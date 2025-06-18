#include "../include/fft_analysis_band.hpp"
#include "../include/fft_utils.hpp"
#include "../../include/iterative_fourier.hpp"

#include <cmath>
#include <iostream>
#include <fstream>

FFTAnalysisBand::FFTAnalysisBand(int size)
    : n_(size), reconstruction_error_(0.0), psnr_value_(0.0) {}

void FFTAnalysisBand::LoadImage(const std::string& path) {
    original_image_ = cv::imread(path, cv::IMREAD_GRAYSCALE);
    if (original_image_.empty()) {
        std::cerr << "Failed to load image." << std::endl;
        return;
    }
    cv::resize(original_image_, original_image_, cv::Size(n_, n_));
    original_image_.convertTo(original_float_, CV_32F);
}

void FFTAnalysisBand::ComputeFFT() {
    auto input = matToComplex2D(original_float_);

    IterativeFourier<std::vector<std::vector<std::complex<double>>>> fft2D;
    fft2D.compute(input, fft2d_);

    filtered_fft2d_ = fft2d_;
}

void FFTAnalysisBand::ApplyBandpassFilterPercentage(double percentage) {
    filtered_fft2d_ = fft2d_;

    // fftshift
    std::rotate(filtered_fft2d_.begin(), filtered_fft2d_.begin() + n_ / 2, filtered_fft2d_.end());
    for (auto& row : filtered_fft2d_)
        std::rotate(row.begin(), row.begin() + n_ / 2, row.end());

    int cx = n_ / 2;
    int cy = n_ / 2;
    double max_radius = std::sqrt(cx * cx + cy * cy);
    double radius_thresh = (percentage / 100.0) * max_radius;

    for (int i = 0; i < n_; i++) {
        for (int j = 0; j < n_; j++) {
            int di = i - cy;
            int dj = j - cx;
            double dist = std::sqrt(di * di + dj * dj);
            if (dist > radius_thresh)
                filtered_fft2d_[i][j] = 0;
        }
    }

    // ifftshift
    std::rotate(filtered_fft2d_.begin(), filtered_fft2d_.begin() + n_ / 2, filtered_fft2d_.end());
    for (auto& row : filtered_fft2d_)
        std::rotate(row.begin(), row.begin() + n_ / 2, row.end());
}

void FFTAnalysisBand::ComputeIFFT() {
    IterativeFourier<std::vector<std::vector<std::complex<double>>>> fft2D;
    fft2D.compute(filtered_fft2d_, filtered_fft2d_, false);

    reconstructed_image_ = complex2DToMat(filtered_fft2d_);
}

void FFTAnalysisBand::ComputeReconstructionError() {
    double error = 0.0;
    double max_pixel = 255.0;

    for (int i = 0; i < n_; i++) {
        for (int j = 0; j < n_; j++) {
            float orig = original_float_.at<float>(i, j);
            float recon = reconstructed_image_.at<float>(i, j);
            error += std::pow(orig - recon, 2);
        }
    }

    double mse = error / (n_ * n_);
    reconstruction_error_ = std::sqrt(error) / (n_ * n_);

    if (mse == 0)
        psnr_value_ = INFINITY;
    else
        psnr_value_ = 10.0 * std::log10((max_pixel * max_pixel) / mse);
}

void FFTAnalysisBand::SaveMagnitudeToCSV(const std::string& filename, bool filtered) const {
    const auto& data = filtered ? filtered_fft2d_ : fft2d_;
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
        return;
    }

    for (int i = 0; i < n_; i++) {
        for (int j = 0; j < n_; j++) {
            file << std::abs(data[i][j]);
            if (j != n_ - 1) file << ",";
        }
        file << "\n";
    }

    file.close();
}

double FFTAnalysisBand::GetError() const {
    return reconstruction_error_;
}

double FFTAnalysisBand::GetPSNR() const {
    return psnr_value_;
}

const std::vector<std::vector<std::complex<double>>>& FFTAnalysisBand::GetOriginalFFT() const {
    return fft2d_;
}

const std::vector<std::vector<std::complex<double>>>& FFTAnalysisBand::GetFilteredFFT() const {
    return filtered_fft2d_;
}

const cv::Mat& FFTAnalysisBand::GetReconstructedImage() const {
    return reconstructed_image_;
}