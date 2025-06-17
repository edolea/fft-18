#include "../include/error_plot.hpp"
#include "../../include/iterative_fourier.hpp"

#include <cmath>
#include <fstream>
#include <iostream>
#include <algorithm>
#include "../include/fft_utils.hpp"


ErrorPlot::ErrorPlot(const cv::Mat& original_image) {
    original_image.convertTo(original_, CV_32F);
    n_ = original_.rows;
}

void ErrorPlot::ComputeErrorsMagnitudeThresholds(const std::vector<double>& thresholds) {
    error_results_.clear();

    auto input = matToComplex2D(original_);
    std::vector<std::vector<std::complex<double>>> fft_full;

    IterativeFourier<std::vector<std::vector<std::complex<double>>>> fft2D;
    fft2D.compute(input, fft_full);

    for (double t : thresholds) {
        auto filtered = fft_full;
        int zeroed = 0;

        for (int i = 0; i < n_; i++) {
            for (int j = 0; j < n_; j++) {
                if (std::abs(filtered[i][j]) < t) {
                    filtered[i][j] = 0;
                    ++zeroed;
                }
            }
        }

        fft2D.compute(filtered, filtered, false);  // inverse FFT in-place
        cv::Mat reconstructed = complex2DToMat(filtered);

        double error = 0.0;
        for (int i = 0; i < n_; i++)
            for (int j = 0; j < n_; j++)
                error += std::pow(original_.at<float>(i, j) - reconstructed.at<float>(i, j), 2);

        error = std::sqrt(error) / (n_ * n_);
        error_results_.emplace_back(t, ErrorData{error, zeroed});
    }
}

void ErrorPlot::ComputeErrorsBandThresholds(const std::vector<double>& percentages) {
    error_results_.clear();

    auto input = matToComplex2D(original_);
    std::vector<std::vector<std::complex<double>>> fft_full;

    IterativeFourier<std::vector<std::vector<std::complex<double>>>> fft2D;
    fft2D.compute(input, fft_full);

    int cx = n_ / 2;
    int cy = n_ / 2;
    double max_radius = std::sqrt(2.0) * (n_ / 2.0);

    for (double perc : percentages) {
        double band_radius = (perc / 100.0) * max_radius;
        auto filtered = fft_full;
        int zeroed = 0;

        for (int i = 0; i < n_; i++) {
            for (int j = 0; j < n_; j++) {
                double dx = i - cy;
                double dy = j - cx;
                double radius = std::sqrt(dx * dx + dy * dy);

                if (radius > band_radius) {
                    filtered[i][j] = 0;
                    ++zeroed;
                }
            }
        }

        fft2D.compute(filtered, filtered, false);  // inverse FFT
        cv::Mat reconstructed = complex2DToMat(filtered);

        double error = 0.0;
        for (int i = 0; i < n_; i++)
            for (int j = 0; j < n_; j++)
                error += std::pow(original_.at<float>(i, j) - reconstructed.at<float>(i, j), 2);

        error = std::sqrt(error) / (n_ * n_);
        error_results_.emplace_back(perc, ErrorData{error, zeroed});
    }
}

void ErrorPlot::SaveToCSV(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }

    file << "ThresholdOrRadius,Error,FrequenciesEliminated\n";
    for (const auto& [threshold, data] : error_results_) {
        file << threshold << "," << data.error << "," << data.frequencies_eliminated << "\n";
    }

    file.close();
}