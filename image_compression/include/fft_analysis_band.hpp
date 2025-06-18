#ifndef FFT_ANALYSIS_BAND_HPP
#define FFT_ANALYSIS_BAND_HPP

#include <vector>
#include <complex>
#include <string>
#include <opencv2/opencv.hpp>
#include "../../include/iterative_fourier.hpp"

class FFTAnalysisBand {
public:
    explicit FFTAnalysisBand(int size);

    void LoadImage(const std::string& path);
    void ComputeFFT();
    void ApplyBandpassFilterPercentage(double percentage);
    void ComputeIFFT();
    void ComputeReconstructionError();
    void SaveMagnitudeToCSV(const std::string& filename, bool filtered) const;

    // === Getters ===
    double GetError() const;
    double GetPSNR() const;
    const std::vector<std::vector<std::complex<double>>>& GetOriginalFFT() const;
    const std::vector<std::vector<std::complex<double>>>& GetFilteredFFT() const;
    const cv::Mat& GetReconstructedImage() const;

private:
    int n_;
    cv::Mat original_image_;
    cv::Mat original_float_;
    cv::Mat reconstructed_image_;

    std::vector<std::vector<std::complex<double>>> fft2d_;
    std::vector<std::vector<std::complex<double>>> filtered_fft2d_;

    double reconstruction_error_;
    double psnr_value_ = 0.0;
};

#endif