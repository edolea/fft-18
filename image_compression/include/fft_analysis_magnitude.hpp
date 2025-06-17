#ifndef FFT_ANALYSIS_MAGNITUDE_HPP
#define FFT_ANALYSIS_MAGNITUDE_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include <complex>
#include <string>
#include "../../include/iterative_fourier.hpp"

class FFTAnalysisMagnitude {
public:
    explicit FFTAnalysisMagnitude(int size);

    void LoadImage(const std::string& path);
    void ComputeFFT();
    void ApplyThresholdPercentage(double percentage);
    void ComputeIFFT();
    void ComputeReconstructionError();
    void SaveFFTToCSV(const std::string& filename) const;
    void SaveMagnitudeToCSV(const std::string& filename, bool after_filter = false) const;

    double GetError() const;
    const std::vector<std::vector<std::complex<double>>>& GetOriginalFFT() const;
    const std::vector<std::vector<std::complex<double>>>& GetFilteredFFT() const;
    const cv::Mat& GetReconstructedImage() const;

private:
    int n_;
    double reconstruction_error_;

    cv::Mat original_image_;
    cv::Mat original_float_;
    cv::Mat reconstructed_image_;

    std::vector<std::vector<std::complex<double>>> fft2d_;
    std::vector<std::vector<std::complex<double>>> filtered_fft2d_;
};


#endif