#ifndef ERROR_PLOT_HPP
#define ERROR_PLOT_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <utility>

#include "../../include/iterative_fourier.hpp"

struct ErrorData {
    double error;
    double psnr;
    int frequencies_eliminated;
};

class ErrorPlot {
public:
    explicit ErrorPlot(const cv::Mat& original_image);

    void ComputeErrorsMagnitudeThresholds(const std::vector<double>& thresholds);
    void ComputeErrorsBandThresholds(const std::vector<double>& percentages);

    void SaveToCSV(const std::string& filename) const;

private:
    int n_;
    cv::Mat original_;
    std::vector<std::pair<double, ErrorData>> error_results_;
};

#endif