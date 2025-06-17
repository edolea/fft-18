#ifndef SVD_ANALYSIS_HPP_
#define SVD_ANALYSIS_HPP_

#include <opencv2/core.hpp>
#include <string>
#include <vector>

class SVDAnalyzer {
public:
    SVDAnalyzer(const std::string& image_path, int size);

    void ComputeSVD();
    void SaveSingularValues(const std::string& filename) const;
    void ShowOriginalImage(const std::string& suffix = "") const;

    std::vector<float> GetSingularValues() const;

private:
    std::string image_path_;
    int size_;

    cv::Mat original_gray_;
    cv::Mat S_;
    cv::Mat U_;
    cv::Mat VT_;

    std::vector<float> singular_values_;
};

#endif