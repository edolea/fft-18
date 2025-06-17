#include "../include/svd_analysis.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <fstream>
#include <iostream>

SVDAnalyzer::SVDAnalyzer(const std::string& image_path, int size)
    : image_path_(image_path), size_(size) {
    cv::Mat img = cv::imread(image_path_, cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cerr << "Impossible to upload image: " << image_path_ << std::endl;
        exit(1);
    }

    cv::resize(img, original_gray_, cv::Size(size_, size_));
    original_gray_.convertTo(original_gray_, CV_32F);
}

void SVDAnalyzer::ComputeSVD() {
    cv::SVD::compute(original_gray_, S_, U_, VT_);
    singular_values_.clear();
    for (int i = 0; i < S_.rows; i++) {
        singular_values_.push_back(S_.at<float>(i));
    }
}

std::vector<float> SVDAnalyzer::GetSingularValues() const {
    return singular_values_;
}

void SVDAnalyzer::SaveSingularValues(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Impossible to open file: " << filename << std::endl;
        return;
    }

    for (float val : singular_values_) {
        file << val << std::endl;
    }

    file.close();
}

void SVDAnalyzer::ShowOriginalImage(const std::string& suffix) const {
    cv::Mat norm_img;
    cv::normalize(original_gray_, norm_img, 0, 255, cv::NORM_MINMAX);
    norm_img.convertTo(norm_img, CV_8U);

    std::string filename = "../output/image_output/original_image" +
                           (suffix.empty() ? "" : "_" + suffix) + ".png";

    cv::imwrite(filename, norm_img);
}