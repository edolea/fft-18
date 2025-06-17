#ifndef IMAGE_SAVER_HPP
#define IMAGE_SAVER_HPP

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <complex>

class ImageSaver {
public:
    static void SaveMagnitudeSpectrum(const std::vector<std::vector<std::complex<double>>>& data,
                                      int size,
                                      double percentage,
                                      const std::string& method,
                                      bool after_filter);

    static void SaveReconstructedImage(const cv::Mat& reconstructed,
                                       double percentage,
                                       const std::string& method);

};

#endif