#include "../include/image_saver.hpp"
#include <filesystem>
#include <opencv2/imgcodecs.hpp>
#include <cmath>
#include <sstream>

namespace fs = std::filesystem;

namespace {
std::string BuildFilename(const std::string& prefix, double percentage, const std::string& type, const std::string& suffix) {
  std::ostringstream oss;
  oss << prefix << "_" << static_cast<int>(percentage) << "p_" << type << suffix;
  return oss.str();
}
}

void ImageSaver::SaveMagnitudeSpectrum(const std::vector<std::vector<std::complex<double>>>& data,
                                       int size,
                                       double percentage,
                                       const std::string& type,
                                       bool logscale) {
  cv::Mat mag(size, size, CV_32F);
  for (int i = 0; i < size; i++)
    for (int j = 0; j < size; j++)
      mag.at<float>(i, j) = std::abs(data[i][j]);

  if (logscale) {
    mag += 1.0;  // avoid log(0)
    cv::log(mag, mag);
  }

  cv::normalize(mag, mag, 0, 255, cv::NORM_MINMAX);
  mag.convertTo(mag, CV_8U);

  std::string filename = BuildFilename("../OUTPUT_RESULT/image_output/filtered_fft_magnitude", percentage, type, logscale ? "_log.png" : ".png");
  cv::imwrite(filename, mag);
  std::cout << (logscale ? "Log-scaled " : "") << "magnitude spectrum saved to " << filename << std::endl;
}

void ImageSaver::SaveReconstructedImage(const cv::Mat& image,
                                        double percentage,
                                        const std::string& type) {
  cv::Mat out;
  cv::normalize(image, out, 0, 255, cv::NORM_MINMAX);
  out.convertTo(out, CV_8U);

  std::string filename = BuildFilename("../OUTPUT_RESULT/image_output/reconstructed", percentage, type, ".png");
  cv::imwrite(filename, out);
  std::cout << "Reconstructed image saved to " << filename << std::endl;
}

