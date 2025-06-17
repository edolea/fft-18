#include "svd_analysis.hpp"
#include "fft_analysis_magnitude.hpp"
#include "fft_analysis_band.hpp"
#include "image_saver.hpp"
#include "error_plot.hpp"

#include <mpi.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <iomanip>

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);

  constexpr int size = 256;
  const std::string image_path = argv[1];
  const std::vector<double> percentages = {1, 5, 10, 15, 25, 40, 50, 60, 75, 85, 90, 95, 99};

  // === SVD ===
  SVDAnalyzer svd(image_path, size);
  svd.ComputeSVD();
  svd.SaveSingularValues("../OUTPUT_RESULT/csv_output/singular_values.csv");
  svd.ShowOriginalImage("svd");

  // === FFT MAGNITUDE ===
  FFTAnalysisMagnitude fft_mag(size);
  fft_mag.LoadImage(image_path);
  fft_mag.ComputeFFT();
  fft_mag.SaveMagnitudeToCSV("../OUTPUT_RESULT/csv_output/fft_magnitude.csv");
  fft_mag.SaveFFTToCSV("../OUTPUT_RESULT/csv_output/fft_output_2d.csv");

  std::ofstream mag_csv("../OUTPUT_RESULT/csv_output/error_vs_threshold_magnitude_percentage.csv");
  mag_csv << "Percentage,Error\n";
  for (double perc : percentages) {
    fft_mag.ApplyThresholdPercentage(perc);
    fft_mag.SaveMagnitudeToCSV("../OUTPUT_RESULT/csv_output/fft_magnitude_filtered_" + std::to_string(static_cast<int>(perc)) + "p_magnitude.csv", true);
    fft_mag.ComputeIFFT();
    fft_mag.ComputeReconstructionError();

    ImageSaver::SaveMagnitudeSpectrum(fft_mag.GetFilteredFFT(), size, perc, "magnitude", true);
    ImageSaver::SaveReconstructedImage(fft_mag.GetReconstructedImage(), perc, "magnitude");

    mag_csv << std::fixed << std::setprecision(50)
            << perc << "," << fft_mag.GetError() << "\n";
  }
  mag_csv.close();

  // === FFT BAND (FREQUENCY FILTERING) ===
  FFTAnalysisBand fft_band(size);
  fft_band.LoadImage(image_path);
  fft_band.ComputeFFT();

  std::ofstream band_csv("../OUTPUT_RESULT/csv_output/error_vs_threshold_band_percentage.csv");
  band_csv << "Percentage,Error\n";

  for (double perc : percentages) {
    fft_band.ApplyBandpassFilterPercentage(perc);
    fft_band.SaveMagnitudeToCSV("../OUTPUT_RESULT/csv_output/fft_magnitude_filtered_" + std::to_string(static_cast<int>(perc)) + "p_band.csv", true);
    fft_band.ComputeIFFT();
    fft_band.ComputeReconstructionError();

    ImageSaver::SaveMagnitudeSpectrum(fft_band.GetFilteredFFT(), size, perc, "band", true);
    ImageSaver::SaveReconstructedImage(fft_band.GetReconstructedImage(), perc, "band");

    band_csv << std::fixed << std::setprecision(50)
             << perc << "," << fft_band.GetError() << "\n";
  }
  band_csv.close();

  // === ERROR CSV EXPORT (ABSOLUTE THRESHOLDS) ===
  cv::Mat input = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
  cv::resize(input, input, cv::Size(size, size));

  ErrorPlot plot_mag(input);
  plot_mag.ComputeErrorsMagnitudeThresholds({10, 50, 100, 200, 500, 1000, 2500, 5000, 7000});
  plot_mag.SaveToCSV("../OUTPUT_RESULT/csv_output/error_vs_threshold_magnitude.csv");

  ErrorPlot plot_band(input);
  plot_band.ComputeErrorsBandThresholds(percentages);
  plot_band.SaveToCSV("../OUTPUT_RESULT/csv_output/error_vs_threshold_band.csv");

  std::cout << "All processing complete. Images and CSVs saved." << std::endl;

  MPI_Finalize();
  return 0;
}