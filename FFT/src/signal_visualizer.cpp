#include "signal_visualizer.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <vector>

bool SignalVisualizer::exportToCSV(const Signal& signal,
                                   const std::string& filename,
                                   double sample_rate,
                                   bool is_frequency_domain) {
    std::ofstream file(filename);
    if (!file) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return false;
    }

    // Write CSV header
    if (is_frequency_domain) {
        file << "Frequency,Magnitude,Phase\n";
    } else {
        file << "Time,Real,Imaginary,Magnitude\n";
    }

    // Write data rows
    for (size_t i = 0; i < signal.size(); ++i) {
        double x_value;
        if (is_frequency_domain) {
            // Calculate frequency in Hz
            if (i <= signal.size() / 2) {
                x_value = i * sample_rate / signal.size();
            } else {
                // Negative frequencies (for complex FFT)
                x_value = (i - signal.size()) * sample_rate / signal.size();
            }

            double magnitude = std::abs(signal[i]);
            double phase = std::arg(signal[i]);
            file << x_value << "," << magnitude << "," << phase << "\n";
        } else {
            // Time domain - calculate time in seconds
            x_value = static_cast<double>(i) / sample_rate;
            double real = signal[i].real();
            double imag = signal[i].imag();
            double magnitude = std::abs(signal[i]);
            file << x_value << "," << real << "," << imag << "," << magnitude << "\n";
        }
    }

    file.close();
    return true;
}

bool SignalVisualizer::exportSpectrogramToCSV(const std::vector<Signal>& spectrogram,
                                              const std::string& filename,
                                              double time_step,
                                              double sample_rate) {
    if (spectrogram.empty()) {
        std::cerr << "Empty spectrogram data" << std::endl;
        return false;
    }

    std::ofstream file(filename);
    if (!file) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return false;
    }

    // Write header with time positions
    file << "Frequency";
    for (size_t t = 0; t < spectrogram.size(); ++t) {
        file << "," << t * time_step;
    }
    file << "\n";

    // Get the size of frequency bins from the first frame
    size_t freq_bins = spectrogram[0].size() / 2 + 1;

    // Write frequency rows
    for (size_t i = 0; i < freq_bins; ++i) {
        // Calculate frequency in Hz
        double freq = i * sample_rate / (2 * freq_bins - 2);
        file << freq;

        // Write magnitude for each time frame
        for (const auto& frame : spectrogram) {
            if (i < frame.size()) {
                file << "," << std::abs(frame[i]);
            } else {
                file << ",0";
            }
        }
        file << "\n";
    }

    file.close();
    return true;
}

void SignalVisualizer::plotASCII(const Signal& signal,
                                 size_t width,
                                 size_t height,
                                 bool real_part) {
    if (signal.empty()) {
        std::cout << "Empty signal - nothing to plot" << std::endl;
        return;
    }

    // Find min and max values for scaling
    double min_val = findMin(signal, real_part);
    double max_val = findMax(signal, real_part);

    // Ensure we have a range to work with
    if (std::abs(max_val - min_val) < 1e-10) {
        max_val = min_val + 1.0;
    }

    // Create plot buffer (filled with spaces)
    std::vector<std::vector<char>> plot(height, std::vector<char>(width, ' '));

    // Draw axis
    int zero_line = static_cast<int>((0 - min_val) / (max_val - min_val) * (height - 1));
    if (zero_line >= 0 && zero_line < static_cast<int>(height)) {
        for (size_t i = 0; i < width; ++i) {
            plot[zero_line][i] = '-';
        }
    }

    // Plot signal points
    for (size_t i = 0; i < std::min(width, signal.size()); ++i) {
        double sample_idx = static_cast<double>(i) / width * signal.size();
        size_t idx = static_cast<size_t>(sample_idx);

        double value;
        if (real_part) {
            value = signal[idx].real();
        } else {
            value = std::abs(signal[idx]);
        }

        // Scale to plot height
        int y = static_cast<int>((value - min_val) / (max_val - min_val) * (height - 1));
        y = std::max(0, std::min(static_cast<int>(height) - 1, y));

        // Plot the point
        plot[height - 1 - y][i] = '*';
    }

    // Print the plot
    std::cout << std::string(width + 2, '-') << std::endl;
    for (const auto& row : plot) {
        std::cout << "|";
        for (char c : row) {
            std::cout << c;
        }
        std::cout << "|" << std::endl;
    }
    std::cout << std::string(width + 2, '-') << std::endl;

    // Print scale
    std::cout << "Min: " << min_val << ", Max: " << max_val
              << (real_part ? " (Real part)" : " (Magnitude)") << std::endl;
}

double SignalVisualizer::findMin(const Signal& signal, bool real_part) {
    double min_val = std::numeric_limits<double>::max();
    for (const auto& sample : signal) {
        double value = real_part ? sample.real() : std::abs(sample);
        min_val = std::min(min_val, value);
    }
    return min_val;
}

double SignalVisualizer::findMax(const Signal& signal, bool real_part) {
    double max_val = std::numeric_limits<double>::lowest();
    for (const auto& sample : signal) {
        double value = real_part ? sample.real() : std::abs(sample);
        max_val = std::max(max_val, value);
    }
    return max_val;
}