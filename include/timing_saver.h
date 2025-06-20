#ifndef TIMING_SAVER_H
#define TIMING_SAVER_H

#include <fstream>
#include <vector>
#include <string>
#include <utility> // For std::pair
#include <tuple>   // For std::tuple
#include <iostream> // For std::cerr
#include <filesystem> // For std::filesystem::create_directory (C++17+)

class TimingSaver {
public:
    // Static methods to save timings
    static void saveFFTTimings(
        const std::vector<std::pair<int, double>>& timings_fft,
        const std::string& outputFilename,
        const std::string& outputDirectory = "../OUTPUT_RESULT", // Default directory for static methods
        const std::string& description = "Fast Fourier Transform (N, Time)") {

        create_directory_if_not_exists(outputDirectory);
        saveTimingsToFile(timings_fft, outputFilename, outputDirectory, description);
    }

    static void saveParallelFFTTimings(
        const std::vector<std::tuple<int, double, int>>& timings_parallel_fft,
        const std::string& outputFilename,
        const std::string& outputDirectory = "../OUTPUT_RESULT/CUDA", // Default directory for static methods
        const std::string& description = "Fast Fourier Transform (N, Time, Threads)") {

        create_directory_if_not_exists(outputDirectory);
        saveParallelTimingsToFile(timings_parallel_fft, outputFilename, outputDirectory, description);
    }

private:
    // Private static helper methods
    static void create_directory_if_not_exists(const std::string& path) {
        if (!std::filesystem::exists(path)) {
            try {
                std::filesystem::create_directories(path);
                std::cout << "Directory created: " << path << "\n";
            } catch (const std::filesystem::filesystem_error& e) {
                std::cerr << "Error creating directory " << path << ": " << e.what() << "\n";
            }
        }
    }

    static void saveTimingsToFile(
        const std::vector<std::pair<int, double>>& timings,
        const std::string& filename,
        const std::string& directory, // Directory passed as a parameter
        const std::string& description) {

        std::string fullPath = directory + "/" + filename;
        std::ofstream outFile(fullPath);
        if (outFile.is_open()) {
            for (const auto& timing : timings) {
                outFile << timing.first << " " << timing.second << "\n";
            }
            outFile.close();
        } else {
            std::cerr << "Unable to open file for writing " << description << " timings: " << fullPath << "\n";
        }
    }

    static void saveParallelTimingsToFile(
        const std::vector<std::tuple<int, double, int>>& timings,
        const std::string& filename,
        const std::string& directory, // Directory passed as a parameter
        const std::string& description) {

        std::string fullPath = directory + "/" + filename;
        std::ofstream outFile(fullPath);
        if (outFile.is_open()) {
            for (const auto& timing : timings) {
                outFile << std::get<0>(timing) << " "
                        << std::get<1>(timing) << " "
                        << std::get<2>(timing) << "\n";
            }
            outFile.close();
        } else {
            std::cerr << "Unable to open file for writing " << description << " timings: " << fullPath << "\n";
        }
    }
};

#endif // TIMING_SAVER_H

