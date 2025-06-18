//
// Created by francesco-virgulti on 6/18/25.
//

#ifndef FFT_RESULT_SAVER_H
#define FFT_RESULT_SAVER_H

#include <vector>
#include <complex>
#include <string>
#include <fstream>
#include <iostream>
#include <filesystem>

class FFTResultSaver
{
public:
    static void save2DFFTResult(
        const std::vector<std::vector<std::complex<double>>> &fft_result,
        const std::string &outputFilename = "fft_output_2d.csv",
        const std::string &outputDirectory = "../OUTPUT_RESULT_2D")
    {

        create_directory_if_not_exists(outputDirectory);
        saveToFile(fft_result, outputFilename, outputDirectory);
    }

private:
    static void create_directory_if_not_exists(const std::string &path)
    {
        if (!std::filesystem::exists(path))
        {
            try
            {
                std::filesystem::create_directories(path);
                std::cout << "Directory created: " << path << "\n";
            }
            catch (const std::filesystem::filesystem_error &e)
            {
                std::cerr << "Error creating directory " << path << ": " << e.what() << "\n";
            }
        }
    }

    static void saveToFile(
        const std::vector<std::vector<std::complex<double>>> &fft_result,
        const std::string &filename,
        const std::string &directory)
    {

        std::string fullPath = directory + "/" + filename;
        std::ofstream file(fullPath);
        if (!file.is_open())
        {
            std::cerr << "Errore: impossibile aprire il file " << fullPath << std::endl;
            return;
        }

        for (const auto &row : fft_result)
        {
            for (size_t j = 0; j < row.size(); ++j)
            {
                const auto &value = row[j];
                if (value.imag() >= 0)
                    file << value.real() << "+" << value.imag() << "j";
                else
                    file << value.real() << value.imag() << "j";

                if (j < row.size() - 1)
                    file << ",";
            }
            file << "\n";
        }

        file.close();
        std::cout << "2D FFT data saved to: " << fullPath << std::endl;
    }
};

#endif // FFT_RESULT_SAVER_H
