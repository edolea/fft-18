#ifndef SINUSOIDAL_2D_HPP
#define SINUSOIDAL_2D_HPP

#include "main_class_2d.hpp"
#include <cmath>

class Sinusoidal2D : public MainClass2D
{
public:
    std::vector<std::vector<std::complex<double>>> createInput() override
    {
        std::vector<std::vector<std::complex<double>>> input(rows, std::vector<std::complex<double>>(cols));
        double frequency_x = 5.0; // Example frequency along X axis
        double frequency_y = 5.0; // Example frequency along Y axis

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                double value = sin(2.0 * M_PI * frequency_x * i / rows) * sin(2.0 * M_PI * frequency_y * j / cols);
                input[i][j] = std::complex<double>(value, 0.0); // Only real part
            }
        }

        return input;
    }
};

#endif // SINUSOIDAL_2D_HPP