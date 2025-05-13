#ifndef GAUSSIAN_2D_HPP
#define GAUSSIAN_2D_HPP

#include "main_class_2d.hpp"
#include <cmath>

class Gaussian2D : public MainClass2D
{
public:
    std::vector<std::vector<std::complex<double>>> createInput() override
    {
        std::vector<std::vector<std::complex<double>>> input(rows, std::vector<std::complex<double>>(cols));
        double sigma_x = 0.5; // Example sigma along X axis
        double sigma_y = 0.5; // Example sigma along Y axis
        double center_x = rows / 2.0;
        double center_y = cols / 2.0;

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                double x_term = pow(i - center_x, 2) / (2 * sigma_x * sigma_x);
                double y_term = pow(j - center_y, 2) / (2 * sigma_y * sigma_y);
                double value = exp(-(x_term + y_term));
                input[i][j] = std::complex<double>(value, 0.0);
            }
        }

        return input;
    }
};

#endif // GAUSSIAN_2D_HPP