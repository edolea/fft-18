#ifndef RANDOM_2D_HPP
#define RANDOM_2D_HPP

#include "main_class_2d.hpp"
#include <cstdlib>

class Random2D : public MainClass2D
{
public:
    std::vector<std::vector<std::complex<double>>> createInput() override
    {
        std::vector<std::vector<std::complex<double>>> input(rows, std::vector<std::complex<double>>(cols));
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                double real_part = (rand() % RAND_MAX) / static_cast<double>(RAND_MAX) * 2.0 - 1.0;
                double imag_part = (rand() % RAND_MAX) / static_cast<double>(RAND_MAX) * 2.0 - 1.0;
                input[i][j] = std::complex<double>(real_part, imag_part);
            }
        }

        return input;
    }
};

#endif // RANDOM_2D_HPP