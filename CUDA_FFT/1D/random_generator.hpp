#ifndef RANDOM_GENERATOR_HPP
#define RANDOM_GENERATOR_HPP

#include "main_class.hpp"
#include <cstdlib>

class RandomGenerator : public MainClass
{
public:
    std::vector<std::complex<double>> createInput() override
    {
        std::vector<std::complex<double>> input;
        for (int i = 0; i < N; i++)
        {
            double real_part = (rand() % (RAND_MAX)) / static_cast<double>(RAND_MAX) * 2.0 - 1.0;
            double imag_part = (rand() % (RAND_MAX)) / static_cast<double>(RAND_MAX) * 2.0 - 1.0;
            input.push_back(std::complex<double>(real_part, imag_part));
        }
        return input;
    }
};

#endif // RANDOM_GENERATOR_HPP