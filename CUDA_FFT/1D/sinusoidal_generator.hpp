#ifndef SINUSOIDAL_GENERATOR_HPP
#define SINUSOIDAL_GENERATOR_HPP

#include "main_class.hpp"
#include <cmath>

class SinusoidalGenerator : public MainClass
{
public:
    std::vector<std::complex<double>> createInput() override
    {
        std::vector<std::complex<double>> input;
        for (int i = 0; i < N; i++)
        {
            double value = amplitude * sin(2.0 * M_PI * frequency * i / N);
            input.push_back(std::complex<double>(value, 0.0)); // Only real part
        }
        return input;
    }
};

#endif // SINUSOIDAL_GENERATOR_HPP