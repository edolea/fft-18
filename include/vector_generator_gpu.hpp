#ifndef FFT_VECTOR_GENERATOR_HPP
#define FFT_VECTOR_GENERATOR_HPP

#include <vector>
#include <complex>
#include <random>
#include <stdexcept>
#include <cassert>

class RandomVectorGenerator
{
public:
    int N;
    // Generate a random vector of complex numbers with a size that is a power of two
    template <typename T>
    static T generate(size_t size)
    {
        std::mt19937 gen(42);                            // fixed seed for testing --> all same size inputs are equal
        std::uniform_real_distribution<> dis(-1.0, 1.0); // Range [-1.0, 1.0]

        T randomVector(size);
        for (auto &elem : randomVector)
        {
            elem = std::complex<typename T::value_type::value_type>(dis(gen), dis(gen)); // T::value::value impone double
        }

        return randomVector;
    }

    template <typename T>
    static T generate(int dim, double frequency, double amplitude)
    {
        T input;
        for (int i = 0; i < dim; ++i)
        {
            double value = amplitude * sin(2.0 * M_PI * frequency * i / dim);
            input.emplace_back(std::complex<typename T::value_type::value_type>(value, 0.0)); // Only real part
        }
        return input;
    }
};

#endif // FFT_VECTOR_GENERATOR_HPP
