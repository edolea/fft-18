#ifndef FFT_VECTOR_GENERATOR_HPP
#define FFT_VECTOR_GENERATOR_HPP

#include <vector>
#include <complex>
#include <random>
#include <cmath>    // for std::sin and M_PI

class RandomVectorGenerator
{
public:
    // Generate 1D complex vector
    template <typename T>
    static T generate(size_t size)
    {
        std::mt19937 gen(42); // Fixed seed for reproducibility
        std::uniform_real_distribution<> dis(-1.0, 1.0);

        T randomVector(size);
        for (auto &elem : randomVector)
        {
            elem = std::complex<typename T::value_type::value_type>(dis(gen), dis(gen));
        }

        return randomVector;
    }

    // Generate 1D sinusoidal pattern
    template <typename T>
    static T generate(int dim, double frequency, double amplitude)
    {
        T input;
        input.reserve(dim);
        for (int i = 0; i < dim; ++i)
        {
            double value = amplitude * std::sin(2.0 * M_PI * frequency * i / dim);
            input.emplace_back(std::complex<typename T::value_type::value_type>(value, 0.0));
        }
        return input;
    }

    // Generate 2D sinusoidal matrix
    template <typename Matrix>
    static Matrix generate2D(int dim, double frequency, double amplitude)
    {
        using Complex = typename Matrix::value_type::value_type;
        Matrix mat;
        mat.reserve(dim);

        for (int i = 0; i < dim; ++i)
        {
            std::vector<Complex> row;
            row.reserve(dim);
            for (int j = 0; j < dim; ++j)
            {
                double value = amplitude *
                               std::sin(2.0 * M_PI * frequency * j / dim) *
                               std::sin(2.0 * M_PI * frequency * i / dim);
                row.emplace_back(Complex(value, 0.0));
            }
            mat.emplace_back(std::move(row));
        }

        return mat;
    }
};

#endif // FFT_VECTOR_GENERATOR_HPP
