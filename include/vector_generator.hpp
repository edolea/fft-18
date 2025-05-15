#ifndef FFT_VECTOR_GENERATOR_HPP
#define FFT_VECTOR_GENERATOR_HPP

#include <vector>
#include <complex>
#include <random>
#include <stdexcept>
#include <cassert>
#include "abstract_transform.hpp" // To ensure it aligns with the ComplexVector concept

class RandomVectorGenerator {
public:
    // Generate a random vector of complex numbers with a size that is a power of two
    template <ComplexVector T>
    static T generate(size_t size) {
        // Check if size is a power of two
        //if (!isPowerOfTwo(size))
          //  throw std::invalid_argument("Size must be a power of 2.");
        assert(isPowerOfTwo(size));

        // Random number generators for real and imaginary parts
        // std::random_device rd;
        // std::mt19937 gen(rd());
        std::mt19937 gen(42);  // fixed seed for testing --> all same size inputs are equal
        std::uniform_real_distribution<> dis(-1.0, 1.0); // Range [-1.0, 1.0]

        T randomVector(size);
        for (auto& elem : randomVector) {
            elem = std::complex<typename T::value_type::value_type>(dis(gen), dis(gen)); // T::value::value impone double
        }

        return randomVector;
    }

    template <ComplexVector T>
    static T generate(int dim, double frequency, double amplitude) {
        T input;
        for (int i = 0; i < dim; ++i)
        {
            double value = amplitude * sin(2.0 * M_PI * frequency * i / dim);
            input.emplace_back(std::complex<typename T::value_type::value_type>(value, 0.0)); // Only real part
        }
        return input;
    }
};

/*   THIS VERSION HAS FIXED SEED FOR REPRODUCIBILITY
#ifndef FFT_RANDOM_VECTOR_GENERATOR_HPP
#define FFT_RANDOM_VECTOR_GENERATOR_HPP

#include <vector>
#include <complex>
#include <random>
#include <cstddef> // for size_t
#include <stdexcept>
#include "abstract_transform.hpp" // To ensure it aligns with the ComplexVector concept

class RandomVectorGenerator {
private:
    std::mt19937 generator; // Mersenne Twister random number generator
    std::uniform_real_distribution<> distribution;

public:
    // Constructor with optional seed
    explicit RandomVectorGenerator(unsigned int seed = std::random_device{}())
        : generator(seed), distribution(-1.0, 1.0) {}

    // Generate a random vector of complex numbers with a size that is a power of two
    template <typename T>
    T generate(size_t size) {
        // Check if size is a power of two
        if (!isPowerOfTwo(size)) {
            throw std::invalid_argument("Size must be a power of 2.");
        }

        T randomVector(size);
        for (auto& elem : randomVector) {
            elem = std::complex<typename T::value_type::value_type>(
                distribution(generator), distribution(generator));
        }

        return randomVector;
    }

    // Reset the generator with a new seed
    void resetSeed(unsigned int seed) {
        generator.seed(seed);
    }
};

#endif // FFT_RANDOM_VECTOR_GENERATOR_HPP
 */

#endif //FFT_VECTOR_GENERATOR_HPP
