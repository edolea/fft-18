#ifndef FFT_VECTOR_GENERATOR_HPP
#define FFT_VECTOR_GENERATOR_HPP

#include <vector>
#include <complex>
#include <random>
#include <cassert>
#include "abstract_transform.hpp" // To ensure it aligns with the ComplexVector concept

class RandomVectorGenerator {
    template<typename T>
    static constexpr T simple_rand(T seed, size_t index) {
        // Simple LCG: next = (a * current + c) % m
        T value = seed;
        for (size_t i = 0; i < index + 1; ++i)
            value = std::fmod(1103515245 * value + 12345, static_cast<T>(1u << 31));

        return value;
    }

public:
    // failed try to make random generator at compile time --> must use array to have it
    template<ComplexVector T>
    static constexpr T make_random_vector(uint32_t seed, size_t size) {
        assert(isPowerOfTwo(size));
        T randomVector(size);

        using ValueType = typename T::value_type::value_type;
        for (size_t i = 0; i < size; ++i) {
            randomVector[i] = std::complex<ValueType>(simple_rand<ValueType>(seed, i), 0);
        }
        return randomVector;
    }

    // Generate a random vector of complex numbers with a size that is a power of two
    template <ComplexVector T>
    static T generate(size_t size) {
        assert(isPowerOfTwo(size));

        // Random number generators for real and imaginary parts
        // std::random_device rd;
        // std::mt19937 gen(rd()); // NON fixed seed
        std::mt19937 gen(42);  // fixed seed for testing --> all same size inputs are equal
        std::uniform_real_distribution<> dis(-1.0, 1.0); // Range [-1.0, 1.0]

        T randomVector(size);
        for (auto& elem : randomVector)
             elem = std::complex<typename T::value_type::value_type>(dis(gen), dis(gen)); // T::value::value impones double

        return randomVector;
    }

    template <ComplexContainer T>
    static T generate(int dim, double frequency, double amplitude) {
        T input;

        if constexpr (ComplexVectorMatrix<T>) {
            input.reserve(dim);
            for (int i = 0; i < dim; ++i) {
                input.emplace_back();
                input.back().reserve(dim);
                for (int x = 0; x < dim; ++x) {
                    typename T::value_type::value_type::value_type value = amplitude * sin(2.0 * M_PI * frequency * x / dim) *
                                  sin(2.0 * M_PI * frequency * i / dim);
                    input.back().emplace_back(std::complex<typename T::value_type::value_type::value_type>(value, 0.0));
                }
            }
        } else {
            input.reserve(dim);
            for (int i = 0; i < dim; ++i) {
                double value = amplitude * sin(2.0 * M_PI * frequency * i / dim);
                input.emplace_back(std::complex<typename T::value_type::value_type>(value, 0.0));
            }
        }

        return input;
    }
};

#endif //FFT_VECTOR_GENERATOR_HPP