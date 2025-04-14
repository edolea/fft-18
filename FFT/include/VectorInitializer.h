//
// Created by Edoardo Leali on 26/03/25.
//

#ifndef VECTOR_INITIALIZER_H
#define VECTOR_INITIALIZER_H

#include <vector>
#include <complex>
#include <random>
#include <cmath>

class VectorInitializer {
public:
    static std::vector<std::complex<double>> createRandomVector(size_t size, unsigned int seed = 95) {
        std::mt19937 gen(seed);
        std::uniform_real_distribution<double> dist(0.0, RAND_MAX);

        std::vector<std::complex<double>> vector;
        vector.reserve(size);

        for (size_t i = 0; i < size; i++) {
            vector.push_back(std::complex<double>(dist(gen), dist(gen)));
        }

        return vector;
    }

    static std::vector<std::complex<double>> createTestVector(size_t size) {
        std::vector<std::complex<double>> vector;
        vector.reserve(size);

        for (size_t i = 0; i < size; i++) {
            double angle = 2.0 * M_PI * i / size;
            vector.push_back(std::complex<double>(cos(angle), sin(angle)));
        }

        return vector;
    }
};

#endif // VECTOR_INITIALIZER_H