#ifndef FFT_MATRIX_UTILS_HPP
#define FFT_MATRIX_UTILS_HPP

#include "abstract_transform.hpp"

template <ComplexMatrix T>
class MatrixUtils {
public:
    // Transpose function (you already have this in CUDA code)
    static T transpose(const T& matrix) {
        if (matrix.empty() || matrix[0].empty()) {
            return T{};
        }

        size_t rows = matrix.size();
        size_t cols = matrix[0].size();
        T transposed(cols, typename T::value_type(rows));

        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                transposed[j][i] = matrix[i][j];
            }
        }
        return transposed;
    }

    // Matrix comparison (you already have this)
    static bool compare(const T& mat1, const T& mat2, double tolerance = 1e-9) {
        if (mat1.size() != mat2.size() || mat1[0].size() != mat2[0].size()) {
            return false;
        }

        for (size_t i = 0; i < mat1.size(); ++i) {
            for (size_t j = 0; j < mat1[i].size(); ++j) {
                auto diff_real = std::abs(mat1[i][j].real() - mat2[i][j].real());
                auto diff_imag = std::abs(mat1[i][j].imag() - mat2[i][j].imag());
                if (diff_real > tolerance || diff_imag > tolerance) {
                    return false;
                }
            }
        }
        return true;
    }

    // Create zero matrix
    static T zeros(size_t rows, size_t cols) {
        return T(rows, typename T::value_type(cols, typename T::value_type::value_type{0, 0}));
    }

    // Create identity matrix
    static T identity(size_t size) {
        auto mat = zeros(size, size);
        for (size_t i = 0; i < size; ++i) {
            mat[i][i] = typename T::value_type::value_type{1, 0};
        }
        return mat;
    }
};

#endif //FFT_MATRIX_UTILS_HPP