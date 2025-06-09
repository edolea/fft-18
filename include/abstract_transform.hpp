#ifndef FFT_ABSTRACT_TRANSFORM_HPP
#define FFT_ABSTRACT_TRANSFORM_HPP

#include <vector>
#include <array>
#include <complex>
#include <type_traits>
#include <concepts>
#include <chrono>

constexpr bool isPowerOfTwo(size_t n) {
    return n > 0 && (n & (n - 1)) == 0; // very efficient bc single bitwise operation
}

template <typename T = std::vector<std::complex<double>>>
concept ComplexVector = std::is_same_v<T, std::vector<std::complex<float>>> ||
                        std::is_same_v<T, std::vector<std::complex<double>>>;

template <typename T, size_t n>
concept ComplexArray = std::is_same_v<T, std::array<std::complex<float>, n>> ||
                       std::is_same_v<T, std::array<std::complex<double>, n>>;

template <typename T>
concept ComplexVectorMatrixPre = std::is_same_v<T, std::vector<std::vector<std::complex<float>>>> ||
                                 std::is_same_v<T, std::vector<std::vector<std::complex<double>>>>;

template <typename T, size_t n>
concept ComplexArrayMatrixPre = std::is_same_v<T, std::array<std::array<std::complex<float>, n>, n>> ||
                                std::is_same_v<T, std::array<std::array<std::complex<double>, n>, n>>;

template <typename T>
concept ComplexVectorMatrix = ComplexVectorMatrixPre<T> && requires(const T &mat) {
    { mat.size() } -> std::convertible_to<size_t>;
    { mat[0].size() } -> std::convertible_to<size_t>;
    { mat.empty() } -> std::convertible_to<bool>;
    { mat[0].empty() } -> std::convertible_to<bool>;
    { isPowerOfTwo(mat.size()) } -> std::convertible_to<bool>;
    { isPowerOfTwo(mat[0].size()) } -> std::convertible_to<bool>;
    { mat.size() == mat[0].size() } -> std::convertible_to<bool>;
};

template <typename T, size_t n>
concept ComplexArrayMatrix = ComplexArrayMatrixPre<T, n> && requires(const T &mat) {
    { mat.size() } -> std::convertible_to<size_t>;
    { mat[0].size() } -> std::convertible_to<size_t>;
    requires !mat.empty() && !mat[0].empty();
    { isPowerOfTwo(mat.size()) } -> std::convertible_to<bool>;
    { isPowerOfTwo(mat[0].size()) } -> std::convertible_to<bool>;
};

using complexDouble = std::complex<double>;
using doubleVector = std::vector<complexDouble>;
using doubleMatrix = std::vector<doubleVector>;

using complexFloat = std::complex<float>;
using floatVector = std::vector<complexFloat>;
using floatMatrix = std::vector<floatVector>;

using TimeDuration = std::chrono::duration<double>;

// combined concept for both vectors and matrices
template <typename T>
concept ComplexContainer = ComplexVector<T> || ComplexVectorMatrix<T>;

// TODO: to implement in abstract
template <typename T, size_t n>
concept ComplexArrayContainer = ComplexArray<T, n> || ComplexArrayMatrix<T, n>;

template <ComplexContainer T>
class BaseTransform {
protected:
    std::chrono::duration<double> time{};

    // private virtual interface
    virtual void computeImpl(const T &input, T &output, const bool&) = 0;

    // TODO: eliminate old inefficient transpose
    template <ComplexVectorMatrix Matrix>
    Matrix transpose2D(const Matrix &input) {
        if (input.empty()) return {};

        size_t rows = input.size();
        size_t cols = input[0].size();
        Matrix output(cols, typename Matrix::value_type(rows));

        for (size_t i = 0; i < rows; ++i)
            for (size_t j = 0; j < cols; ++j)
                output[j][i] = input[i][j];

        return output;
    }

    template <ComplexVectorMatrix Matrix>
    void transpose2D_more_efficient(Matrix &input) {
        if (input.empty() || input.size() != input[0].size()) return; // Ensure square matrix

        const size_t n = input.size();
        for (size_t i = 0; i < n; ++i)
            for (size_t j = i + 1; j < n; ++j)
                std::swap(input[i][j], input[j][i]);
    }

public:
    // Public non-virtual interface with timing
    void compute(const T &input, T &output, const bool& direct=true) {
        if constexpr (ComplexVector<T>) {
            assert(isPowerOfTwo(input.size()));
        } else if constexpr (ComplexVectorMatrix<T>) {
            assert(!input.empty() && !input[0].empty());
            assert(isPowerOfTwo(input.size()));
            assert(isPowerOfTwo(input[0].size()));
            assert(input.size() == input[0].size()); // Ensure square matrix
        }
        const auto start = std::chrono::high_resolution_clock::now();
        computeImpl(input, output, direct);
        const auto end = std::chrono::high_resolution_clock::now();
        time = end - start;
    }

    virtual void executionTime() const {
        std::cout << "Abstract FFT time: "
                  << this->time.count() << " seconds" << std::endl;
    }

    const std::chrono::duration<double> &getTime() const {
        return time;
    }

    virtual ~BaseTransform() = default;
};

/* PER EDO: template partial specialization

    template<ComplexVector T>
    class BaseTransform<T*>{

    };

 */

#endif // FFT_ABSTRACT_TRANSFORM_HPP