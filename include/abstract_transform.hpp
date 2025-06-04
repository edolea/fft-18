#ifndef FFT_ABSTRACT_TRANSFORM_HPP
#define FFT_ABSTRACT_TRANSFORM_HPP

#include <vector>
#include <array>
#include <complex>
#include <ostream>
#include <type_traits>
#include <concepts>

constexpr bool isPowerOfTwo(size_t n)
{
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
    requires !mat.empty() && !mat[0].empty();
    { isPowerOfTwo(mat.size()) } -> std::convertible_to<bool>;
    { isPowerOfTwo(mat[0].size()) } -> std::convertible_to<bool>;
    requires mat.size() == mat[0].size(); // Ensure matrix is square
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
class BaseTransform
{
protected:
    std::chrono::duration<double> time{};

    // private virtual interface
    virtual void computeDirect(const T &input, T &output) = 0;
    virtual void computeInverse(const T &input, T &output) = 0;

public:
    // Public non-virtual interface with timing
    void computeDir(const T &input, T &output)
    {
        assert(isPowerOfTwo(input.size()));

        auto start = std::chrono::high_resolution_clock::now();
        computeDirect(input, output);
        auto end = std::chrono::high_resolution_clock::now();
        time = end - start;
    }

    void computeInv(const T &input, T &output)
    {
        assert(isPowerOfTwo(input.size()));
        computeInverse(input, output);
    }

    virtual void executionTime() const
    {
        std::cout << "FFT time: "
                  << this->time.count() << " seconds" << std::endl;
    }

    virtual ~BaseTransform() = default;
};

/* PER EDO: template partial specialization

    template<ComplexVector T>
    class BaseTransform<T*>{

    };

 */

#endif // FFT_ABSTRACT_TRANSFORM_HPP