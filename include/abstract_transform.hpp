#ifndef FFT_ABSTRACT_TRANSFORM_HPP
#define FFT_ABSTRACT_TRANSFORM_HPP

#include <vector>
#include <complex>
#include <ostream>
#include <type_traits>
#include <concepts>

template <typename T = std::vector<std::complex<double>> >
concept ComplexVectorPre = std::is_same_v<T, std::vector<std::complex<float>> > ||
                           std::is_same_v<T, std::vector<std::complex<double>> >;

constexpr bool isPowerOfTwo(size_t n){
    return n > 0 && (n & (n-1)) == 0; // very efficient bc single bitwise operation
}

template <typename T>
concept ComplexVector = ComplexVectorPre<T> && requires(const T& vec){
    { isPowerOfTwo(vec.size()) } -> std::convertible_to<bool>;  // useless bc run time check,
                                                                  // but left to double check at compile time if isPowerOfTwo is callable
};

template <typename T>
concept ComplexMatrixPre = std::is_same_v<T, std::vector<std::vector<std::complex<float>>> > ||
                           std::is_same_v<T, std::vector< std::vector<std::complex<double>>> >;

template <typename T>
concept ComplexMatrix = ComplexMatrixPre<T> && requires(const T& mat) {
    { mat.size() } -> std::convertible_to<size_t>;
    { mat[0].size() } -> std::convertible_to<size_t>;
    requires !mat.empty() && !mat[0].empty();
    { isPowerOfTwo(mat.size()) } -> std::convertible_to<bool>;
    { isPowerOfTwo(mat[0].size()) } -> std::convertible_to<bool>;
    requires mat.size() == mat[0].size(); // Ensure matrix is square
};

using complexDouble = std::complex<double>;
using doubleVector = std::vector<complexDouble>;
using doubleMatrix = std::vector<doubleVector>;

using complexFloat = std::complex<float>;
using floatVector = std::vector<complexFloat>;
using floatMatrix = std::vector<floatVector>;


// template specialization needed to check input
template <ComplexVector T>
class BaseTransform {
protected:
    // TODO: remove default constructure and put input constant !!!
    const T input;
    T output;

public:
    explicit BaseTransform(const T& input) : input(input) {
        if (!isPowerOfTwo(input.size())) {
            throw std::invalid_argument("Vector size must be a power of 2.");
        }
    }
    BaseTransform() = default;

    virtual void compute() = 0;

    T getOutput() const {
        return output;
    }

    friend std::ostream &operator<<(std::ostream &os, const BaseTransform<T> &transform) {
        os << "output: " << transform.output;
        return os;
    }

    virtual ~BaseTransform() = default;
};


/* PER EDO: template partial specialization

    template<ComplexVector T>
    class BaseTransform<T*>{

    };

 */

#endif //FFT_ABSTRACT_TRANSFORM_HPP