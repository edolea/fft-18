#ifndef FFT_ABSTRACT_TRANSFORM_HPP
#define FFT_ABSTRACT_TRANSFORM_HPP

#include <vector>
#include <complex>
#include <ostream>
#include <type_traits>

template <typename T>
concept ComplexVectorPre = std::is_same_v<T, std::vector<std::complex<float>>> ||
                        std::is_same_v<T, std::vector<std::complex<double>>>;

template <typename T>
concept ComplexVector = ComplexVectorPre<T> && requires(const T& vec){
    { isPowerOfTwo(vec.size()) } -> std::convertible_to<bool>;
};

constexpr bool isPowerOfTwo(size_t n){
    return n > 0 && (n & (n-1)) == 0; // very efficient bc single bitwise operation
}
// template specialization needed to check input
template <ComplexVector T>
class BaseTransform
{
protected:
    T input;
    T output;

public:
    explicit BaseTransform(const T& input) : input(input) {}
    BaseTransform() = default;

    virtual void compute(const T& input) = 0;
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

#endif //FFT_ABSTRACT_TRANSFORM_HPP