//TODO: nice concept but i think i'll trash it

/*
#ifndef FFT_VECTOR_TEMPLATE_HPP
#define FFT_VECTOR_TEMPLATE_HPP

#include <vector>

template <typename T>
class ComplexVector {
    static_assert(
            std::is_same<T, std::complex<float>>::value ||
            std::is_same<T, std::complex<double>>::value,
            "compile time assertion of input type failed"
    );

public:
    using inputType = std::vector<T>;

    ComplexVector(const inputType& vec) : vector(vec) {}

    inputType& getVec(){
        return vector;
    }

private:
    inputType vector;
};


#endif //FFT_VECTOR_TEMPLATE_HPP
*/