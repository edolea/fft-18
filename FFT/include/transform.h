#ifndef TRANSFORM_H
#define TRANSFORM_H

#include <vector>
#include <complex>

// Type alias for convenience
using Complex = std::complex<double>;
using Signal = std::vector<Complex>;

/**
 * Abstract base class for signal transforms
 */
class TransformBase {
public:
    virtual ~TransformBase() = default;

    // Pure virtual functions that derived classes must implement
    virtual Signal forward(const Signal& signal) = 0;
    virtual Signal inverse(const Signal& transformed_signal) = 0;

    // Optional method for checking if the transform can handle the given signal
    virtual bool canProcess(const Signal& signal) const = 0;

    // Optional method for getting the name of the transform
    virtual std::string getName() const = 0;
};

#endif // TRANSFORM_H