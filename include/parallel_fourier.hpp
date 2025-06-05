#ifndef FFT_ITERATIVE_FOURIER_HPP
#define FFT_ITERATIVE_FOURIER_HPP

#include <abstract_transform.hpp>
#include "../CUDA_FFT/1D/header.hpp"

template <typename T = doubleVector>
class ParallelFourier
{
protected:
    T input;
    T output;
    std::chrono::duration<double> time;

public:
    explicit ParallelFourier(const T &in) : input(in) {}

    void computeDir()
    {
        computationDir(this->input);
    }

    void computeInv(const T &inv_input) {
        computationInv(inv_input);
    }

    virtual void executionTime() const
    {
        std::cout << "GPU-FFT time: "
                  << this->time.count() << " seconds" << std::endl;
    }

    const std::chrono::duration<double> &getTime() const
    {
        return time;
    }

    const T &getOutput() const
    {
        return output;
    }
    const T &getInput() const
    {
        return input;
    }

private:
    void computationDir(const T &input)
    {
        auto result = kernel_direct_fft(input);
        this->output = result.first;
        this->time = result.second;
    }
    void computationInv(const T &input)
    {
       this->output  = kernel_inverse_fft(input);
    }
};

#endif // FFT_ITERATIVE_FOURIER_HPP
