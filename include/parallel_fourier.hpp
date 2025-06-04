#ifndef FFT_ITERATIVE_FOURIER_HPP
#define FFT_ITERATIVE_FOURIER_HPP

#include <abstract_transform.hpp>
#include "../CUDA_FFT/1D/header.hpp"

template <typename T>
class ParallelFourier
{
protected:
public:
    T input;
    T output;
    std::chrono::duration<double> execution_time;
    explicit ParallelFourier(const T &in) : input(in) {}

    void compute()
    {
        computation(this->input);
    }

    virtual void executionTime() const
    {
        std::cout << "GPU-FFT time: "
                  << this->execution_time.count() << " seconds" << std::endl;
    }

private:
    void computation(const T &input)
    {
        auto result = kernel_direct_fft(input);
        this->output = result.first;
        this->execution_time = result.second;
    }
};

#endif // FFT_ITERATIVE_FOURIER_HPP
