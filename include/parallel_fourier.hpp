#ifndef FFT_ITERATIVE_FOURIER_HPP
#define FFT_ITERATIVE_FOURIER_HPP

#include <abstract_transform.hpp>
#include "../CUDA_FFT/1D/header.hpp"
#include "../CUDA_FFT/2D/header.hpp"

template <typename T = doubleVector>
class ParallelFourier
{
protected:
    T input;
    T output;
    std::chrono::duration<double> time;

public:
    explicit ParallelFourier(const T &in) : input(in) {}

    void compute(const T &input_i, const bool &isDirect = true)
    {
        if constexpr (ComplexVector<T>) {
            // 1D FFT
            if (isDirect) {
                this->computationDir1D(this->input);
            } else {
                this->computationInv1D(input_i);
            }
        }else {
           if (isDirect) {this->computationDir2d(this->input);}
           else {this->computationInv2d(input_i);}
        }



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
    void computationDir1D(const T &input)
    {
        auto result = kernel_direct_fft(input);
        this->output = result.first;
        this->time = result.second;
    }
    void computationInv1D(const T &input)
    {
       this->output  = kernel_inverse_fft(input);
    }
    void computationDir2d(const T &input)
    {
        auto result = direct_fft_2d(input);
        this->output = result.first;
        this->time = result.second;
    }
    void computationInv2d(const T &input) {
        this->output = inverse_fft_2d(input);
    }
};

#endif // FFT_ITERATIVE_FOURIER_HPP
