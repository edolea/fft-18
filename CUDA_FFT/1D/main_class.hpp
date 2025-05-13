#ifndef MAIN_CLASS_HPP
#define MAIN_CLASS_HPP

#include <vector>
#include <complex>

class MainClass
{
public:
    static int N;            // Dimension for FFT
    static double frequency; // Frequency for sinusoidal input
    static double amplitude; // Amplitude for sinusoidal input

    static void initializeParameters(int dim, double freq, double amp)
    {
        N = dim;
        frequency = freq;
        amplitude = amp;
    }

    virtual std::vector<std::complex<double>> createInput() = 0;
};

#endif // MAIN_CLASS_HPP