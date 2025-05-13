#ifndef MAIN_CLASS_2D_HPP
#define MAIN_CLASS_2D_HPP

#include <vector>
#include <complex>

class MainClass2D
{
public:
    static int rows; // Number of rows for 2D FFT
    static int cols; // Number of columns for 2D FFT
    static int max_problem_size;
    static std::vector<double> PROBLEM_SIZES;

    static void initializeParameters(int r, int c)
    {
        rows = r;
        cols = c;
    }

    void initialized_size()
    {
        for (int i = 0; i <= max_problem_size; i++)
        {
            PROBLEM_SIZES.push_back(pow(2, i));
        }
    }

    virtual std::vector<std::vector<std::complex<double>>>
    createInput() = 0;
};

#endif // MAIN_CLASS_2D_HPP