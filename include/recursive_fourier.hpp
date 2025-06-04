#ifndef FFT_RECURSIVE_FOURIER_HPP
#define FFT_RECURSIVE_FOURIER_HPP

#include "abstract_transform.hpp"

template <typename T>
class RecursiveFourier : public BaseTransform<T>
{
private:
    typename T::value_type initial_w{1, 0};
    void computeDirect(const T &input, T &output) override
    {
        initial_w = {+1, 0};
        output = std::move(algorithm(input));
    }

    void computeInverse(const T &input, T &output) override
    {
        initial_w = {-1, 0};
        output = std::move(algorithm(input));
    }

    T algorithm(const T &x)
    {
        if (x.size() == 1)
        {
            return x;
        }
        else
        {
            int n = x.size();
            typename T::value_type wn{std::cos(2 * M_PI / n), std::sin(2 * M_PI / n)};
            typename T::value_type w{initial_w};
            T x_even;
            T x_odd;
            for (int i = 0; i < n; i++)
            {
                if (i % 2 == 0)
                    x_even.push_back(x[i]);
                else
                    x_odd.push_back(x[i]);
            }

            T y_even{algorithm(x_even)};
            T y_odd{algorithm(x_odd)};

            T y(n);
            for (int i = 0; i < n / 2; i++)
            {
                y[i] = y_even[i] + w * y_odd[i];
                y[i + n / 2] = y_even[i] - w * y_odd[i];
                w = w * wn;
            }
            return y;
        }
    }

public:
    void executionTime() const override
    {
        std::cout << "Recursive FFT time: "
                  << this->time.count() << " seconds" << std::endl;
    }
};

template<typename T>
class RecursiveTemplateFft : public BaseTransform<T> {
protected:
    void compute() override {
        this->output = computation(this->input);
    }

private:
    T computation(const T& x) {
        if(x.size() == 1){
            return x;
        }
        else{
            int n = x.size();
            complexDouble wn{std::cos(2 * M_PI / n), std::sin(2 * M_PI / n)} ;
            complexDouble w{1,0};

            doubleVector x_even;
            doubleVector x_odd;
            for(int i=0; i < n; i++){
                if(i % 2 == 0){
                    x_even.push_back(x[i]);
                }
                else{
                    x_odd.push_back(x[i]);
                }
            }

            doubleVector y_even = computation(x_even);
            doubleVector y_odd = computation(x_odd);

            doubleVector y(n);
            for(int i = 0; i < n/2; i++){
                y[i] = y_even[i] + w * y_odd[i];
                y[i + n/2] = y_even[i] - w * y_odd[i];
                w = w * wn;
            }
            return y;
        }
    }
};

#endif // FFT_RECURSIVE_FOURIER_HPP
