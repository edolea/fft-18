#ifndef FFT_CUDA_FOURIER_HPP
#define FFT_CUDA_FOURIER_HPP

#include "header.hpp"
#include "static_header.hpp"
#include "../../include/abstract_transform.hpp"


template <typename T = std::vector<std::complex<double>>> // fixme: eccessivo, ma per sicurezza
class CudaFourier : public BaseTransform<T> {
    std::chrono::duration<double> time;
    std::chrono::duration<double> time_no_malloc;
    std::chrono::duration<double> time_inverse;
    std::chrono::duration<double> time_inverse_no_malloc;

public:
    explicit CudaFourier(const T& input) : BaseTransform<T>(input) {}

    void compute() override {
        this->output = computation(this->input);
    }

private:
    // FRA: in sostanza questo è il tuo vecchio main
    T computation() { // todo: vedi se mettere const
        // INITIAL COMPUTATION TO WARM UP THE GPU AND AVOID TIMING DISCREPANCIES
        T cuda_output_vector = kernel_direct_fft().first;

        // DIRECT TRANSFORM
        cuda_output_vector = kernel_direct_fft();
        plot_fft_result(cuda_output_vector);

        // INVERSE TRANSFORM
        T cuda_inverse_output_vector = kernel_inverse_fft();
    }



    // TODO x EDO: inverse ancora da vedere come metterla.
    //  prob userò un overload con un bool che non verrà utilizzato

    /**
 * @brief Performs a direct Fast Fourier Transform (FFT) on the input data using CUDA.
 *
 * This function allocates memory on the GPU, copies the input data to the GPU, and performs
 * the FFT using CUDA kernels. The execution time is measured and printed to the console.
 *
 * @param input A vector of complex numbers representing the input data to be transformed, taken directly from the attribute input
 * @return A vector of complex numbers representing the transformed data.
 */
    T kernel_direct_fft()
    {
        const auto& input{this->input};
        // CUDA malloc initialization
        int *prova;
        cudaMallocManaged((void **)&prova, sizeof(int));

        int grid_size = input.size() / THREAD_PER_BLOCK;
        if (input.size() % THREAD_PER_BLOCK != 0)
            grid_size++;

        std::cout << "\nGPU_PARALLEL   DIRECT_FFT\n";
        auto start_parallel = std::chrono::high_resolution_clock::now();
        int log_n = (int)(log(input.size()) / log(2)); // perxhè non usi N anzichè input.size()
        cuDoubleComplex *a;
        cuDoubleComplex *y;
        cuDoubleComplex *x;
        cuDoubleComplex *t;
        int *atomic_array;
        int *first_computation_j;

        cudaMallocManaged((void **)&a, sizeof(cuDoubleComplex) * (THREAD_PER_BLOCK * grid_size));
        cudaMallocManaged((void **)&y, sizeof(cuDoubleComplex) * (THREAD_PER_BLOCK * grid_size));
        cudaMallocManaged((void **)&x, sizeof(cuDoubleComplex) * (THREAD_PER_BLOCK * grid_size));
        cudaMallocManaged((void **)&t, sizeof(cuDoubleComplex) * (THREAD_PER_BLOCK * grid_size));
        cudaMallocManaged((void **)&atomic_array, sizeof(int) * (log_n + 1));
        cudaMallocManaged((void **)&first_computation_j, sizeof(int));

        // Copy data from std::complex to cuDoubleComplex
        for (size_t i = 0; i < (THREAD_PER_BLOCK * grid_size); ++i)
        {
            if (i < input.size())
                a[i] = make_cuDoubleComplex(input[i].real(), input[i].imag());
            else
                a[i] = make_cuDoubleComplex(0, 0);
        }

        auto malloc_complete = std::chrono::high_resolution_clock::now();
        gpu_fft(grid_size, log_n, a, y, x, t, atomic_array, first_computation_j);
        auto end_parallel = std::chrono::high_resolution_clock::now();

        time = end_parallel - start_parallel;
        time_no_malloc = end_parallel - malloc_complete;
        std::cout << "Parallel FFT execution time: " << time.count() << " seconds" << std::endl;
        std::cout << "Parallel FFT execution time WITHOUT MALLOC: " << time_no_malloc.count() << " seconds" << std::endl;

        ComplexVector auto result = cuDoubleComplexToVector(y, input.size());
        return result;
    }

    /**
     * @brief Performs an inverse Fast Fourier Transform (IFFT) on the input data using CUDA.
     *
     * This function allocates memory on the GPU, copies the input data to the GPU, and performs
     * the IFFT using CUDA kernels. The execution time is measured and printed to the console.
     * The result is normalized by dividing each element by the size of the input.
     *
     * @param input A vector of complex numbers representing the input data to be transformed, taken directly from the attribute input
     * @return A vector of complex numbers representing the transformed data.
     */
    T kernel_inverse_fft()
    {
        const auto& input{this->input};
        // CUDA malloc initialization
        int *prova;
        cudaMallocManaged((void **)&prova, sizeof(int));

        int grid_size = input.size() / THREAD_PER_BLOCK;
        if (input.size() % THREAD_PER_BLOCK != 0)
            grid_size++;

        std::cout << "\nGPU_PARALLEL INVERSE_FFT\n";
        auto start_parallel = std::chrono::high_resolution_clock::now();
        int log_n = (int)(log(input.size()) / log(2));
        cuDoubleComplex *a;
        cuDoubleComplex *y;
        cuDoubleComplex *x;
        cuDoubleComplex *t;
        int *atomic_array;
        int *first_computation_j;

        cudaMallocManaged((void **)&a, sizeof(cuDoubleComplex) * (THREAD_PER_BLOCK * grid_size));
        cudaMallocManaged((void **)&y, sizeof(cuDoubleComplex) * (THREAD_PER_BLOCK * grid_size));
        cudaMallocManaged((void **)&x, sizeof(cuDoubleComplex) * (THREAD_PER_BLOCK * grid_size));
        cudaMallocManaged((void **)&t, sizeof(cuDoubleComplex) * (THREAD_PER_BLOCK * grid_size));
        cudaMallocManaged((void **)&atomic_array, sizeof(int) * (log_n + 1));
        cudaMallocManaged((void **)&first_computation_j, sizeof(int));

        // Copy data from std::complex to cuDoubleComplex
        for (size_t i = 0; i < (THREAD_PER_BLOCK * grid_size); ++i)
        {
            if (i < input.size())
                a[i] = make_cuDoubleComplex(input[i].real(), input[i].imag());
            else
                a[i] = make_cuDoubleComplex(0, 0);
        }

        auto malloc_complete = std::chrono::high_resolution_clock::now();
        gpu_inverse_fft(grid_size, log_n, a, y, x, t, atomic_array, first_computation_j);
        auto end_parallel = std::chrono::high_resolution_clock::now();
        time_inverse = end_parallel - start_parallel;
        time_inverse_no_malloc = end_parallel - malloc_complete;
        std::cout << "Parallel IFFT execution time: " << time_inverse.count() << " seconds" << std::endl;
        std::cout << "Parallel IFFT execution time WITHOUT MALLOC: " << time_inverse_no_malloc.count() << " seconds" << std::endl;

        ComplexVector auto result = cuDoubleComplexToVector(y, input.size());
        // Normalize the IFFT result
        for (auto &val : result)
        {
            val /= input.size(); // Divide by N
        }

        return result;
    }

};

#endif //FFT_CUDA_FOURIER_HPP
