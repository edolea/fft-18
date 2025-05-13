#include "./static_header.hpp"
#include "main_class.hpp"
__device__ void direct_compute_wd_pow_h(double *result_real, double *result_imag, int d, cuDoubleComplex h)
{
    // Extract real and imaginary parts of h
    double real_h = cuCreal(h);
    double imag_h = cuCimag(h);

    // Compute the natural logarithm of wd, ln(wd) = i * (2π/d)
    // double ln_wd_real = 0.0;          // Real part of ln(wd) is 0
    double ln_wd_imag = 2 * M_PI / d; // Imaginary part of ln(wd)

    // Compute h * ln(wd) (correct complex multiplication)
    double exponent_real = -imag_h * ln_wd_imag; // Real part of h * ln(wd)
    double exponent_imag = real_h * ln_wd_imag;  // Imaginary part of h * ln(wd)

    // Compute e^(h * ln(wd)) = e^(exponent_real + i * exponent_imag)
    double magnitude = exp(exponent_real);         // Magnitude = e^(real part of exponent)
    *result_real = magnitude * cos(exponent_imag); // Real part of result
    *result_imag = magnitude * sin(exponent_imag); // Imaginary part of result
    return;
}

__device__ void inverse_compute_wd_pow_h(double *result_real, double *result_imag, int d, cuDoubleComplex h)
{
    // Extract real and imaginary parts of h
    double real_h = cuCreal(h);
    double imag_h = cuCimag(h);

    // Compute the natural logarithm of wd, ln(wd) = -i * (2π/d) for IFFT
    double ln_wd_imag = -2 * M_PI / d; // Reversed sign for IFFT

    // Compute h * ln(wd) (correct complex multiplication)
    double exponent_real = -imag_h * ln_wd_imag; // Real part of h * ln(wd)
    double exponent_imag = real_h * ln_wd_imag;  // Imaginary part of h * ln(wd)

    // Compute e^(h * ln(wd)) = e^(exponent_real + i * exponent_imag)
    double magnitude = exp(exponent_real);         // Magnitude = e^(real part of exponent)
    *result_real = magnitude * cos(exponent_imag); // Real part of result
    *result_imag = magnitude * sin(exponent_imag); // Imaginary part of result
}

__device__ void thread_write(bool direct, int t_x, int d, cuDoubleComplex *y, cuDoubleComplex *x, cuDoubleComplex *t)
{
    double result_real, result_imag;

    bool up_down; // up = 1 down = 0 ( if down it will write on x otherwise it will write on t)
    int tmp_index;
    int p;
    cuDoubleComplex w;

    if ((t_x % d) < d / 2)
    {
        up_down = 0;
    }
    else
        up_down = 1;

    // DOWN CASE
    if (!up_down)
    {
        tmp_index = t_x + d / 2;
        // printf("Thread n: %d, Iteration : %d => up_down = %d, index_of_writing : %d\n",t_x, j ,up_down, tmp_index);
        x[tmp_index] = y[t_x];
        x[t_x] = y[t_x];
    }
    // UP CASE
    else
    {
        tmp_index = t_x - d / 2;
        // printf("Thread n: %d, Iteration : %d => up_down = %d, index_of_writing : %d\n",t_x, j ,up_down, tmp_index);
        p = (int)(tmp_index % d);
        if (direct)
        {
            direct_compute_wd_pow_h(&result_real, &result_imag, d, make_cuDoubleComplex(static_cast<double>(p), 0.0));
        }
        else
        {
            inverse_compute_wd_pow_h(&result_real, &result_imag, d, make_cuDoubleComplex(static_cast<double>(p), 0.0));
        }
        w = make_cuDoubleComplex(result_real, result_imag);
        t[tmp_index] = cuCmul(y[t_x], w);
        t[t_x] = cuCmul(y[t_x], w);
    }

    __syncthreads();
}

__device__ void thread_sum(int t_x, int d, cuDoubleComplex *y, cuDoubleComplex *x, cuDoubleComplex *t)
{
    if (t_x % d < d / 2)
    {
        y[t_x] = cuCadd(x[t_x], t[t_x]);
    }
    else
    {
        y[t_x] = cuCsub(x[t_x], t[t_x]);
    }
    __syncthreads();
}

__device__ void permutation(int t_x, int threadIdx_x, int log_n, int input_size, cuDoubleComplex *shared_y, cuDoubleComplex *a)
{
    if (t_x >= input_size)
        return;
    int j = 0;
    for (int k = 0; k < log_n; k++)
    {
        if (t_x & (1 << k))
        {
            j |= (1 << (log_n - 1 - k));
        }
    }
    shared_y[threadIdx_x] = a[j];
}

__global__ void parallel_fft_first_computation(bool direct, int *first_computation_j, int gpu_grid_size, int input_size, int input_grid_size, int *atomic_array, cuDoubleComplex *a, cuDoubleComplex *y, cuDoubleComplex *x, cuDoubleComplex *t, int log_n)
{
    unsigned int t_x = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ cuDoubleComplex shared_y[THREAD_PER_BLOCK];
    __shared__ cuDoubleComplex shared_x[THREAD_PER_BLOCK];
    __shared__ cuDoubleComplex shared_t[THREAD_PER_BLOCK];
    int d;
    int tmp_j;
    bool flag = false;
    permutation(t_x, threadIdx.x, log_n, input_size, shared_y, a);
    __syncthreads();

    for (int j = 1; j <= log_n && !flag; j++)
    {
        d = 1 << j;
        if (d < THREAD_PER_BLOCK)
        {
            thread_write(direct, threadIdx.x, d, shared_y, shared_x, shared_t);
            thread_sum(threadIdx.x, d, shared_y, shared_x, shared_t);
        }
        else if (d == THREAD_PER_BLOCK)
        {
            thread_write(direct, threadIdx.x, d, shared_y, shared_x, shared_t);
            thread_sum(threadIdx.x, d, shared_y, shared_x, shared_t);
            tmp_j = j;
            flag = true;
        }
        __syncthreads();
    }

    if (t_x == 0)
    {
        first_computation_j[0] = tmp_j;
    }

    y[t_x] = shared_y[threadIdx.x];
    __syncthreads();
    return;
}

__global__ void parallel_fft_second_computation(bool direct, int actual_computation_j, int gpu_grid_size, int input_size, int input_grid_size, int *atomic_array, cuDoubleComplex *y, cuDoubleComplex *x, cuDoubleComplex *t, int log_n)
{

    int d, prec_d;
    unsigned int t_x = blockIdx.x * blockDim.x + threadIdx.x;
    int element_thread_computation;

    int num_iteration;

    int j = actual_computation_j;
    prec_d = 1 << (j - 1);
    d = 1 << j;
    num_iteration = d / THREAD_PER_BLOCK;

    if ((blockIdx.x % num_iteration == 0))
    {

        for (int i = 0; i < num_iteration; i++)
        {
            element_thread_computation = t_x + blockDim.x * i;
            thread_write(direct, element_thread_computation, d, y, x, t);
        }

        for (int i = 0; i < num_iteration; i++)
        {
            element_thread_computation = t_x + blockDim.x * i;
            thread_sum(element_thread_computation, d, y, x, t);
        }
        if (threadIdx.x == 0)
        {
            atomicAdd(&atomic_array[j], 1);
        }
        __syncthreads();
    }
    else
    {

        return;
    }
}

void gpu_fft(int grid_size, int log_n, cuDoubleComplex *a, cuDoubleComplex *y, cuDoubleComplex *x, cuDoubleComplex *t, int *atomic_array, int *first_computation_j)
{
    dim3 dimGrid(grid_size);
    dim3 dimBlock(THREAD_PER_BLOCK);
    bool direct = true;
    parallel_fft_first_computation<<<dimGrid, dimBlock>>>(direct, first_computation_j, grid_size, MainClass::N, grid_size, atomic_array, a, y, x, t, log_n);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Error after first computation: " << cudaGetErrorString(err) << std::endl;
    }
    for (int j = *first_computation_j + 1; j <= log_n; j++)
    {
        parallel_fft_second_computation<<<dimGrid, dimBlock>>>(direct, j, grid_size, MainClass::N, grid_size, atomic_array, y, x, t, log_n);
        cudaDeviceSynchronize();
    }
}

void gpu_inverse_fft(int grid_size, int log_n, cuDoubleComplex *a, cuDoubleComplex *y, cuDoubleComplex *x, cuDoubleComplex *t, int *atomic_array, int *first_computation_j)
{
    dim3 dimGrid(grid_size);
    dim3 dimBlock(THREAD_PER_BLOCK);
    bool direct = false;
    parallel_fft_first_computation<<<dimGrid, dimBlock>>>(direct, first_computation_j, grid_size, MainClass::N, grid_size, atomic_array, a, y, x, t, log_n);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Error after first computation: " << cudaGetErrorString(err) << std::endl;
    }
    for (int j = *first_computation_j + 1; j <= log_n; j++)
    {
        parallel_fft_second_computation<<<dimGrid, dimBlock>>>(direct, j, grid_size, MainClass::N, grid_size, atomic_array, y, x, t, log_n);
        cudaDeviceSynchronize();
    }
}

/**
 * @brief Performs a direct Fast Fourier Transform (FFT) on the input data using CUDA.
 *
 * This function allocates memory on the GPU, copies the input data to the GPU, and performs
 * the FFT using CUDA kernels. The execution time is measured and printed to the console.
 *
 * @param input A vector of complex numbers representing the input data to be transformed.
 * @return A pair containing a vector of complex numbers representing the transformed data and the duration of the FFT execution.
 */
std::pair<std::vector<std::complex<double>>, std::chrono::duration<double>> kernel_direct_fft(const std::vector<std::complex<double>> input)
{
    // CUDA malloc initialization
    int *prova;
    cudaMallocManaged((void **)&prova, sizeof(int));

    int grid_size = input.size() / THREAD_PER_BLOCK;
    if (input.size() % THREAD_PER_BLOCK != 0)
        grid_size++;
    cout << "\nGPU_PARALLEL   DIRECT_FFT\n";
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
    gpu_fft(grid_size, log_n, a, y, x, t, atomic_array, first_computation_j);
    auto end_parallel = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_parallel = end_parallel - start_parallel;
    std::chrono::duration<double> duration_parallel_without_malloc = end_parallel - malloc_complete;
    std::cout << "Parallel FFT execution time: " << duration_parallel.count() << " seconds" << std::endl;
    std::cout << "Parallel FFT execution time WITHOUT MALLOC: " << duration_parallel_without_malloc.count() << " seconds" << std::endl;

    return {cuDoubleComplexToVector(y, input.size()), duration_parallel};
}

/**
 * @brief Performs an inverse Fast Fourier Transform (IFFT) on the input data using CUDA.
 *
 * This function allocates memory on the GPU, copies the input data to the GPU, and performs
 * the IFFT using CUDA kernels. The execution time is measured and printed to the console.
 * The result is normalized by dividing each element by the size of the input.
 *
 * @param input A vector of complex numbers representing the input data to be transformed.
 * @return A vector of complex numbers representing the transformed data.
 */
std::vector<std::complex<double>> kernel_inverse_fft(const std::vector<std::complex<double>> input)
{
    // CUDA malloc initialization
    int *prova;
    cudaMallocManaged((void **)&prova, sizeof(int));

    int grid_size = input.size() / THREAD_PER_BLOCK;
    if (input.size() % THREAD_PER_BLOCK != 0)
        grid_size++;
    cout << "\nGPU_PARALLEL INVERSE_FFT\n";
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
    std::chrono::duration<double> duration_parallel = end_parallel - start_parallel;
    std::chrono::duration<double> duration_parallel_without_malloc = end_parallel - malloc_complete;
    std::cout << "Parallel IFFT execution time: " << duration_parallel.count() << " seconds" << std::endl;
    std::cout << "Parallel IFFT execution time WITHOUT MALLOC: " << duration_parallel_without_malloc.count() << " seconds" << std::endl;

    auto result = cuDoubleComplexToVector(y, input.size());
    // Normalize the IFFT result
    for (auto &val : result)
    {
        val /= input.size(); // Divide by N
    }

    return result;
}
