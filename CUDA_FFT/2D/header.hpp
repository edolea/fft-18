#include "./static_header.hpp"

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

// Note: THREAD_PER_BLOCK must be defined somewhere (e.g. #define THREAD_PER_BLOCK 256)

// This function performs a batch of 1D FFTs on a set of vectors stored in contiguous memory.
// "batch" is the number of FFTs to perform and each FFT is of length "fft_length".
// "log_fft" is log2(fft_length).
void compute_batch_fft(bool direct, int grid_size, int batch, int fft_length, int log_fft,
                       cuDoubleComplex *a, cuDoubleComplex *y, cuDoubleComplex *x, cuDoubleComplex *t,
                       int *atomic_array, int *first_computation_j)
{
    dim3 dimGrid(grid_size);
    dim3 dimBlock(THREAD_PER_BLOCK);

    // Allocate one CUDA stream per FFT in the batch.
    cudaStream_t *streams = (cudaStream_t *)malloc(batch * sizeof(cudaStream_t));
    for (int i = 0; i < batch; i++)
    {
        cudaStreamCreate(&streams[i]);
    }

    // Launch the first computation kernel for each FFT vector.
    for (int i = 0; i < batch; i++)
    {
        // a, y, x, and t are stored consecutively.
        parallel_fft_first_computation<<<dimGrid, dimBlock, 0, streams[i]>>>(
            direct, first_computation_j + i, grid_size, fft_length, grid_size,
            atomic_array, a + i * fft_length, y + i * fft_length, x + i * fft_length, t + i * fft_length, log_fft);
    }
    for (int i = 0; i < batch; i++)
    {
        cudaStreamSynchronize(streams[i]);
    }

    // Launch the second computation kernels for each stage of the FFT.
    for (int j = first_computation_j[0] + 1; j <= log_fft; j++)
    {
        for (int i = 0; i < batch; i++)
        {
            parallel_fft_second_computation<<<dimGrid, dimBlock, 0, streams[i]>>>(
                direct, j, grid_size, fft_length, grid_size,
                atomic_array, y + i * fft_length, x + i * fft_length, t + i * fft_length, log_fft);
        }
        for (int i = 0; i < batch; i++)
        {
            cudaStreamSynchronize(streams[i]);
        }
    }
    free(streams);
}

// **********************************************************************
// Direct 2D FFT using the GPU.
// The input is a 2D vector of std::complex<double> with dimensions [rows x cols].
// The row-wise FFT is performed on each row (each row is of length = cols),
// then the result is transposed and a column-wise FFT is applied (each FFT is length = rows).
std::pair<std::vector<std::vector<std::complex<double>>>, std::chrono::duration<double>> direct_fft_2d(const std::vector<std::vector<std::complex<double>>> &input)
{
    int rows = input.size();
    int cols = input[0].size();

    std::cout << "\nGPU_PARALLEL DIRECT FFT\n";
    auto start_parallel = std::chrono::high_resolution_clock::now();

    int log_cols = (int)(log(cols) / log(2));
    int log_rows = (int)(log(rows) / log(2));

    // Compute grid sizes based on the length of each 1D FFT.
    int grid_size_rows = (cols + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK; // For row-wise FFT (length = cols)
    int grid_size_cols = (rows + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK; // For column-wise FFT (length = rows)

    // Allocate device memory (using rows*cols elements; note: some padding might be needed in a production code).
    cuDoubleComplex *d_a, *d_y, *d_x, *d_t;
    int *d_atomic_array;
    int *d_first_computation_j;
    cudaMallocManaged((void **)&d_a, sizeof(cuDoubleComplex) * rows * cols);
    cudaMallocManaged((void **)&d_y, sizeof(cuDoubleComplex) * rows * cols);
    cudaMallocManaged((void **)&d_x, sizeof(cuDoubleComplex) * rows * cols);
    cudaMallocManaged((void **)&d_t, sizeof(cuDoubleComplex) * rows * cols);
    cudaMallocManaged((void **)&d_atomic_array, sizeof(int) * (log_cols + 1));
    cudaMallocManaged((void **)&d_first_computation_j, sizeof(int) * rows);

    // Copy input matrix to d_a (stored row-major).
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            d_a[i * cols + j] = make_cuDoubleComplex(input[i][j].real(), input[i][j].imag());
        }
    }

    auto malloc_complete = std::chrono::high_resolution_clock::now();
    bool direct_flag = true;
    // --- Step 1: Row-wise FFT ---
    // There are 'rows' FFTs to perform; each FFT is of length 'cols'.
    compute_batch_fft(direct_flag, grid_size_rows, rows, cols, log_cols,
                      d_a, d_y, d_x, d_t, d_atomic_array, d_first_computation_j);

    // --- Step 2: Transpose the row-wise FFT output ---
    // d_y is currently in a [rows x cols] layout.
    // Transpose into d_a so that the result is in a [cols x rows] layout.
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            d_a[j * rows + i] = d_y[i * cols + j];
        }
    }

    // --- Step 3: Column-wise FFT ---
    // Now the “rows” of the transposed matrix are the original columns.
    // There are 'cols' FFTs, each of length 'rows'.
    compute_batch_fft(direct_flag, grid_size_cols, cols, rows, log_rows,
                      d_a, d_y, d_x, d_t, d_atomic_array, d_first_computation_j);

    // --- Step 4: Transpose the result back ---
    // d_y now holds a [cols x rows] matrix; we transpose it into d_a as [rows x cols].
    for (int i = 0; i < cols; i++)
    {
        for (int j = 0; j < rows; j++)
        {
            d_a[j * cols + i] = d_y[i * rows + j];
        }
    }

    auto end_parallel = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_parallel = end_parallel - start_parallel;
    std::chrono::duration<double> duration_parallel_without_malloc = end_parallel - malloc_complete;
    std::cout << "Parallel 2D FFT execution time: " << duration_parallel.count() << " seconds" << std::endl;
    std::cout << "Parallel 2D FFT execution time WITHOUT MALLOC: " << duration_parallel_without_malloc.count() << " seconds" << std::endl;

    // Build output vector from d_a.
    std::vector<std::vector<std::complex<double>>> output(rows, std::vector<std::complex<double>>(cols));
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            output[i][j] = std::complex<double>(cuCreal(d_a[i * cols + j]), cuCimag(d_a[i * cols + j]));
        }
    }

    // Free device memory.
    cudaFree(d_a);
    cudaFree(d_y);
    cudaFree(d_x);
    cudaFree(d_t);
    cudaFree(d_atomic_array);
    cudaFree(d_first_computation_j);

    return {output, duration_parallel};
}

// **********************************************************************
// Inverse 2D FFT using the GPU.
// This is similar to direct_fft_2d but uses the inverse transform kernels and
// scales the final result by 1/(rows*cols).
std::vector<std::vector<std::complex<double>>> inverse_fft_2d(const std::vector<std::vector<std::complex<double>>> &input)
{
    int rows = input.size();
    int cols = input[0].size();

    std::cout << "\nGPU_PARALLEL INVERSE FFT\n";
    auto start_parallel = std::chrono::high_resolution_clock::now();

    int log_cols = (int)(log(cols) / log(2));
    int log_rows = (int)(log(rows) / log(2));

    int grid_size_rows = (cols + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;
    int grid_size_cols = (rows + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;

    cuDoubleComplex *d_a, *d_y, *d_x, *d_t;
    int *d_atomic_array;
    int *d_first_computation_j;
    cudaMallocManaged((void **)&d_a, sizeof(cuDoubleComplex) * rows * cols);
    cudaMallocManaged((void **)&d_y, sizeof(cuDoubleComplex) * rows * cols);
    cudaMallocManaged((void **)&d_x, sizeof(cuDoubleComplex) * rows * cols);
    cudaMallocManaged((void **)&d_t, sizeof(cuDoubleComplex) * rows * cols);
    cudaMallocManaged((void **)&d_atomic_array, sizeof(int) * (log_cols + 1));
    cudaMallocManaged((void **)&d_first_computation_j, sizeof(int) * rows);

    // Copy input matrix into device memory.
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            d_a[i * cols + j] = make_cuDoubleComplex(input[i][j].real(), input[i][j].imag());
        }
    }

    auto malloc_complete = std::chrono::high_resolution_clock::now();
    bool direct_flag = false; // Inverse FFT mode
    // --- Step 1: Perform inverse FFT on rows ---
    compute_batch_fft(direct_flag, grid_size_rows, rows, cols, log_cols,
                      d_a, d_y, d_x, d_t, d_atomic_array, d_first_computation_j);

    // --- Step 2: Transpose the result ---
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            d_a[j * rows + i] = d_y[i * cols + j];
        }
    }

    // --- Step 3: Perform inverse FFT on columns ---
    compute_batch_fft(direct_flag, grid_size_cols, cols, rows, log_rows,
                      d_a, d_y, d_x, d_t, d_atomic_array, d_first_computation_j);

    // --- Step 4: Transpose back ---
    for (int i = 0; i < cols; i++)
    {
        for (int j = 0; j < rows; j++)
        {
            d_a[j * cols + i] = d_y[i * rows + j];
        }
    }

    auto end_parallel = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_parallel = end_parallel - start_parallel;
    std::chrono::duration<double> duration_parallel_without_malloc = end_parallel - malloc_complete;
    std::cout << "Parallel 2D IFFT execution time: " << duration_parallel.count() << " seconds" << std::endl;
    std::cout << "Parallel 2D IFFT execution time WITHOUT MALLOC: " << duration_parallel_without_malloc.count() << " seconds" << std::endl;

    // Build output and apply scaling (1/(rows*cols)) for the inverse FFT.
    std::vector<std::vector<std::complex<double>>> output(rows, std::vector<std::complex<double>>(cols));
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            output[i][j] = std::complex<double>(
                cuCreal(d_a[i * cols + j]) / (rows * cols),
                cuCimag(d_a[i * cols + j]) / (rows * cols));
        }
    }

    cudaFree(d_a);
    cudaFree(d_y);
    cudaFree(d_x);
    cudaFree(d_t);
    cudaFree(d_atomic_array);
    cudaFree(d_first_computation_j);

    return output;
}
