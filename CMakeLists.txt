cmake_minimum_required(VERSION 3.10)
cmake_policy(SET CMP0074 NEW)  # policy needed for openMP homebrew paths

project(FFT_HPC)

# === C++ standard ===
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# === MAC (Homebrew OpenMP fix) ===
if(APPLE)
    execute_process(
            COMMAND brew --prefix libomp
            OUTPUT_VARIABLE HOMEBREW_LIBOMP_PREFIX
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
    )

    if(HOMEBREW_LIBOMP_PREFIX)
        set(OpenMP_ROOT ${HOMEBREW_LIBOMP_PREFIX})

        # Explicitly set the library paths that CMake is looking for
        set(OpenMP_omp_LIBRARY ${HOMEBREW_LIBOMP_PREFIX}/lib/libomp.dylib)
        set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I${HOMEBREW_LIBOMP_PREFIX}/include")
        set(OpenMP_C_LIB_NAMES "omp")
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${HOMEBREW_LIBOMP_PREFIX}/include")
        set(OpenMP_CXX_LIB_NAMES "omp")

        # Set the libraries
        set(OpenMP_C_LIBRARIES ${HOMEBREW_LIBOMP_PREFIX}/lib/libomp.dylib)
        set(OpenMP_CXX_LIBRARIES ${HOMEBREW_LIBOMP_PREFIX}/lib/libomp.dylib)
        set(OpenCV_DIR /opt/homebrew/Cellar/opencv/4.11.0_1/lib/cmake/opencv4)
    endif()
endif()

# === Find dependencies ===
find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)
find_package(OpenCV QUIET)

# === GoogleTest via FetchContent ===
include(FetchContent)
FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG main
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

# === Include directories ===
include_directories(
        image_compression/include
        include
        ${MPI_INCLUDE_PATH}
)

# === Original executables (from teammates) ===
add_executable(main_cpu src/cpu_main.cpp)
add_executable(cpu_test test/cpu_test.cpp)

target_link_libraries(cpu_test gtest_main MPI::MPI_CXX)
include(GoogleTest)
gtest_discover_tests(cpu_test)

# === GPU block ===
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    enable_language(CUDA)

    add_executable(main_gpu src/gpu_main.cu)
    set_target_properties(main_gpu PROPERTIES CUDA_ARCHITECTURES 70)
    target_compile_options(main_gpu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--diag-suppress=550>)
    target_link_libraries(main_gpu PRIVATE cufft)

    add_executable(gpu_test test/gpu_test.cu)
    target_link_libraries(gpu_test gtest_main cufft)
    set_target_properties(gpu_test PROPERTIES CUDA_ARCHITECTURES 70)
    target_compile_options(gpu_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--diag-suppress=550>)
    include(GoogleTest)
    gtest_discover_tests(gpu_test)
endif()

if(OpenCV_FOUND)
    include_directories(${OpenCV_INCLUDE_DIRS})

    # === Image Compression Executable ===
    add_executable(image_compression_exec
            image_compression/src/svd_analysis.cpp
            image_compression/src/error_plot.cpp
            image_compression/src/fft_analysis_magnitude.cpp
            image_compression/src/fft_analysis_band.cpp
            image_compression/src/image_saver.cpp
            image_compression/src/fft_utils.cpp
            src/image_compression_main.cpp
    )

    target_link_libraries(image_compression_exec
            ${MPI_LIBRARIES}
            OpenMP::OpenMP_CXX
            Threads::Threads
            ${OpenCV_LIBS}
            curl
            sqlite3
    )
endif ()

# === MPI executables ===
add_executable(mpi_main src/mpi_main.cpp)
add_executable(mpi_main_inverse src/mpi_main_inverse.cpp)
target_link_libraries(mpi_main ${MPI_LIBRARIES} OpenMP::OpenMP_CXX Threads::Threads)
target_link_libraries(mpi_main_inverse ${MPI_LIBRARIES} OpenMP::OpenMP_CXX Threads::Threads)

