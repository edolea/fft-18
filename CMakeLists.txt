cmake_minimum_required(VERSION 3.10)
cmake_policy(SET CMP0074 NEW)  # policy needed for openMP homebrew paths

# Project name
project(FFT_HPC)

# C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Link libraries
find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)

# Add Google Test using FetchContent
include(FetchContent)
FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        main
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

# add_subdirectory(external/googletest)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(googletest)

enable_testing()

# Include directories
include_directories(include)

# add executable
add_executable(main_cpu
        src/cpu_main.cpp
)

# testing
add_executable(cpu_test
        test/cpu_test.cpp
)

# Add this after your cpu_test executable declaration
#add_executable(cpu_mpi_test
        #test/cpu_mpi_test.cpp
        #test/mpi01.cpp )


# Try to find CUDA (modern CMake)
find_package(CUDAToolkit QUIET)

if(CUDAToolkit_FOUND)
    enable_language(CUDA)

    # GPU executable
    add_executable(gpu_exec
        src/gpu_main.cu
    )
    set_target_properties(gpu_exec PROPERTIES CUDA_ARCHITECTURES 70)
    target_compile_options(gpu_exec PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--diag-suppress=550>)
    target_link_libraries(gpu_exec PRIVATE cufft)

    add_executable(gpu_test
            test/gpu_test.cu
    )
    target_link_libraries(gpu_test gtest_main cufft)
    set_target_properties(gpu_test PROPERTIES CUDA_ARCHITECTURES 70)
    target_compile_options(gpu_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--diag-suppress=550>)
    include(GoogleTest)
    gtest_discover_tests(gpu_test)

endif()


# CPU (sequential) GoogleTest
target_link_libraries(cpu_test gtest_main MPI::MPI_CXX)
include(GoogleTest)
gtest_discover_tests(cpu_test)




# include_directories(${MPI_INCLUDE_PATH})
include_directories(${MPI_INCLUDE_PATH})

# Set OpenMP paths for Homebrew on macOS
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
    endif()
endif()

# Link libraries
find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)

target_link_libraries(main_cpu ${MPI_LIBRARIES} OpenMP::OpenMP_CXX)
target_link_libraries(main_cpu Threads::Threads)

# Link with MPI and other libraries
#[[
target_link_libraries(cpu_mpi_test gtest ${MPI_LIBRARIES} OpenMP::OpenMP_CXX)
target_include_directories(cpu_mpi_test PRIVATE ${MPI_INCLUDE_PATH})

# Add custom test target to run with MPI
add_test(
        NAME MPITest
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4
        ${MPIEXEC_PREFLAGS} $<TARGET_FILE:cpu_mpi_test> ${MPIEXEC_POSTFLAGS}
)
]]

# Set compiler flags
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fopenmp")