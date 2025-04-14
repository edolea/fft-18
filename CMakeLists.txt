cmake_minimum_required(VERSION 3.10)

# Project name
project(FFTProject)

# C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add the executable
add_executable(FFTProject
        FFT/src/main.cpp
        FFT/src/ParallelIterativeFFT.cpp
        FFT/src/Cooley-Tukey.cpp
        # FFT/src/Cooley-Tukey-parallel.cpp
)

# Include directories
include_directories(FFT/include)
include_directories(${MPI_INCLUDE_PATH})

# Link libraries
find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)

target_link_libraries(FFTProject ${MPI_LIBRARIES} OpenMP::OpenMP_CXX)
target_link_libraries(FFTProject Threads::Threads)

# Set compiler flags
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fopenmp")