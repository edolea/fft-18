cmake_minimum_required(VERSION 3.10)

# Project name
project(fft)

# C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

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
include_directories(
        include
)

# add executable
add_executable(fft
        src/main.cpp
        src/fourier_transform.cpp
)

# testing
add_executable(my_tests
        test/fourier_testing.cpp
)
target_link_libraries(my_tests gtest_main)
include(GoogleTest)
gtest_discover_tests(my_tests)

# include_directories(${MPI_INCLUDE_PATH})

# Link libraries
# find_package(Threads REQUIRED)
# find_package(OpenMP REQUIRED)
# find_package(MPI REQUIRED)

# target_link_libraries(FFTProject ${MPI_LIBRARIES} OpenMP::OpenMP_CXX)
# target_link_libraries(FFTProject Threads::Threads)

# Set compiler flags
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fopenmp")