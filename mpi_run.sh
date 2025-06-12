#!/bin/bash

rm -rf build_mpi/

mkdir build_mpi

# Navigate to the build directory
cd build_mpi || exit 1

# Only set compilers on Apple (macOS) -- needed to run MPI
if [[ "$(uname)" == "Darwin" ]]; then
    export CC=/opt/homebrew/bin/gcc-15
    export CXX=/opt/homebrew/bin/g++-15
fi

# Run CMake to generate build files
cmake ..

# Compile the project
make

mpiexec -n 2 ./mpi_main 2 8