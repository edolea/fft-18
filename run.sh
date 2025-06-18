#!/bin/bash

rm -rf build/
mkdir build

# Navigate to the build directory
cd build

# Run CMake to generate build files
cmake ..

# Compile the project
make

# Run the resulting executables
#./main_cpu
#./cpu_test
./main_gpu
# Only run gpu_test if it exists
if [ -f ./gpu_test ]; then
#    ./gpu_test


