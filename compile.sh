#!/bin/bash

rm -rf build/

mkdir build

# Navigate to the build directory
cd build

# Run CMake to generate build files
cmake ..

# Compile the project
make

# Run the resulting executable
./FFTProject