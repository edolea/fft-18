# FFT High Performance Computing Project

This project provides a high-performance implementation of the Fast Fourier Transform (FFT) with support for CPU, GPU (CUDA), and MPI-based parallelism. It also includes an image-processing pipeline that applies FFT to images.

---

## Build Instructions

To build the project, use the provided `build.sh` script. This script will:

- Remove any previous build directory.
- Create a new `build` directory.
- Run CMake to configure the project.
- Compile all sources.

**Usage:**

```sh
./build.sh
```

---

## Run Instructions

After building, use the `run.sh` script to execute the available FFT options. The script provides an interactive menu:

1. **Run a simple function for timing tests**

   - Choose between CPU, GPU, or MPI hybrid execution.
   - For MPI, you can benchmark both 1D and 2D FFTs with various process/thread configurations. Results are saved in the `OUTPUT_RESULT/mpi` directory.

2. **Apply FFT to an image**
   - You will be prompted to enter the path to an image file.
   - The FFT will be applied to the image using the image processing executable.

**Usage:**

```sh
./run.sh
```

---

## Directory Structure

- `build.sh` — Script to configure and build the project.
- `run.sh` — Interactive script to run FFT benchmarks or image FFT.
- `src/` — Source code for CPU, GPU, and MPI implementations.
- `test/` — Unit and integration tests.
- `OUTPUT_RESULT/` — Output directory for benchmark results.
- `image_compression/` — Image FFT pipeline and related code.

---

## Requirements

- CMake 3.10+
- C++20 compiler
- MPI (e.g., OpenMPI)
- CUDA Toolkit (for GPU support)
- OpenMP
- OpenCV (for image processing)
- SQLite3 development libraries (for image compression exec)

Install dependencies on Ubuntu:

```sh
sudo apt-get install build-essential cmake libopenmpi-dev libopencv-dev libsqlite3-dev
```

For CUDA support, install the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads).

---

## Notes

- All executables are built in the `build/` directory.
- For MPI runs, ensure you have enough resources for the chosen process/thread configuration.
- For image FFT, provide a valid image path (e.g., `./image_compression/images/image_6.png`).

---

## Example Usage

**Build the project:**

```sh
./build.sh
```

**Run benchmarks or image FFT:**

```sh
./run.sh
```

---

For more details, see the source code and scripts in this repository.
