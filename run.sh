#!/bin/bash

echo "=========================================="
echo "        FFT EXECUTION OPTIONS             "
echo "=========================================="
echo "1) Run a simple function for timing tests"
echo "2) Apply FFT to an image"
read -p "Enter your choice (1 or 2): " mode_choice

if [ "$mode_choice" == "1" ]; then
    cd build || { echo "Build directory not found!"; exit 1; }

    echo
    echo "Which execution type do you want to use?"
    echo "1) CPU Sequential"
    echo "2) GPU"
    echo "3) MPI Hybrid"
    read -p "Enter your choice (1/2/3): " exec_choice

    case $exec_choice in
        1)
            echo "Running CPU sequential..."
            ./main_cpu
            ;;
        2)
            echo "Running GPU..."
            ./main_gpu
            ;;
        3)
            echo
            echo "Would you like to run:"
            echo "1) 1D MPI benchmark"
            echo "2) 2D MPI benchmark"
            read -p "Enter your choice (1 or 2): " mpi_dim

            if [[ "$mpi_dim" != "1" && "$mpi_dim" != "2" ]]; then
                echo "Invalid MPI dimension choice. Exiting."
                exit 1
            fi

            mkdir -p ../output_result/mpi/1D
            mkdir -p ../output_result/mpi/2D

            echo
            echo "Starting MPI ${mpi_dim}D benchmark..."

            for CONFIG in "4 4" "4 8" "8 4" "8 8" "16 4" "16 8" "16 16" "1 16" "16 1"; do
                read PROCS THREADS <<< "$CONFIG"
                export OMP_NUM_THREADS=$THREADS

                if [ "$mpi_dim" == "1" ]; then
                    OUTPUT_FILE="../output_result/mpi/1D/hybrid_p${PROCS}_t${THREADS}.txt"
                    echo "===== 1D HYBRID: $PROCS PROCESSES × $THREADS THREADS ====="
                    echo "===== 1D HYBRID: $PROCS PROCESSES × $THREADS THREADS =====" > "$OUTPUT_FILE"
                    echo "N  sequential  mpi  speedup" >> "$OUTPUT_FILE"

                    for N in 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576; do
                        echo "--> Running $PROCS MPI processes with $THREADS threads (N=$N)"
                        mpiexec -n $PROCS ./mpi_main 1 $N >> "$OUTPUT_FILE" 2>&1
                    done
                else
                    OUTPUT_FILE="../output_result/mpi/2D/hybrid_p${PROCS}_t${THREADS}.txt"
                    echo "===== 2D HYBRID: $PROCS PROCESSES × $THREADS THREADS ====="
                    echo "===== 2D HYBRID: $PROCS PROCESSES × $THREADS THREADS =====" > "$OUTPUT_FILE"
                    echo "N  sequential  mpi  speedup" >> "$OUTPUT_FILE"

                    for N in 128 256 512 1024 2048 4096; do
                        echo "--> Running $PROCS MPI processes with $THREADS threads (N=$N)"
                        mpiexec -n $PROCS ./mpi_main 2 $N >> "$OUTPUT_FILE" 2>&1
                    done
                fi

                echo "Output saved to $OUTPUT_FILE"
                echo
            done
            ;;
        *)
            echo "Invalid option. Exiting."
            exit 1
            ;;
    esac

elif [ "$mode_choice" == "2" ]; then
    echo "Running FFT on an image..."
    read -p "Enter the path to the image (e.g., image_compression/images/image_6.png): " image_path

    if [ ! -f "$image_path" ]; then
        echo "Error: File '$image_path' does not exist."
        exit 1
    fi

    cd build || { echo "Build directory not found!"; exit 1; }
    ./image_compression_exec "../$image_path"

    echo
    echo "===== Running Python analysis (full_analysis.py) ====="

        PYTHON_DIR="../Python"
        VENV_PATH="../.venv"
        PYTHON_BIN="$VENV_PATH/bin/python"
        PIP_BIN="$VENV_PATH/bin/pip"

        if [ ! -d "$VENV_PATH" ]; then
            echo "Creating Python virtual environment..."
            python3 -m venv "$VENV_PATH"
        fi

        echo "Activating virtual environment and installing dependencies..."
        source "$VENV_PATH/bin/activate"

        $PIP_BIN install --break-system-packages --quiet --upgrade pip
        $PIP_BIN install --break-system-packages --quiet matplotlib numpy pandas

        echo "Running full_analysis.py"
        $PYTHON_BIN "$PYTHON_DIR/full_analysis.py"

        deactivate
        echo "Python analysis completed."
        fi