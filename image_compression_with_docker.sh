#!/bin/bash

CONTAINER_NAME=hpc-projects
WORK_DIR=/home/ubuntu/shared-folder/fft
PYTHON_SCRIPT=full_analysis.py
OUTPUT_DIR=OUTPUT_RESULT/Plot_result

echo "Starting Docker container ($CONTAINER_NAME)..."

docker start $CONTAINER_NAME > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "Error: the container $CONTAINER_NAME does not exist. Please create it first!"
    exit 1
fi
echo "Container started."

echo "Available images:"
IMAGE_LIST=($(ls image_compression/images))

if [ ${#IMAGE_LIST[@]} -eq 0 ]; then
    echo "No images found in the images directory!"
    exit 1
fi

for i in "${!IMAGE_LIST[@]}"; do
    echo "$((i+1)). ${IMAGE_LIST[$i]}"
done

echo
while true; do
    echo "Select the number of the image you want to process:"
    read selection

    if [[ $selection =~ ^[0-9]+$ ]] && [ "$selection" -ge 1 ] && [ "$selection" -le ${#IMAGE_LIST[@]} ]; then
        IMAGE_FILE=${IMAGE_LIST[$((selection-1))]}
        break
    else
        echo "Invalid selection, please try again."
    fi
done

echo
while true; do
    echo "Enter the number of cores (processes) to use with MPI:"
    read core_count

    if [[ $core_count =~ ^[0-9]+$ ]] && [ "$core_count" -ge 1 ]; then
        break
    else
        echo "Invalid number, please enter a positive integer."
    fi
done

echo "You selected: $IMAGE_FILE"
echo "Using $core_count MPI processes"

docker exec -it $CONTAINER_NAME bash -c "
    set -e
    echo 'Navigating to project directory...'
    cd $WORK_DIR

    # Setup virtual environment if needed
    if [ ! -d '.venv' ]; then
        echo 'Creating virtual environment...'
        python3 -m venv .venv
        source .venv/bin/activate
        pip install numpy matplotlib pandas --break-system-packages
    else
        echo 'Activating existing virtual environment...'
        source .venv/bin/activate
    fi

    echo 'Rebuilding the project...'
    rm -rf build
    mkdir build
    cd build
    cmake ..
    make -j

    echo 'Running image_compression_exec with MPI...'
    mpirun --allow-run-as-root -np $core_count ./image_compression_exec ../image_compression/images/$IMAGE_FILE

    echo 'Running Python plotting script...'
    cd ../Python
    mkdir -p ../$OUTPUT_DIR
    chmod -R 777 ../$OUTPUT_DIR
    python3 $PYTHON_SCRIPT
"

echo "All tasks completed. Results saved to $OUTPUT_DIR."