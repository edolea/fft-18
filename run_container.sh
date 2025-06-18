#!/bin/bash

IMAGE_NAME=fft-hpc
CONTAINER_NAME=fft-hpc-container
WORK_DIR=/workspace
PYTHON_SCRIPT=full_analysis.py
OUTPUT_DIR=OUTPUT_RESULT/Plot_result

# Build image if not found
if ! docker images | awk '{print $1}' | grep -q "^$IMAGE_NAME$"; then
    echo "Docker image '$IMAGE_NAME' not found. Building it now..."
    docker build -t $IMAGE_NAME .
fi

# Run the container in background with mounted volume
docker run -dit --name $CONTAINER_NAME -v "$(pwd)":$WORK_DIR $IMAGE_NAME

echo "Docker container '$CONTAINER_NAME' started."

# Get available images
echo "Available images:"
IMAGE_LIST=($(ls image_compression/images))

if [ ${#IMAGE_LIST[@]} -eq 0 ]; then
    echo "No images found in the images directory!"
    docker stop $CONTAINER_NAME > /dev/null
    docker rm $CONTAINER_NAME > /dev/null
    exit 1
fi

for i in "${!IMAGE_LIST[@]}"; do
    echo "$((i+1)). ${IMAGE_LIST[$i]}"
done

echo
while true; do
    echo "Select the number of the image you want to process:"
    read selection
    if [[ "$selection" =~ ^[0-9]+$ ]] && [ "$selection" -ge 1 ] && [ "$selection" -le ${#IMAGE_LIST[@]} ]; then
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
    if [[ "$core_count" =~ ^[0-9]+$ ]] && [ "$core_count" -ge 1 ]; then
        break
    else
        echo "Invalid number, please enter a positive integer."
    fi
done

echo "You selected: $IMAGE_FILE"
echo "Using $core_count MPI processes"

# Run the analysis inside the container
docker exec -it $CONTAINER_NAME bash -c "
    set -e
    cd $WORK_DIR

    if [ ! -d '.venv' ]; then
        echo 'Creating virtual environment...'
        python3 -m venv .venv
        source .venv/bin/activate
        pip install numpy matplotlib --break-system-packages
    else
        echo 'Activating existing virtual environment...'
        source .venv/bin/activate
    fi

    echo 'Rebuilding project...'
    rm -rf build
    mkdir build
    cd build
    cmake ..
    make -j

    echo 'Running MPI image compression...'
    mpirun --allow-run-as-root -np $core_count ./image_compression_exec ../image_compression/images/$IMAGE_FILE

    echo 'Running Python plot script...'
    cd ../Python
    mkdir -p ../$OUTPUT_DIR
    chmod -R 777 ../$OUTPUT_DIR
    python3 $PYTHON_SCRIPT
"

echo "All tasks completed. Results saved to $OUTPUT_DIR."

# Optionally stop and remove container
docker stop $CONTAINER_NAME > /dev/null
docker rm $CONTAINER_NAME > /dev/null