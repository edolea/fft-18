#!/bin/bash

# Navigate to the script directory
SCRIPT_DIR="$(dirname "$(realpath "$0")")"
PYTHON_SCRIPT="$SCRIPT_DIR/Python/mpi_v5.py"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python3 is not installed. Please install it first."
    exit 1
fi

# Run the Python script
python3 "$PYTHON_SCRIPT"