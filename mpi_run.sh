#!/bin/bash

# for CONFIG in "4 1" "4 6" "8 1" "8 6" "16 1" "16 6" "32 1" "32 6"; do


CONFIG in "4 24" "4 48" "8 24" "8 48" "16 1" "16 24" "16 48";

cd build || exit 1

mkdir -p ../output_result/mpi
mkdir -p ../output_result/mpi/1D
mkdir -p ../output_result/mpi/2D

for CONFIG in "4 4" "4 8" "8 4" "8 8" "16 4" "16 8"
do
  read PROCS THREADS <<< "$CONFIG"
  export OMP_NUM_THREADS=$THREADS
  OUTPUT_FILE="../output_result/mpi/1D/hybrid_p${PROCS}_t${THREADS}.txt"
  echo "===== HYBRID: $PROCS PROCESSES × $THREADS THREADS ====="
  echo "===== HYBRID: $PROCS PROCESSES × $THREADS THREADS =====" > "$OUTPUT_FILE"
  echo "N  sequential  mpi  speedup" >> "$OUTPUT_FILE"

for N in 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576 # 2097152 4194304 8388608 16777216 33554432 67108864 134217728 268435456 536870912
  do
    echo "--> Running $PROCS MPI processes with $THREADS threads each (N=$N)"
    # echo "--> Running $PROCS MPI processes with $THREADS threads each (N=$N)" >> "$OUTPUT_FILE"
    mpiexec -n $PROCS ./mpi_main 1 $N >> "$OUTPUT_FILE" 2>&1
  done
  echo "Output saved to $OUTPUT_FILE"
  echo "    "
done

echo "   "
echo "   "
echo "**************** 2D CASE ****************"
echo "   "
echo "   "

for CONFIG in "4 4" "4 8" "8 4" "8 8" "16 4" "16 8"
do
  read PROCS THREADS <<< "$CONFIG"
  export OMP_NUM_THREADS=$THREADS

  OUTPUT_FILE="../output_result/mpi/2D/hybrid_p${PROCS}_t${THREADS}.txt"
  echo "===== HYBRID: $PROCS PROCESSES × $THREADS THREADS ====="
  echo "===== HYBRID: $PROCS PROCESSES × $THREADS THREADS =====" > "$OUTPUT_FILE"
  echo "N  sequential  mpi  speedup" >> "$OUTPUT_FILE"

  for N in 512 1024 2048 4096
  do
    echo "--> Running $PROCS MPI processes with $THREADS threads each (N=$N)"
    # echo "--> Running $PROCS MPI processes with $THREADS threads each (N=$N)" >> "$OUTPUT_FILE"
    mpiexec -n $PROCS ./mpi_main 2 $N >> "$OUTPUT_FILE" 2>&1
  done
  echo "Output saved to $OUTPUT_FILE"
  echo "    "
done