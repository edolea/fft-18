#!/bin/bash

cd build || exit 1

mkdir -p ../OUTPUT_RESULT/mpi
mkdir -p ../OUTPUT_RESULT/mpi/1D_inverse
mkdir -p ../OUTPUT_RESULT/mpi/2D_inverse

for CONFIG in "2 4" "2 8" "4 2" "4 4" "4 8" "1 4" "1 8" # TODO x FRA: trova tu la combo migliore
do
  read PROCS THREADS <<< "$CONFIG"
  export OMP_NUM_THREADS=$THREADS

  OUTPUT_FILE="../OUTPUT_RESULT/mpi/1D_inverse/hybrid_p${PROCS}_t${THREADS}.txt"
  echo "===== HYBRID: $PROCS PROCESSES × $THREADS THREADS ====="
  echo "===== HYBRID: $PROCS PROCESSES × $THREADS THREADS =====" > "$OUTPUT_FILE"
  echo "N  sequential  mpi  speedup" >> "$OUTPUT_FILE"

  for N in 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576
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
echo "**************** 2D INVERSE ****************"
echo "   "
echo "   "

for CONFIG in "2 4" "2 8" "4 2" "4 4" "4 8" "1 4" "1 8" # TODO x FRA: trova tu la combo migliore
do
  read PROCS THREADS <<< "$CONFIG"
  export OMP_NUM_THREADS=$THREADS

  OUTPUT_FILE="../OUTPUT_RESULT/mpi/2D_inverse/hybrid_p${PROCS}_t${THREADS}.txt"
  echo "===== HYBRID: $PROCS PROCESSES × $THREADS THREADS ====="
  echo "===== HYBRID: $PROCS PROCESSES × $THREADS THREADS =====" > "$OUTPUT_FILE"
  echo "N  sequential  mpi  speedup" >> "$OUTPUT_FILE"

  for N in 512 1024 2048 4096 8192 #16384 32768 65536 131072 262144 524288 1048576
  do
    echo "--> Running $PROCS MPI processes with $THREADS threads each (N=$N)"
    # echo "--> Running $PROCS MPI processes with $THREADS threads each (N=$N)" >> "$OUTPUT_FILE"
    mpiexec -n $PROCS ./mpi_main 2 $N >> "$OUTPUT_FILE" 2>&1
  done
  echo "Output saved to $OUTPUT_FILE"
  echo "    "
done