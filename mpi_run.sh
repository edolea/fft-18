#!/bin/bash

cd build || exit 1

mkdir -p ../OUTPUT_RESULT/mpi
mkdir -p ../OUTPUT_RESULT/mpi/1D
mkdir -p ../OUTPUT_RESULT/mpi/2D

: <<'EOF'


# Test hybrid configurations
for CONFIG in "2 4" "2 8" "4 2" "4 4" "4 8" "1 4" "1 8" # TODO x FRA: trova tu la combo migliore
do
  read PROCS THREADS <<< "$CONFIG"
  export OMP_NUM_THREADS=$THREADS

  OUTPUT_FILE="../OUTPUT_RESULT/mpi/1D/hybrid_p${PROCS}_t${THREADS}.txt"
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

EOF


echo "   "
echo "   "
echo "**************** 2D CASE ****************"
echo "   "
echo "   "

# Test hybrid configurations
for CONFIG in "2 4" "2 8" "4 2" "4 4" "4 8" "1 4" "1 8" # TODO x FRA: trova tu la combo migliore
do
  read PROCS THREADS <<< "$CONFIG"
  export OMP_NUM_THREADS=$THREADS

  OUTPUT_FILE="../OUTPUT_RESULT/mpi/2D/hybrid_p${PROCS}_t${THREADS}.txt"
  echo "===== HYBRID: $PROCS PROCESSES × $THREADS THREADS ====="
  echo "===== HYBRID: $PROCS PROCESSES × $THREADS THREADS =====" > "$OUTPUT_FILE"
  echo "N  sequential  mpi  speedup" >> "$OUTPUT_FILE"

  for N in 512 1024 2048 4096 # 8192 16384 32768 65536 131072 262144 524288 1048576
  do
    echo "--> Running $PROCS MPI processes with $THREADS threads each (N=$N)"
    # echo "--> Running $PROCS MPI processes with $THREADS threads each (N=$N)" >> "$OUTPUT_FILE"
    mpiexec -n $PROCS ./mpi_main 2 $N >> "$OUTPUT_FILE" 2>&1
  done
  echo "Output saved to $OUTPUT_FILE"
  echo "    "
done



: <<'EOF'


# First test MPI-only for baseline
echo "===== BASELINE: MPI ONLY ====="
export OMP_NUM_THREADS=1
for PROCS in 2 4 # TODO x FRA: tu dovresti riuscire a fare anche 8 e 16
do
  echo ">>>>> USING $PROCS MPI PROCESSES <<<<<"
  for N in 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576
  do
    echo "--> Running $PROCS MPI processes (N=$N)"
    mpiexec -n $PROCS ./mpi_main 1 $N
  done
  echo "    "
done

echo "     "
echo "     "
echo "     "

# Test hybrid configurations
for CONFIG in "2 4" "2 8" "4 2" "4 4" "4 8" "1 4" "1 8" # TODO x FRA: trova tu la combo migliore
do
  read PROCS THREADS <<< "$CONFIG"
  export OMP_NUM_THREADS=$THREADS

  echo "===== HYBRID: $PROCS PROCESSES × $THREADS THREADS ====="
  for N in 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576
  do
    echo "--> Running $PROCS MPI processes with $THREADS threads each (N=$N)"
    mpiexec -n $PROCS ./mpi_main 1 $N
  done
  echo "    "

done


cd build || exit 1

# Set OpenMP environment variables
export OMP_PROC_BIND=true
export OMP_PLACES=cores

# Test different thread configurations
for THREADS in 1 2 4
do
  export OMP_NUM_THREADS=$THREADS
  echo "===== RUNNING WITH $THREADS OpenMP THREADS PER PROCESS ====="

  # Test both 1D and 2D
    # Test different MPI process counts
    for PROCS in 4
    do
      echo ">>>>> USING $PROCS MPI PROCESSES <<<<<"

      # Use fewer large test cases to save time
      for N in 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576
      do
        echo "--> Running $PROCS MPI processes with $THREADS threads each (N=$N)"
        mpiexec -n $PROCS ./mpi_main 1 $N
      done
      echo "============================================="
    done
done
EOF