#!/bin/bash

for Dim in 1
do
  for PROCS in 4
  do
    for N in 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576 2097152 4194304 8388608 16777216 33554432 67108864 134217728 268435456 536870912 1073741824 2147483648 4294967296 8589934592 17179869184 34359738368 68719476736 137438953472 274877906944 549755813888 1099511627776 2199023255552 4398046511104 8796093022208 17592186044416
    do
        echo "--> Running $PROCS PROCESSES with N=$N"
        mpiexec -n $PROCS ./build_mpi/mpi_main $Dim $N
    done
    echo "============================================="
  done
done

: <<'EOF'
output_file="mpi_output.log"
# echo "MPI Execution Results" > $output_file

for PROCS in 2 4 8
do
    for N in 8 16 32 64 128 256
    do
        echo "Running $PROCS PROCESSES with N=$N"
        echo "Running $PROCS PROCESSES with N=$N" >> $output_file
        mpiexec -n $PROCS ./build_mpi/mpi_main 1 $N >> $output_file
    done
    echo "============================================="
    echo "=============================================" >> $output_file
done
EOF