#!/bin/bash

EXEC=./cputnl
GRAPH=out.livejournal-groupmemberships
OUTFILE=cpu_thread_scaling.txt
NODES=500000

# List of threads to test
THREAD_LIST=(1 2 4 8 16)

# Prepare output file
echo "#threads communities lpa_seconds" > $OUTFILE

for T in "${THREAD_LIST[@]}"
do
    echo "Running with $T threads..."

    # Set environment variables for OpenMP
    export OMP_NUM_THREADS=$T
    export OMP_PROC_BIND=true
    export OMP_PLACES=cores

    # Run program and capture RESULT line
    RESULT=$($EXEC $GRAPH $NODES | grep RESULT)

    # Extract nodes, communities, time
    COMMS=$(echo $RESULT | awk '{print $3}')
    TIME=$(echo $RESULT | awk '{print $4}')

    # Save to file
    echo "$T $COMMS $TIME" >> $OUTFILE
done

echo "Done. Results saved in $OUTFILE"