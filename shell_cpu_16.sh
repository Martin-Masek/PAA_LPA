#!/bin/bash

EXEC=./cputnl
GRAPH=out.citeseer
OUTFILE=communities_cpu_16_citeseer_2.txt


# set OpenMP threads
export OMP_NUM_THREADS=16

echo "#nodes communities lpa_seconds" > $OUTFILE

for N in $(seq 50000 50000 400000)
do
    echo "Running N=$N with $OMP_NUM_THREADS threads"

    RESULT=$($EXEC $GRAPH $N | grep RESULT)

    NODES=$(echo $RESULT | awk '{print $2}')
    COMMS=$(echo $RESULT | awk '{print $3}')
    TIME=$(echo $RESULT | awk '{print $4}')

    echo "$NODES $COMMS $TIME" >> $OUTFILE
done

echo "Done."