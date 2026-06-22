#!/bin/bash

EXEC=./tlpa_cuda
GRAPH=out.citeseer
OUTFILE=communities_cuda_citeseer_2.txt

echo "#nodes communities lpa_seconds" > $OUTFILE

for N in $(seq 50000 50000 400000)
do
    echo "Running N=$N"

    RESULT=$($EXEC $GRAPH $N | grep RESULT)

    NODES=$(echo $RESULT | awk '{print $2}')
    COMMS=$(echo $RESULT | awk '{print $3}')
    TIME=$(echo $RESULT | awk '{print $4}')

    echo "$NODES $COMMS $TIME" >> $OUTFILE

    # optional GPU cooldown
    sleep 2
done

echo "Done. Results saved to $OUTFILE"
