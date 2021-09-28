#!/bin/bash

FILENAME="batch_X_data.csv"
FREQUENCIES=( 114750000 216750000 318750000 420750000 522750000 624750000 675750000 828750000 905250000 1032750000 1198500000 1236750000 1338750000 1377000000 )

write () {
  echo $1 >> data/$FILENAME;
}

write-n () {
  echo -n $1 >> data/$FILENAME;
}

write "Frequency,Concurrency,Inferences/Second,Client Send,Network+Server Send/Recv,Server Queue,Server Compute Input,Server Compute Infer,Server Compute Output,Client Recv,p50 latency,p90 latency,p95 latency,p99 latency"

for freq in "${FREQUENCIES[@]}"
do

write-n "$freq,";
write $(tail -n 1 data/$freq.csv);
rm data/$freq.csv;

done
