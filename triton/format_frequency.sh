#!/bin/bash

FILENAME="batch_X_data.csv"
FREQUENCIES=( 76800000 153600000 230400000 307200000 384000000 460800000 537600000 614400000 691200000 768000000 844800000 921600000 )

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
