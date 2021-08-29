#!/bin/bash

FILENAME="XXXXXX_data.csv"

write () {
  echo $1 >> data/$FILENAME;
}

write-n () {
  echo -n $1 >> data/$FILENAME;
}

write "Batch Size,Concurrency,Inferences/Second,Client Send,Network+Server Send/Recv,Server Queue,Server Compute Input,Server Compute Infer,Server Compute Output,Client Recv,p50 latency,p90 latency,p95 latency,p99 latency"

for i in {1..32}
do

write-n "$i,";
write $(tail -n 1 data/$i.csv);
rm data/$i.csv;

done