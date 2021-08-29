#!/bin/bash

FILENAME="XXXXXX_timings.csv"

write () {
  echo $1 >> data/$FILENAME;
}

write "Start,End,Batch Size";

for i in {1..32}
do
  START=$(date +%s%3N)
  ./tritonserver/clients/bin/perf_analyzer --shared-memory=system -m mobilenet -b $i -f data/$i.csv
  END=$(date +%s%3N)

  write "$START,$END,$i"
  sleep 10
done
