#!/bin/bash

FILENAME="batch_X_timings.csv"
BATCH=1
FREQUENCIES=( 76800000 153600000 230400000 307200000 384000000 460800000 537600000 614400000 691200000 768000000 844800000 921600000 )

write () {
  echo $1 >> data/$FILENAME;
}

write "Start,End,Frequency";

# Set Max CPU Frequency at the start
echo "1479000" | sudo tee /sys/devices/system/cpu/cpu0/cpufreq/scaling_{min,max}_freq

for freq in "${FREQUENCIES[@]}"
do
  echo $freq | sudo tee /sys/devices/gpu.0/devfreq/57000000.gpu/{min,max}_freq
  START=$(date +%s%3N)
  ./tritonserver/clients/bin/perf_analyzer --shared-memory=system -m mobilenet -b $BATCH -f data/$freq.csv -p 15000
  END=$(date +%s%3N)

  write "$START,$END,$freq"
  sleep 10
done
