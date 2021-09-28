#!/bin/bash

FILENAME="batch_X_timings.csv"
BATCH=32
FREQUENCIES=( 114750000 216750000 318750000 420750000 522750000 624750000 675750000 828750000 905250000 1032750000 1198500000 1236750000 1338750000 1377000000 )

write () {
  echo $1 >> data/$FILENAME;
}

write "Start,End,Frequency";

# Set Max CPU Frequency at the start
# Nano:
# echo "1479000" | sudo tee /sys/devices/system/cpu/cpu0/cpufreq/scaling_{min,max}_freq

# AGX
echo "2265600" | sudo tee /sys/devices/system/cpu/cpu0/cpufreq/scaling_{min,max}_freq
echo "2265600" | sudo tee /sys/devices/system/cpu/cpu0/cpufreq/scaling_{min,max}_freq

for freq in "${FREQUENCIES[@]}"
do
  # Nano
  # echo $freq | sudo tee /sys/devices/gpu.0/devfreq/57000000.gpu/{max,min}_freq
  # echo $freq | sudo tee /sys/devices/gpu.0/devfreq/57000000.gpu/{max,min}_freq

  # AGX
  echo $freq | sudo tee /sys/devices/17000000.gv11b/devfreq/17000000.gv11b/{max,min}_freq
  echo $freq | sudo tee /sys/devices/17000000.gv11b/devfreq/17000000.gv11b/{max,min}_freq

  START=$(date +%s%3N)
  ./tritonserver/clients/bin/perf_analyzer --shared-memory=system -m mobilenet -b $BATCH -f data/$freq.csv -p 15000
  END=$(date +%s%3N)

  write "$START,$END,$freq"
  sleep 10
done
