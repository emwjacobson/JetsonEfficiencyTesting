import os
import csv

FREQ = {
  "AGX": [114750000, 216750000, 318750000, 420750000, 522750000, 624750000, 675750000, 828750000, 905250000, 1032750000, 1198500000, 1236750000, 1338750000, 1377000000],
  "Nano": [76800000, 153600000, 230400000, 307200000, 384000000, 460800000, 537600000, 614400000, 691200000, 768000000, 844800000, 921600000]
}

SQUARE_SIZES = [64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1088, 1152, 1216, 1280, 1344, 1408, 1472, 1536, 1600, 1664, 1728, 1792, 1856, 1920, 1984, 2048]

matrix_filename = './data/{device}/square_all_frequency/{device}-float-{m_size}-tensor-{freq}.data'

def format_matrix(device: str):
  with open(F"{device}_matrix.csv", 'w') as csv_out:
    csv_writer = csv.writer(csv_out)
    csv_writer.writerow(['Matrix Size', 'GPU Frequency', 'Average Watts', 'FLOPS', 'Iterations', 'Seconds'])
    for f in FREQ[device]:
      for size in SQUARE_SIZES:
        print(matrix_filename.format(device=device, m_size=size, freq=f))
        with open(matrix_filename.format(device=device, m_size=size, freq=f), mode="r") as data:
          lines = data.readlines()
          power_data = [float(l.split(',')[1].strip()) for l in lines[1:-1]]
          avg_power = sum(power_data)/len(power_data)
          flops, iterations, seconds = lines[-1].split(',')
          csv_writer.writerow([size, f, avg_power, flops, iterations, seconds])



ml_batch_filename = './triton/data/{device}/batch_{batch_size}_{type}.csv'

def format_ml_batch(device: str):
  with open(F"{device}_ml_batch.csv", 'w') as csv_out:
    csv_writer = csv.writer(csv_out)
    csv_writer.writerow(['Frequency', 'Batch Size', 'Inferences/Second', 'Avg Watts', 'Client Send', 'Network+Server Send/Recv', 'Server Queue', 'Server Compute Input', 'Server Compute Infer', 'Server Compute Output', 'Client Recv', 'p50 latency', 'p90 latency', 'p95 latency', 'p99 latency'])
    for b_size in [1, 2, 4, 8, 16, 32]:
      with open(ml_batch_filename.format(device=device, batch_size=b_size, type='power')) as file_power, open(ml_batch_filename.format(device=device, batch_size=b_size, type='timings')) as file_timings, open(ml_batch_filename.format(device=device, batch_size=b_size, type='data')) as file_data:
        timings = [[int(t2.strip()) for t2 in t.split(',')] for t in file_timings.readlines()[1:]]
        power = [[float(l2.strip()) for l2 in l.split(',')] for l in file_power.readlines()]
        data = [[float(l2.strip()) for l2 in l.split(',')] for l in file_data.readlines()[1:]]
        for i, f_based in enumerate(timings):
          # start = f_based[0]
          # end = f_based[1]
          # freq = f_based[2]
          power_temp = [f[1] for f in power if f[0] > f_based[0] and f[0] < f_based[1]][1:-1]
          avg_power = sum(power_temp)/len(power_temp)
          csv_writer.writerow([f_based[2], b_size, data[i][2], avg_power, data[i][3], data[i][4], data[i][5], data[i][6], data[i][7], data[i][8], data[i][9], data[i][10], data[i][11], data[i][12], data[i][13]])



specific_f = {
  "agx": [905250000, 1377000000],
  "nano": [691200000, 768000000]
}

def format_ml_fixed(device: str):
  pass

if __name__ == "__main__":
  # format_matrix("AGX")
  # format_matrix("Nano")
  format_ml_batch("agx")
  format_ml_batch("nano")