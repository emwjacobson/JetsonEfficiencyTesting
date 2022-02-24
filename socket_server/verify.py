import os

path = "../data/Nano/square_all_frequency/"
files = os.listdir(path)

data = []

for file_name in files:
    with open(path+file_name, 'r') as f:
        lines = f.readlines()
        power_data = lines[0:-1]
        flops = lines[-1]

        # power_data should be an array with each element
        # formatted as timestamp,powerdata
        for pd in power_data:
            if "," not in pd:
                print("Error in power data:", file_name)
            else:
                ts, power = pd.split(",")
                if float(power) <= 2:
                    print("Power error?", file_name)

        # flops, time, and num_inferences should be the last line
        if len(flops.split(',')) != 3:
            print("Error in last line", file_name)