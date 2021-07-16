import os

path = "../data/AGX/"
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

        # flops should be the last line, with a single float number
        if "," in flops:
            print("Error in FLOPS:", file_name)