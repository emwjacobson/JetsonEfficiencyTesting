import socketserver
import sys
import serial
import time
import threading

POWER_PORT = "/dev/ttyUSB0"

LISTEN = True

class MyTCPHandler(socketserver.BaseRequestHandler):
    def setup(self):
        print(f"New connection from {self.client_address[0]}")

    def handle(self):
        global LISTEN

        while True:
            # START
            data = self.request.recv(1024).strip().decode("UTF-8").split(",")
            print(data)
            while data[0] != "START":
                data = self.request.recv(1024).strip().decode("UTF-8").split(",")
                print(data, end="")
            print(f"STARTING {'-'.join(data[1:])}")

            LISTEN = True
            name = f"AGX-{'-'.join(data[1:])}.data"
            monitor = threading.Thread(target=listen_power, args=(name,))
            monitor.start()

            # END
            data = self.request.recv(1024).strip().decode("UTF-8").split(",")
            print(data)
            while data[0] != "DONE":
                data = self.request.recv(1024).strip().decode("UTF-8").split(",")
                print(data, end="")

            print("KILLING THREAD")
            LISTEN = False
            monitor.join(timeout=1)
            with open(name, "a") as f:
                f.seek(0)
                f.write(data[1] + "," + data[2])

            print(f"DONE {data[1]} FLOPS, {data[2]} ITERATIONS")

def listen_power(name):
    with serial.Serial(POWER_PORT, 115200) as s, open(name, "w") as f:
        s.write("#L,W,3,E,,{};".format(int(1)).encode('ascii'))
        while LISTEN:
            l = s.readline().decode('ascii')
            if l[0:2] == '#d':
                vals = l.split(';')[0].split(',')[3:]
                now = int(time.time()*1000)
                m = {
                    'time': now,
                    'watts': float(vals[0])/10.0,
                    'volts': float(vals[1])/10.0,
                    'amps': float(vals[2])/10.0,
                    'watt-hours': float(vals[3])/10.0,
                    'dollars': float(vals[4])/1000.0,
                    'watt hours monthly': float(vals[5]),
                    'dollars monthly': float(vals[6])*10.0,
                    'power factor': float(vals[7]),
                    'duty cycle': float(vals[8]),
                    'power cycle': float(vals[9]),
                    'frequency': float(vals[10])/10.0,
                    'volt-amps': float(vals[10])/10.0
                }

                msg = f"{m['time']},{m['watts']}"
                print(msg)
                f.write(msg+"\n")
            f.flush()



HOST, PORT = "", 8888
if __name__ == "__main__":
    # listen_power("data.csv")
    with socketserver.TCPServer((HOST, PORT), MyTCPHandler) as server:
        print("Waiting for connections")
        server.serve_forever()