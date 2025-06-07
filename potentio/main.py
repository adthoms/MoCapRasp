import serial
import time

port = "/dev/ttyACM0"  # Check with ls /dev/ttyACM* if unsure
baud = 9600

ser = serial.Serial(port, baud, timeout=1)
time.sleep(2)  # Give the Teensy time to reset

print("Reading potentiometer values...\n")
try:
    while True:
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        if line:
            try:
                values = list(map(int, line.split(",")))
                print(f"Pots: {values}")  # [pot0, pot1, ..., pot5]
            except ValueError:
                print(f"[WARN] Bad data: {line}")
except KeyboardInterrupt:
    print("Stopped by user.")
finally:
    ser.close()
