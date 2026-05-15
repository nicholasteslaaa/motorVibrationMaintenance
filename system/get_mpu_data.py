import serial
import json
import time
import pandas as pd

ser = serial.Serial('/dev/ttyACM0', 115200)
time.sleep(2)

temp = []
counter = 0
try:
    while True:
        line = ser.readline().decode(errors='ignore').strip()

        try:
            data = json.loads(line)

            ax = data["ax"]
            ay = data["ay"]
            az = data["az"]
            gx = data["gx"]
            gy = data["gy"]
            gz = data["gz"]

            mpu_data = {
                "ax": ax,
                "ay": ay,
                "az": az,
                "gx": gx,
                "gy": gy,
                "gz": gz
            }
            temp.append(mpu_data)

            counter += 1
            print(counter,mpu_data)

        except json.JSONDecodeError:
            pass

except KeyboardInterrupt:
    print("\n\nStopped by user (Ctrl+C)")
    print(f"Total data points: {len(temp)}")
    
    for d in temp:
        print(d)
    
    pd.DataFrame(temp).to_csv('data/data.csv',index=False)
    
    with open("mpu_data.json", "w") as f:
        json.dump(temp, f, indent=2)

    ser.close()