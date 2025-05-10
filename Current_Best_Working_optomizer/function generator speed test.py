import pyvisa
import numpy as np
import matplotlib.pyplot as plt
import serial
import time

#############################
# Function Generator Setup  #
#############################

# Open serial connection to the function generator
ser = serial.Serial(
    port='COM33',
    baudrate=9600,
    bytesize=serial.EIGHTBITS,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_TWO,
    timeout=1
)

# Verify communication with the function generator
ser.write(b'*IDN?\r\n')
response = ser.read(100)
print("Function Generator ID:", response.decode().strip())

# Configure the function generator to produce a square wave:
# 5 Vpp, 100 Hz, 0 V offset.
ser.write(b'FUNC 1;FUNC?;FREQ 100;FREQ?;AMPL 5.00VP;AMPL?;OFFS 0;OFFS? \n')
response = ser.read(100)
print("Function Generator settings:", response.decode().strip())

# Turn on the output continuously (square wave will run continuously)
ser.write(b'OUTP 1\n')


#############################
# Oscilloscope Setup (Ethernet) #
#############################

# Connect to the oscilloscope using PyVISA over Ethernet.
rm = pyvisa.ResourceManager()
scope_resource = "TCPIP0::10.0.3.15::INSTR"
scope = rm.open_resource(scope_resource)
scope.timeout = 5000  # Timeout for each command in ms

# (Optional) Verify connection by printing the oscilloscope's ID.
idn = scope.query("*IDN?")
print("Oscilloscope ID:", idn.strip())

# Set up data transfer parameters (this is done once)
scope.write("DATA:SOURCE CH1")       # Use Channel 1
scope.write("DATA:ENCdg ASCII")      # Use ASCII encoding for easy parsing
scope.write("DATA:WIDth 1")          # One data point per sample (if supported)


############################################
# Measurement Loop: 100 Iterations Example #
############################################

# Adjustable parameters:
iteration_count = 100  # Number of iterations (adjustable)
iteration_pause = 0.015  # Pause between iterations (in seconds) changed from 100 ms. .001 seems about optimal. maybe try sticking with .015 for a 7.48 second 100 iteration
#EPC caps at 7.26 seconds for 100 iterations (100 Hz should work)

# Thresholds for determining the falling edge in the square wave:
threshold_high = 4.9   # Voltage near 5V
threshold_low = 0.1    # Voltage near 0V

# Lists to hold measured discharge times and iteration durations
discharge_times = []
iteration_durations = []

total_start = time.time()

for i in range(iteration_count):
    iter_start = time.time()
    
    # Query the oscilloscope for the current waveform capture.
    # Since the square wave is continuously running, each capture should include one or more periods.
    raw_data = scope.query("CURVE?")
    try:
        # Convert the comma‐separated string into a NumPy array of floats.
        data_points = np.array(raw_data.strip().split(','), dtype=float)
    except Exception as e:
        print(f"Iteration {i+1}: Error parsing waveform data: {e}")
        iteration_durations.append(time.time() - iter_start)
        continue
        
    # Retrieve the time axis scaling information from the oscilloscope.
    try:
        xzero = float(scope.query("WFMPre:XZE?"))
        xincrement = float(scope.query("WFMPre:XIN?"))
    except Exception as e:
        print(f"Iteration {i+1}: Error retrieving time axis info: {e}")
        iteration_durations.append(time.time() - iter_start)
        continue
        
    num_points = len(data_points)
    time_axis = xzero + np.arange(num_points) * xincrement

    # Find the falling edge within the captured waveform:
    # 1. Find the first index where the voltage is near 5V.
    indices_high = np.where(data_points >= threshold_high)[0]
    if len(indices_high) == 0:
        print(f"Iteration {i+1}: No data point found near 5V.")
        iteration_durations.append(time.time() - iter_start)
        continue
    start_index = indices_high[0]
    
    # 2. Find the first index after that where the voltage drops to near 0V.
    indices_low = np.where(data_points[start_index:] <= threshold_low)[0]
    if len(indices_low) == 0:
        print(f"Iteration {i+1}: No data point found where voltage drops to near 0V after the 5V level.")
        iteration_durations.append(time.time() - iter_start)
        continue
    end_index = start_index + indices_low[0]
    discharge_time = time_axis[end_index] - time_axis[start_index]
    discharge_times.append(discharge_time)
    
    print(f"Iteration {i+1}: Discharge time = {discharge_time:.6f} seconds")
    
    iter_end = time.time()
    iteration_durations.append(iter_end - iter_start)
    
    # Optional pause between iterations
    time.sleep(iteration_pause)

total_end = time.time()
total_duration = total_end - total_start

if iteration_durations:
    average_iteration_duration = np.mean(iteration_durations)
else:
    average_iteration_duration = 0

if discharge_times:
    average_discharge_time = np.mean(discharge_times)
else:
    average_discharge_time = 0

print("\n========== Results ==========")
print(f"Total time for {iteration_count} iterations: {total_duration:.6f} seconds")
print(f"Average time per iteration: {average_iteration_duration:.6f} seconds")
print(f"Average discharge time (5V -> 0V): {average_discharge_time:.6f} seconds")

# Optionally, plot the discharge times for each iteration.
plt.figure(figsize=(8, 4))
plt.plot(range(1, len(discharge_times)+1), discharge_times, marker='o', linestyle='-')
plt.xlabel("Iteration")
plt.ylabel("Discharge Time (s)")
plt.title("Discharge Time per Iteration")
plt.grid(True)
plt.show()

##################################
# Close the Instrument Connections
##################################
scope.close()
ser.close()
