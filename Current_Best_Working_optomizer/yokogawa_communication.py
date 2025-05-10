import pyvisa
import time
import csv
import matplotlib.pyplot as plt

# Instrument connection parameters
ip_address = '10.0.3.21'
port = 10001
resource_string = f"TCPIP::{ip_address}::{port}::SOCKET"

# Create a resource manager and open the instrument
rm = pyvisa.ResourceManager()
inst = rm.open_resource(resource_string)
inst.write_termination = "\n"
inst.read_termination = "\n"
inst.timeout = 60000  # Set timeout to 60 seconds

# Authentication
print("Authenticating...")
response = inst.query('OPEN "osa"').strip()
print("Response:", response)  # Expect: AUTHENTICATECRAM-MD5.
response = inst.query('osa').strip()
print("Authentication Response:", response)  # Expect: ready

# Optional: Check instrument identity
idn = inst.query('*IDN?').strip()
print("Instrument IDN:", idn)

# --- Set up sweep and continuous repeat mode ---
# These commands may need adjustment based on your instrument’s manual.
print("Setting continuous sweep mode...")
inst.write(':INITiate:SMODe AUTO 3;:INITiate')
inst.write(':INITiate:SMODe SINGLe 1;:INITiate')
time.sleep(3)  # Wait for the sweep to initiate

# # Set up marker centering (if required)
# inst.write(':CALCulate:MARKer:SCENter')
# time.sleep(0.2)

# --- Continuous query for peak search data ---
record_duration = 10  # seconds
start_time = time.time()
data = []  # List to hold [frequency, power] pairs
# data = inst.write('CALCulate:DATA?')
# print(data)
# print("Continuous sweep active. Recording peak data...")


while time.time() - start_time < record_duration:
    try:
        inst.write(':INIT:SMOD SINGL|1;:INIT')

        # Trigger the peak search command.
        # Adjust the command if needed—here we send the command once per loop.
        inst.write(':CALCulate:MARKer:MAXimum:SCENter:AUTO OFF|ON|0|1')
        time.sleep(0.5)  # Allow time for the marker to update

        # Trigger the peak search (marker goes to maximum point)
        inst.write(':CALCulate:MARKer:MAXimum')
        time.sleep(0.5)  # Allow time for the instrument to update the marker

        # Query the marker's frequency (X-axis) for marker 1
        freq_str = inst.query(':CALCulate:MARKer:X? 0').strip()
        # Query the marker's power (Y-axis) for marker 1
        power_str = inst.query(':CALCulate:MARKer:Y? 0').strip()


        try:
            frequency = float(freq_str)
            power = float(power_str)
            data.append([frequency, power])
            print(f"Recorded: Frequency = {frequency} Hz, Power = {power} dBm")
        except ValueError:
            print("Invalid data received:", freq_str, power_str)
    except Exception as e:
        print("Peak search query failed:", e)
    time.sleep(0.5)  # Sampling interval

# Save collected data to CSV
csv_filename = 'laser_sweep_peak_data.csv'
with open(csv_filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Frequency (Hz)', 'Power (dBm)'])
    writer.writerows(data)
print(f"Data written to {csv_filename}")

# Plot the data if available
if data:
    frequencies = [row[0] for row in data]
    powers = [row[1] for row in data]
    plt.figure()
    plt.plot(frequencies, powers, '-o')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power (dBm)')
    plt.title('Peak Laser Sweep Data (Continuous Mode)')
    plt.grid(True)
    plt.show()
else:
    print("No valid data collected. Plotting skipped.")

# Close the connection
inst.write('CLOSE')
inst.close()
rm.close()
