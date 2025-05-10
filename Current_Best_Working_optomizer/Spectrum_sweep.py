import pyvisa
import time
import csv
import matplotlib.pyplot as plt
import numpy as np

def Santec_Wavelength(wavelength):
    """
    Sets the Santec source wavelength.
    
    Parameters:
        wavelength (float): Desired wavelength to set (units as required by your Santec, e.g., NM).
    """
    # Instrument connection parameters for the Santec
    ip_address = '10.0.3.30'
    port = 5000
    resource_string = f"TCPIP::{ip_address}::{port}::SOCKET"
    
    # Create a resource manager and open the instrument connection.
    rm = pyvisa.ResourceManager()
    inst = rm.open_resource(resource_string)
    inst.write_termination = "\n"
    inst.read_termination = "\n"
    inst.timeout = 60000  # 60 seconds timeout

    # Set the wavelength using the Santec command.
    inst.write(f':WAV {wavelength}')
    
    # Close the connection.
    inst.write('CLOSE')
    inst.close()
    rm.close()

def measure_peak_power(display_center_frequency):
    """
    Set the display window center frequency, perform a single sweep,
    and return the peak frequency (from the marker X value) and peak power (marker Y value).

    Parameters:
        display_center_frequency (float): The center frequency for the display window in NM.

    Returns:
        tuple: (peak_frequency, peak_power) or (None, None) if measurement fails.
    """
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

    # Check instrument identity
    idn = inst.query('*IDN?').strip()
    print("Instrument IDN:", idn)

    # # --- Set up single sweep mode ---
    # print("Initiating single sweep mode...")
    # inst.write(':INITIATE:SMODe SINGLE')
    
    time.sleep(3)  # Wait for the sweep to settle

    # --- Set the center frequency of the display window ---
    # According to your documentation, the command is:
    # :SENSE:WAVELENGTH:CENTER <display_center_frequency>NM
    print(f"Setting display window center frequency to {display_center_frequency} NM...")
    inst.write(f':SENSE:WAVELENGTH:CENTER {display_center_frequency}NM')
    time.sleep(0.5)

    # --- Perform peak search ---
    print("Searching for peak power...")
    time.sleep(2)  # Allow time for the peak search to complete

    inst.write(':CALCULATE:MARKER:STATE 0,ON')
    time.sleep(2)  # Allow time for the peak search to complete

    inst.write(f':CALCULATE:MARKER:X 0,{display_center_frequency}NM')
    time.sleep(2)  # Allow time for the peak search to complete

    # Query the marker's power (Y-axis) and frequency (X-axis) values for marker 0
    power_str = inst.query(':CALCulate:MARKer:Y? 0').strip()
    freq_str = inst.query(':CALCulate:MARKer:X? 0').strip()

    try:
        peak_power = float(power_str)
    except ValueError:
        print("Failed to read peak power. Received:", power_str)
        peak_power = None

    try:
        peak_frequency = float(freq_str)
    except ValueError:
        print("Failed to read peak frequency. Received:", freq_str)
        peak_frequency = None

    # Optional: Save the measurement to a CSV file if valid data is obtained.
    if (peak_power is not None) and (peak_frequency is not None):
        csv_filename = 'peak_power_data.csv'
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Peak Frequency (Units)', 'Peak Power (dBm)'])
            writer.writerow([peak_frequency, peak_power])
        print(f"Peak measurement recorded to {csv_filename}")
    else:
        print("Measurement data incomplete; CSV not saved.")

    # Optionally, plot the result (a single data point)
    # if (peak_power is not None) and (peak_frequency is not None):
    #     plt.figure()
    #     plt.plot([peak_frequency], [peak_power], 'o')
    #     plt.xlabel('Frequency (Units)')
    #     plt.ylabel('Peak Power (dBm)')
    #     plt.title('Recorded Peak Power')
    #     plt.grid(True)
    #     plt.show()
    # else:
    #     print("No valid data to plot.")

    # Close the instrument connection
    inst.write('CLOSE')
    inst.close()
    rm.close()

    return peak_frequency, peak_power


# Main script: Sweep the Santec across different wavelengths and record measurements.
if __name__ == "__main__":
    # Define a sweep range for the Santec wavelength (in NM).
    # For example, sweeping from 1610.0 NM to 1615.0 NM in 1.0 NM steps:
    start_wavelength = 1610.4
    end_wavelength = 1611.4 # end_wavelength is non-inclusive in np.arange so use a value above the maximum.
    step = .005
    sweep_wavelengths = np.arange(start_wavelength, end_wavelength, step)

    # Lists to record data.
    measured_data = []  # Each element will be: (set_wavelength, measured_peak_frequency, measured_peak_power)

    for wl in sweep_wavelengths:
        print("\n-------------------------")
        print(f"Setting Santec wavelength to {wl} NM")
        Santec_Wavelength(wl)
        
        # Wait for the Santec to stabilize after changing wavelengths.
        time.sleep(2)
        # measure_peak_power(wl)
        print(f"Measuring peak power for display center set to {wl} NM")
        peak_freq, peak_power = measure_peak_power(wl)
        print(f"Result: Peak Frequency = {peak_freq} Hz, Peak Power = {peak_power} dBm")
        measured_data.append((wl, peak_freq, peak_power))
        
        # Optional: Additional delay between measurements.
        time.sleep(1)

    # Optionally, save all measurement data to a CSV file.
    csv_filename = 'santec_sweep_data_attenuated_3_cascade_Red_4_14_MATT.csv' # update counter
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Set Wavelength (NM)', 'Measured Peak Frequency (Hz)', 'Measured Peak Power (dBm)'])
        for row in measured_data:
            writer.writerow(row)
    print(f"\nSweep data saved to {csv_filename}")

    # Extract measured peak frequency and power for plotting.
    # Only include measurements where both values are valid.
    x_values = [row[1] for row in measured_data if row[1] is not None and row[2] is not None]
    y_values = [row[2] for row in measured_data if row[1] is not None and row[2] is not None]

    # Plot peak power versus peak frequency.
    plt.figure()
    plt.plot(x_values, y_values, '-o')
    plt.xlabel('Measured Peak Frequency (NM)')
    plt.ylabel('Measured Peak Power (dBm)')
    plt.title('Peak Power vs. Peak Frequency')
    plt.grid(True)
    plt.show()
