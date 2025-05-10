import pyvisa
import time
import csv
import matplotlib.pyplot as plt

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
    data = inst.query(':CALCulate:DATA? POWER')
    print(data)
    inst.write('CLOSE')
    inst.close()
    rm.close()

    return peak_frequency, peak_power

# Example usage:
if __name__ == "__main__":
    # Specify the desired center frequency (in NM) for the display window.
    display_center_frequency = 1550  # Adjust as needed.
    peak_freq, peak_power = measure_peak_power(display_center_frequency)
    print("Final Result:")
    print("Peak Frequency:", peak_freq/10e11 )
    print("Peak Power:", peak_power)
