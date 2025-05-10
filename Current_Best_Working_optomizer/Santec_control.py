import pyvisa
import time
import csv
import matplotlib.pyplot as plt

def Santec_Wavelength(wavelength):
    """
    Set the display window center frequency, perform a single sweep,
    and return the peak frequency (from the marker X value) and peak power (marker Y value).

    Parameters:
        display_center_frequency (float): The center frequency for the display window in NM.

    Returns:
        tuple: (peak_frequency, peak_power) or (None, None) if measurement fails.
    """
    # Instrument connection parameters
    ip_address = '10.0.3.30'
    port = 5000
    resource_string = f"TCPIP::{ip_address}::{port}::SOCKET"

    # Create a resource manager and open the instrument
    rm = pyvisa.ResourceManager()
    inst = rm.open_resource(resource_string)
    inst.write_termination = "\n"
    inst.read_termination = "\n"
    inst.timeout = 60000  # Set timeout to 60 seconds


    inst.write(f':WAV {wavelength}')
    inst.write('CLOSE')
    inst.close()
    rm.close()

    return 

# Example usage:
if __name__ == "__main__":
    # Specify the desired center frequency (in NM) for the display window.
    wavelength = 1551
    Santec_Wavelength(wavelength)
    print("DONE")
    
