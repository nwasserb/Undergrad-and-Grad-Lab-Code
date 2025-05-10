import pyvisa as visa
import time
import math
from time import sleep
import serial
import threading
import sys

def display_percentages(session, stop_event):
    """
    Continuously queries the VISA session for measurements,
    calculates the relative percentages, and prints them.
    """
    while not stop_event.is_set():
        try:
            result = session.query('READ? 0')
            ports = result.split(',')
            port1 = float(ports[0].split(':')[-1])
            port2 = float(ports[1])
            port3 = float(ports[2])
            port4 = float(ports[3])
            
            # Convert dBm readings to mW
            port1mW = pow(10, port1 / 10)
            port2mW = pow(10, port2 / 10)
            port3mW = pow(10, port3 / 10)
            port4mW = pow(10, port4 / 10)
            totalmW = port1mW + port2mW + port3mW + port4mW
            
            # Calculate relative percentages
            relPort1 = (port1mW / totalmW) * 100
            relPort2 = (port2mW / totalmW) * 100
            relPort3 = (port3mW / totalmW) * 100
            relPort4 = (port4mW / totalmW) * 100

            # Clear the line and print the updated percentages
            sys.stdout.write("\r\033[K")  # \033[K clears from cursor to the end of the line
            sys.stdout.write(f"Port 1: {relPort1:6.2f}% | Port 2: {relPort2:6.2f}% | Port 3: {relPort3:6.2f}% | Port 4: {relPort4:6.2f}%")
            sys.stdout.flush()
        except Exception as e:
            sys.stdout.write("\r\033[K")
            print("Error reading percentages:", e)
        sleep(0.1)

def main():
    # ---------------------------
    # Open VISA session first so we can display measurements
    # ---------------------------
    try:
        resourceManager = visa.ResourceManager()
        dev = 'TCPIP0::10.0.3.17::5000::SOCKET'
        session = resourceManager.open_resource(dev)
        session.read_termination = '\n'
        session.write_termination = '\n'
        print("VISA Session Opened. Resources:", resourceManager.list_resources())
    except visa.VisaIOError as e:
        print(f"Error opening VISA session: {e}")
        return

    # ---------------------------
    # Open the serial connection for basis initialization
    # ---------------------------
    try:
        ser = serial.Serial(
            port='COM33',
            baudrate=19200,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_TWO,
            timeout=1
        )
    except Exception as e:
        print("Error opening serial port:", e)
        session.close()
        return

    # ---------------------------
    # Step 1: Set Horizontal Bases via serial command
    # ---------------------------
    try:
        ser.write(b'FUNC 1;FUNC?;FREQ 0;FREQ?;AMPL 0.00VP;AMPL?;OFFS 5;OFFS? \n')
        response = ser.read(100)
        print("\nResponse for Horizontal Bases:", response)
        ser.write(b'OUTP 1\n')
        print("Set Horizontal Bases, press enter when done")
    except Exception as e:
        print("Error during Horizontal Bases initialization:", e)
    
    # ---------------------------
    # Display VISA percentages while waiting for user to continue
    # ---------------------------
    stop_event = threading.Event()
    disp_thread = threading.Thread(target=display_percentages, args=(session, stop_event))
    disp_thread.start()
    
    input("\n\nPress Enter when ready to setup Diagonal Bases...")  # User sees the continuously updated percentages
    stop_event.set()
    disp_thread.join()
    
    # ---------------------------
    # Step 2: Set Diagonal Bases via serial command
    # ---------------------------
    try:
        ser.write(b'FUNC 1;FUNC?;FREQ 0;FREQ?;AMPL 0.00VP;AMPL?;OFFS 0;OFFS? \n')
        response = ser.read(100)
        print("\nResponse for Diagonal Bases:", response)
        ser.write(b'OUTP 1\n')
        print("Set up Diagonal Bases, press enter when done")
    except Exception as e:
        print("Error during Diagonal Bases initialization:", e)
    
    # Optionally, display percentages again until final confirmation
    stop_event = threading.Event()
    disp_thread = threading.Thread(target=display_percentages, args=(session, stop_event))
    disp_thread.start()
    
    input("\n\nPress Enter when ready to complete basis initialization...")
    stop_event.set()
    disp_thread.join()
    
    print("\nBasis initialization complete")
    
    # ---------------------------
    # (Optional) Continue with continuous VISA monitoring
    # ---------------------------
    try:
        while True:
            sleep(0.09)
            result = session.query('READ? 0')
            ports = result.split(',')
            port1 = float(ports[0].split(':')[-1])
            port2 = float(ports[1])
            port3 = float(ports[2])
            port4 = float(ports[3])
            
            port1mW = pow(10, port1 / 10)
            port2mW = pow(10, port2 / 10)
            port3mW = pow(10, port3 / 10)
            port4mW = pow(10, port4 / 10)
            totalmW = port1mW + port2mW + port3mW + port4mW
            
            relPort1 = (port1mW / totalmW) * 100
            relPort2 = (port2mW / totalmW) * 100
            relPort3 = (port3mW / totalmW) * 100
            relPort4 = (port4mW / totalmW) * 100
            
            sys.stdout.write("\r\033[K")
            sys.stdout.write(f"Port 1: {relPort1:6.2f}% | Port 2: {relPort2:6.2f}% | Port 3: {relPort3:6.2f}% | Port 4: {relPort4:6.2f}%")
            sys.stdout.flush()
    except visa.VisaIOError as e:
        print(f'\nError: {e}')
    finally:
        session.close()

if __name__ == "__main__":
    main()
