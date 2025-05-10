import pyvisa
import time
import csv
import threading
import queue
import tkinter as tk
from tkinter import scrolledtext
import matplotlib
# Use TkAgg backend for embedding matplotlib in Tkinter
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt  # For optional manual plotting

# ------------------------------------------------
# Instrument Connection and Initialization Section
# ------------------------------------------------
ip_address = '10.0.3.21'
port = 10001
resource_string = f"TCPIP::{ip_address}::{port}::SOCKET"

try:
    rm = pyvisa.ResourceManager()
    inst = rm.open_resource(resource_string)
    inst.write_termination = "\n"
    inst.read_termination = "\n"
    inst.timeout = 60000  # 60-second timeout
except Exception as e:
    print(f"Error establishing connection: {e}")
    exit(1)

# Authentication
try:
    print("Authenticating...")
    auth_response = inst.query('OPEN "osa"').strip()
    print("Auth Response:", auth_response)
    second_response = inst.query('osa').strip()
    print("Authentication confirmation:", second_response)
except Exception as e:
    print(f"Error during authentication: {e}")
    exit(1)

# Query instrument identity for confirmation
try:
    idn = inst.query('*IDN?').strip()
    print("Instrument IDN:", idn)
except Exception as e:
    print(f"Error querying IDN: {e}")

# Set continuous sweep mode
try:
    print("Setting continuous sweep mode...")
    inst.write(':INITiate:SMODe AUTO 3;:INITiate')
    inst.write(':INITiate:SMODe SINGLe 1;:INITiate')
    time.sleep(3)  # Give time for configuration
except Exception as e:
    print(f"Error configuring sweep mode: {e}")

# ------------------------------------------------
# Global Variables and Queue for Thread Communication
# ------------------------------------------------
data = []  # To store peak marker [frequency, power] pairs.
msg_queue = queue.Queue()  # For GUI log messages.
last_full_trace = []       # To hold full trace (spectrum) data.

# ------------------------------------------------
# Function to Update the Embedded Plot
# ------------------------------------------------
def update_plot():
    global last_full_trace, line, canvas
    if last_full_trace:
        x_vals = list(range(len(last_full_trace)))
        line.set_data(x_vals, last_full_trace)
        ax.relim()
        ax.autoscale_view()
        canvas.draw()
    root.after(500, update_plot)

# ------------------------------------------------
# Instrument Polling Function (Background Thread)
# ------------------------------------------------
def instrument_poll():
    global last_full_trace, data
    record_duration = 30  # seconds (adjust as needed)
    start_time = time.time()

    while time.time() - start_time < record_duration:
        try:
            # Trigger a single sweep
            msg_queue.put("Triggering single sweep...\n")
            inst.write(':INIT:SMOD SINGLe')
            inst.write(':INIT')
            time.sleep(0.2)
            
            # Wait for sweep completion using OPC.
            opc_timeout = 10  # seconds
            opc_start = time.time()
            opc_response = ""
            while time.time() - opc_start < opc_timeout:
                try:
                    opc_response = inst.query('*OPC?').strip()
                    msg_queue.put(f"*OPC? response: {opc_response}\n")
                    if opc_response == '1':
                        break
                except Exception as e:
                    msg_queue.put(f"Error querying *OPC?: {e}\n")
                time.sleep(0.2)
            else:
                msg_queue.put("Timeout waiting for sweep completion (*OPC? not '1').\n")
            
            # Query the full trace data
            try:
                raw_trace = inst.query(':CALCulate:DATA? SPECTRUM').strip()
                msg_queue.put(f"Raw full trace: {raw_trace}\n")
                try:
                    trace_data = [float(x) for x in raw_trace.split(',') if x.strip() != '']
                    last_full_trace = trace_data
                    msg_queue.put("Full trace updated successfully.\n")
                except Exception as parse_err:
                    msg_queue.put(f"Error parsing full trace: {parse_err}\n")
                    last_full_trace = []
            except Exception as query_err:
                msg_queue.put(f"Error querying full trace: {query_err}\n")
            
            # Query peak marker data
            try:
                inst.write(':CALCulate:MARKer:MAXimum')
                time.sleep(0.5)
                freq_str = inst.query(':CALCulate:MARKer:X? 0').strip()
                power_str = inst.query(':CALCulate:MARKer:Y? 0').strip()
                try:
                    frequency = float(freq_str)
                    power = float(power_str)
                    data.append([frequency, power])
                    msg_queue.put(f"Peak Recorded: Freq = {frequency} Hz, Power = {power} dBm\n")
                except Exception as conv_err:
                    msg_queue.put(f"Conversion error for peak data: {conv_err}\n")
            except Exception as peak_err:
                msg_queue.put(f"Error querying peak marker: {peak_err}\n")
            
        except Exception as e:
            msg_queue.put(f"Instrument polling error: {e}\n")
        
        time.sleep(0.5)

    # End of recording period: Save CSV and close connection.
    csv_filename = 'laser_sweep_peak_data.csv'
    try:
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Frequency (Hz)', 'Power (dBm)'])
            writer.writerows(data)
        msg_queue.put(f"Peak data saved to {csv_filename}\n")
    except Exception as csv_err:
        msg_queue.put(f"Error saving CSV: {csv_err}\n")
    
    try:
        inst.write('CLOSE')
        inst.close()
        rm.close()
        msg_queue.put("Instrument connection closed.\n")
    except Exception as close_err:
        msg_queue.put(f"Error closing instrument connection: {close_err}\n")

    msg_queue.put("Data collection finished.\n")

# ------------------------------------------------
# Set Up the Tkinter GUI
# ------------------------------------------------
root = tk.Tk()
root.title("Yokogawa OSA Display Emulator")

# Scrolled text widget for logs
screen = scrolledtext.ScrolledText(root, width=80, height=10, wrap='word')
screen.pack(padx=10, pady=10)

def update_screen():
    while not msg_queue.empty():
        line_text = msg_queue.get()
        screen.insert(tk.END, line_text)
        screen.see(tk.END)
    root.after(100, update_screen)

# Embedded matplotlib figure for full sweep display
fig = Figure(figsize=(6, 4))
ax = fig.add_subplot(111)
line, = ax.plot([], [], '-')
ax.set_xlabel("Data Point")
ax.set_ylabel("Power (dBm)")
ax.set_title("Full Trace (Spectrum Data)")
ax.grid(True)

canvas = FigureCanvasTkAgg(fig, master=root)
canvas_widget = canvas.get_tk_widget()
canvas_widget.pack(padx=10, pady=10)

# Optional: Button for manual plotting
def manual_plot():
    if last_full_trace:
        plt.figure()
        plt.plot(range(len(last_full_trace)), last_full_trace, '-o')
        plt.xlabel("Data Point")
        plt.ylabel("Power (dBm)")
        plt.title("Manual Plot of Full Trace")
        plt.grid(True)
        plt.show()
    else:
        msg_queue.put("No full trace data available to plot.\n")

plot_button = tk.Button(root, text="Manual Plot", command=manual_plot)
plot_button.pack(pady=5)

# ------------------------------------------------
# Start Threads and GUI Update Loops
# ------------------------------------------------
poll_thread = threading.Thread(target=instrument_poll, daemon=True)
poll_thread.start()
update_screen()
update_plot()

root.mainloop()
