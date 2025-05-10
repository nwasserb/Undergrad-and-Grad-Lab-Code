import pyvisa as visa
import numpy as np
from time import sleep, time
import pandas as pd
import matplotlib.pyplot as plt
import random
from CalculatingFidelity import *
import datetime
import csv
import serial

# ---------------------------
# Serial Connection to SRS DS345 (Function Generator)
# ---------------------------
ser = serial.Serial(
    port='COM33',
    baudrate=19200,
    bytesize=serial.EIGHTBITS,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_TWO,
    timeout=1
)

def init_function_generator():
    """
    Initialize the DS345 function generator at the start of each run:
      - Set TTL levels to 5V (high) and 0V (low) using ATTL.
      - Set the trigger source to SINGLE.
      - Set the burst count (BCNT) to 1.
      - Turn on the output (OUTP 1).
      - Set the frequency.
    """
    ser.write("ATTL;FREQ 8 ;FREQ?;OUTP1\n".encode('utf-8'))
    response = ser.read(100)
    print("Setup frequency:")
    print(response)

# ---------------------------
# TTL Query Functions using $ATD? for DS345
# ---------------------------
def query_ttl_state():
    """
    Query the DS345's output voltage using the $ATD? command.
    Returns 5 if voltage is above threshold (i.e. TTL high), else 0.
    In the dead zone ("NO MANS LAND") no measurement is taken.
    """
    ser.reset_input_buffer()
    ser.write("$ATD? 0,2\n".encode('utf-8'))
    response = ser.readline().decode().strip()
    if not response:
        print("Warning: $ATD? returned an empty string. Returning default value 5.")
        return 5
    try:
        voltage = 10 * (float(response) / 4095)
    except Exception as e:
        print("Error converting $ATD? response to float:", e)
        return 5
    if voltage > 4.0:
        return 5
    elif voltage < 1.0:
        return 0
    else:
        return None  # explicitly return None in the dead zone

def wait_for_ttl_state(desired_state, timeout=1.0, poll_interval=0.01):
    """
    Poll the DS345 until its TTL output matches the desired state (5 for high, 0 for low).
    A short poll_interval (now 0.01s) gives the device time to settle.
    """
    start = time()
    while time() - start < timeout:
        current_state = query_ttl_state()
        if current_state == desired_state:
            return True
        sleep(poll_interval)
    print(f"Timeout waiting for TTL state {desired_state}")
    return False

def wrap_voltage(value, low=-5000, high=5000):
    if low <= value <= high:
        return value
    else:
        if (value // (2*high)) == 1: 
            return low - ((-np.abs(value) % 5000) * np.sign(value))
        else:
            return (-np.abs(value) % 5000) * np.sign(value)

# ---------------------------
# Custom EPC Optimizer Using DS345 TTL Synchronization
# ---------------------------
class CustomEPCOptimizer:
    def __init__(self, detector_session, epc_feedback_session,
                 max_iterations=400, initial_step_size=100, refinement_step_size=10, repeat_iterations=10):
        self.epc_feedback_session = epc_feedback_session
        self.detector_session = detector_session
        self.SLEEP_TIME = 0.001  # Adjust as needed
        self.max_iterations = max_iterations
        self.initial_step_size = initial_step_size
        self.refinement_step_size = refinement_step_size
        self.repeat_iterations = repeat_iterations
        self.input_params = []
        self.cost = []
        self.fidelity = []
        self.horizontal_power = []  # list of tuples: (port1, port2, port3, port4) percentages
        self.diagonal_power = []    # for diagonal measurements
        self.all_voltages = []
        self.iteration = -1

    def read_detector_dB(self):
        sleep(self.SLEEP_TIME)
        result = self.detector_session.query("READ? 0")
        ports = result.split(",")
        port1_dB = float(ports[0].split(":")[-1])
        port2_dB = float(ports[1])
        port3_dB = float(ports[2])
        port4_dB = float(ports[3])
        return (port1_dB, port2_dB, port3_dB, port4_dB)

    def read_detector_mW(self):
        ports_dB = self.read_detector_dB()
        return tuple(10 ** (dB / 10) for dB in ports_dB)

    def read_detector_percent(self):
        ports_mW = self.read_detector_mW()
        total = sum(ports_mW) + 1e-12
        return tuple(val / total for val in ports_mW)

    def set_epc_feedback(self, voltages):
        v1, v2, v3, v4 = voltages
        for i, voltage in enumerate([v1, v2, v3, v4], start=1):
            sleep(self.SLEEP_TIME)
            self.epc_feedback_session.write(f"V{i},{int(voltage)}")
        self.input_params.append(voltages)

    def compute_cost(self, voltages):
        self.set_epc_feedback(voltages)
        # Allow a short delay for the system to settle
        sleep(0.005)
        if wait_for_ttl_state(5, timeout=1.0):
            H_values = self.read_detector_mW()
            horizontal_percent = tuple(val / (sum(H_values) + 1e-12) for val in H_values)
        else:
            H_values = (0.0, 0.0, 0.0, 0.0)
            horizontal_percent = H_values

        if wait_for_ttl_state(0, timeout=2.0):
            D_values = self.read_detector_mW()
            diagonal_percent = tuple(val / (sum(D_values) + 1e-12) for val in D_values)
        else:
            D_values = (0.0, 0.0, 0.0, 0.0)
            diagonal_percent = D_values

        self.horizontal_power.append(horizontal_percent)
        self.diagonal_power.append(diagonal_percent)

        standard_cost = (
            (((H_values[0] / (H_values[0] + H_values[1] + 1e-12)) - 1) ** 2 +
             ((H_values[1] / (H_values[0] + H_values[1] + 1e-12))) ** 2 +
             ((H_values[2] / (H_values[2] + H_values[3] + 1e-12)) - 0.5) ** 2 +
             ((H_values[3] / (H_values[2] + H_values[3] + 1e-12)) - 0.5) ** 2 +
             ((D_values[0] / (D_values[0] + D_values[1] + 1e-12)) - 0.5) ** 2 +
             ((D_values[1] / (D_values[0] + D_values[1] + 1e-12)) - 0.5) ** 2 +
             ((D_values[2] / (D_values[2] + D_values[3] + 1e-12)) - 1) ** 2 +
             ((D_values[3] / (D_values[2] + D_values[3] + 1e-12))) ** 2) / 5
        )
        total_cost = standard_cost
        return total_cost

    def compute_cost_record(self, voltages):
        self.iteration += 1
        self.set_epc_feedback(voltages)
        sleep(0.005)
        if wait_for_ttl_state(5, timeout=1.0):
            H_values = self.read_detector_mW()
            horizontal_percent = tuple(val / (sum(H_values) + 1e-12) for val in H_values)
        else:
            H_values = (0.0, 0.0, 0.0, 0.0)
            horizontal_percent = H_values

        if wait_for_ttl_state(0, timeout=2.0):
            D_values = self.read_detector_mW()
            diagonal_percent = tuple(val / (sum(D_values) + 1e-12) for val in D_values)
        else:
            D_values = (0.0, 0.0, 0.0, 0.0)
            diagonal_percent = D_values

        self.horizontal_power.append(horizontal_percent)
        self.diagonal_power.append(diagonal_percent)

        standard_cost = (
            (((H_values[0] / (H_values[0] + H_values[1] + 1e-12)) - 1) ** 2 +
             ((H_values[1] / (H_values[0] + H_values[1] + 1e-12))) ** 2 +
             ((H_values[2] / (H_values[2] + H_values[3] + 1e-12)) - 0.5) ** 2 +
             ((H_values[3] / (H_values[2] + H_values[3] + 1e-12)) - 0.5) ** 2 +
             ((D_values[0] / (D_values[0] + D_values[1] + 1e-12)) - 0.5) ** 2 +
             ((D_values[1] / (D_values[0] + D_values[1] + 1e-12)) - 0.5) ** 2 +
             ((D_values[2] / (D_values[2] + D_values[3] + 1e-12)) - 1) ** 2 +
             ((D_values[3] / (D_values[2] + D_values[3] + 1e-12))) ** 2) / 5
        )
        total_cost = standard_cost
        self.cost.append(total_cost)

        theta, psi, lambdas = calculate_HBasis_new(
            (H_values[0] / (H_values[0] + H_values[1] + 1e-12)),
            (H_values[2] / (H_values[2] + H_values[3] + 1e-12)),
            (D_values[0] / (D_values[0] + D_values[1] + 1e-12))
        )
        Fid = fidelity(theta, psi, lambdas)
        self.fidelity.append(Fid)
        self.all_voltages.append(voltages)

        print(f"Iteration: {self.iteration}, Cost: {total_cost:.5f}")
        return total_cost
    
    def compute_cost_record_fid(self, voltages):
        self.iteration += 1
        self.set_epc_feedback(voltages)
        sleep(0.005)
        if wait_for_ttl_state(5, timeout=1.0):
            H_values = self.read_detector_mW()
            horizontal_percent = tuple(val / (sum(H_values) + 1e-12) for val in H_values)
        else:
            H_values = (0.0, 0.0, 0.0, 0.0)
            horizontal_percent = H_values

        if wait_for_ttl_state(0, timeout=2.0):
            D_values = self.read_detector_mW()
            diagonal_percent = tuple(val / (sum(D_values) + 1e-12) for val in D_values)
        else:
            D_values = (0.0, 0.0, 0.0, 0.0)
            diagonal_percent = D_values

        self.horizontal_power.append(horizontal_percent)
        self.diagonal_power.append(diagonal_percent)

        standard_cost = (
            (((H_values[0] / (H_values[0] + H_values[1] + 1e-12)) - 1) ** 2 +
             ((H_values[1] / (H_values[0] + H_values[1] + 1e-12))) ** 2 +
             ((H_values[2] / (H_values[2] + H_values[3] + 1e-12)) - 0.5) ** 2 +
             ((H_values[3] / (H_values[2] + H_values[3] + 1e-12)) - 0.5) ** 2 +
             ((D_values[0] / (D_values[0] + D_values[1] + 1e-12)) - 0.5) ** 2 +
             ((D_values[1] / (D_values[0] + D_values[1] + 1e-12)) - 0.5) ** 2 +
             ((D_values[2] / (D_values[2] + D_values[3] + 1e-12)) - 1) ** 2 +
             ((D_values[3] / (D_values[2] + D_values[3] + 1e-12))) ** 2) / 5
        )
        total_cost = standard_cost
        self.cost.append(total_cost)

        theta, psi, lambdas = calculate_HBasis_new(
            (H_values[0] / (H_values[0] + H_values[1] + 1e-12)),
            (H_values[2] / (H_values[2] + H_values[3] + 1e-12)),
            (D_values[0] / (D_values[0] + D_values[1] + 1e-12))
        )
        Fid = fidelity(theta, psi, lambdas)
        self.fidelity.append(Fid)
        self.all_voltages.append(voltages)

        print(f"Iteration: {self.iteration}, Cost: {total_cost:.5f}")
        return Fid

    def MATT(self, initial_voltages):
        """
        Revised MATT optimizer:
          - For each iteration, each voltage is perturbed in both positive and negative directions.
          - If a perturbation lowers the cost (even by a small amount), the new voltages are accepted.
          - If no improvement is made during an iteration, the step size is reduced.
          - If the step size becomes too small and the cost is still high, a moderate random jump is applied.
        """
        current_voltages = np.array(initial_voltages, dtype=np.float64)
        current_cost = self.compute_cost_record(current_voltages)
        step_size = self.initial_step_size
        min_step_threshold = 1e-2  # minimum step size before a random jump
        for i in range(self.max_iterations):
            improved = False
            prev_cost = current_cost
            # try perturbing each voltage in both directions
            for j in range(len(current_voltages)):
                for delta in [step_size, -step_size]:
                    perturbed_voltages = np.copy(current_voltages)
                    perturbed_voltages[j] += delta
                    perturbed_voltages = np.array([wrap_voltage(v) for v in perturbed_voltages])
                    perturbed_cost = self.compute_cost(perturbed_voltages)
                    if perturbed_cost < current_cost:
                        current_cost = perturbed_cost
                        current_voltages = perturbed_voltages
                        improved = True
            # record the new cost
            current_cost = self.compute_cost_record(current_voltages)
            print(f"Iteration {i}: Cost = {current_cost:.5f}, Step size = {step_size:.3f}")
            # if no improvement, reduce the step size
            if not improved:
                step_size *= 0.95
                # if step size becomes very small and cost is still high, perform a moderate random jump
                if step_size < min_step_threshold and current_cost > 0.05:
                    print("No improvement; applying random jump.")
                    random_jump = np.random.uniform(-200, 200, size=current_voltages.shape)
                    current_voltages += random_jump
                    current_voltages = np.array([wrap_voltage(v) for v in current_voltages])
                    step_size = self.initial_step_size  # reset step size
                    current_cost = self.compute_cost_record(current_voltages)
            # Optionally, you can break early if the cost is below a desired threshold
            if current_cost < 0.01:
                break

        print(f"Final Voltages after Optimization: {current_voltages}, Final Cost: {current_cost:.5f}, Fidelity: {self.fidelity[-1]:.5f}")
        return current_voltages

# ---------------------------
# Data Plotting Function with CSV Writing Fix
# ---------------------------
def plot_data(horizontal_power, diagonal_power, cost, fidelity, voltages, filename):
    n = min(len(horizontal_power), len(diagonal_power), len(cost), len(fidelity), len(voltages))
    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)
        for i in range(n):
            row = [
                horizontal_power[i][0], horizontal_power[i][1], horizontal_power[i][2], horizontal_power[i][3],
                diagonal_power[i][0], diagonal_power[i][1], diagonal_power[i][2], diagonal_power[i][3],
                cost[i], fidelity[i],
                voltages[i][0], voltages[i][1], voltages[i][2], voltages[i][3]
            ]
            writer.writerow(row)
    
    data_H = pd.DataFrame(horizontal_power[:n], columns=['Port 1', 'Port 2', 'Port 3', 'Port 4'])
    data_D = pd.DataFrame(diagonal_power[:n], columns=['Port 1', 'Port 2', 'Port 3', 'Port 4'])
    cost_df = pd.Series(cost[:n], name='Cost')
    fidelity_df = pd.Series(fidelity[:n], name='Fidelity')
    voltages_df = pd.DataFrame(voltages[:n], columns=['V1', 'V2', 'V3', 'V4'])

    plt.figure(figsize=(10, 5))
    plt.plot(data_H.index, data_H['Port 1'], label='Port 1 Horizontal Power')
    plt.plot(data_H.index, data_H['Port 2'], label='Port 2 Horizontal Power')
    plt.plot(data_H.index, data_H['Port 3'], label='Port 3 Horizontal Power')
    plt.plot(data_H.index, data_H['Port 4'], label='Port 4 Horizontal Power')
    plt.xlabel('Iteration')
    plt.ylabel('Power Percentage')
    plt.title('Horizontal Power Percentages Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(data_D.index, data_D['Port 1'], label='Port 1 Diagonal Power')
    plt.plot(data_D.index, data_D['Port 2'], label='Port 2 Diagonal Power')
    plt.plot(data_D.index, data_D['Port 3'], label='Port 3 Diagonal Power')
    plt.plot(data_D.index, data_D['Port 4'], label='Port 4 Diagonal Power')
    plt.xlabel('Iteration')
    plt.ylabel('Power Percentage')
    plt.title('Diagonal Power Percentages Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(cost_df.index, cost_df, label='Cost')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title('Cost Over Time')
    plt.yscale("log")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(fidelity_df.index, fidelity_df, label='Fidelity')
    plt.xlabel('Iteration')
    plt.ylabel('Fidelity')
    plt.title('Fidelity Over Iterations')
    plt.yscale("log")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(voltages_df.index, voltages_df['V1'], label='Voltage 1')
    plt.plot(voltages_df.index, voltages_df['V2'], label='Voltage 2')
    plt.plot(voltages_df.index, voltages_df['V3'], label='Voltage 3')
    plt.plot(voltages_df.index, voltages_df['V4'], label='Voltage 4')
    plt.xlabel('Iteration')
    plt.ylabel('Voltage')
    plt.title('Voltages Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

# ---------------------------
# Main Routine
# ---------------------------
def main():
    init_function_generator()
    
    DETECTOR_RESOURCE_STR = "TCPIP0::10.0.3.17::5000::SOCKET"
    EPC_FEEDBACK_RESOURCE_STR = "ASRL3::INSTR"
    INITIAL_VOLTAGES = [0, 0, 0, 0]
    print("Initial Voltages:", INITIAL_VOLTAGES)
    MAX_ITERATIONS = 300
    REPEAT_ITERATIONS = 10

    resource_manager = visa.ResourceManager()
    detector_session = resource_manager.open_resource(DETECTOR_RESOURCE_STR)
    detector_session.read_termination = "\n"
    detector_session.write_termination = "\n"
    epc_feedback_session = resource_manager.open_resource(EPC_FEEDBACK_RESOURCE_STR)
    epc_feedback_session.read_termination = "\r\n"
    epc_feedback_session.write_termination = "\r\n"
    epc_feedback_session.query_termination = "\r\n"

    epc_optimizer = CustomEPCOptimizer(detector_session, epc_feedback_session,
                                       max_iterations=MAX_ITERATIONS, repeat_iterations=REPEAT_ITERATIONS)

    # Measure only the optimization time:
    opt_start = time()
    optimized_voltages = epc_optimizer.MATT(INITIAL_VOLTAGES)
    opt_time = time() - opt_start
    print("Optimized Voltages:", optimized_voltages)
    print(f"Optimization time: {opt_time:.2f} seconds")

    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    filename = fr'C:\Users\Alice1\Documents\Custom Algorithm Optimization\DO NOT TOUCH THESE FILES\BOB_Optimization_Values_{timestamp}.csv'
    header = ["H1", "H2", "H3", "H4", "D1", "D2", "D3", "D4", "Cost", "Fidelity", "V1", "V2", "V3", "V4"]
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
    
    plot_data(epc_optimizer.horizontal_power, epc_optimizer.diagonal_power, 
              epc_optimizer.cost, epc_optimizer.fidelity, 
              epc_optimizer.all_voltages, filename)
    
    ser.close()

if __name__ == "__main__":
    main()
