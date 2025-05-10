import serial
import time

ser = serial.Serial(
    port='COM33',
    baudrate=19200,
    bytesize=serial.EIGHTBITS,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_TWO,
    timeout=1
)
############################################################
# Instruction formatting
#
ser.write(b'*IDN?\r\n')

#time.sleep(.5)
response = ser.read(100)
print(response)
####*IDN?\n
#
############################################################



# ser.write(b'FUNC 1\r\n')
# ser.write(b'FREQ 100\r\n')
# ser.write(b'AMPL 5.00VP\r\n')
# ser.write(b'OFFS 0\r\n')
# ser.write(b'OP1\r\n')

# ser.write(b'ATTL\n')
# ser.write(b'OP1\n')
# For multiple instructions same line reference below. query instructions are seperate from other instructions

#ser.write(b'FUNC 1;FUNC?;FREQ 0;FREQ?;AMPL 0.00VP;AMPL?;OFFS 0;OFFS? \n')
# ser.write(b'ATTL;FREQ 1;FREQ?\n')
# response = ser.read(100)
# print(response)
# ser.write(b'OUTP 1\n')
ser.write(b'FUNC 1;FUNC?;AMPL 0.00VP;AMPL?;OFFS 5;OFFS? \n')
response = ser.read(100)  # adjust the number of bytes if needed
print(response)
ser.write(b'OUTP 1\n')

# ser.write(b'FREQ 1;FUNC? \n')
# response = ser.read(100)
# print (response)

# ser.write(b'FREQ 500;FREQ?\n')
# response = ser.read(100)
# print(response)
# ser.write(b'OUTP 0\n')

# num_iterations = 25

# for i in range(num_iterations):

    # ser.write(b'FUNC 1;FUNC?;AMPL 0.00VP;AMPL?;OFFS 5;OFFS? \n')
    # response = ser.read(100)  # adjust the number of bytes if needed
    # print(response)
    # ser.write(b'OUTP 1\n')
#     #time.sleep(0.5)
#     ser.write(b'FUNC 1;FUNC?;AMPL 0.00VP;AMPL?;OFFS 0;OFFS? \n')

# ser.close



# ##################################################################
# # DS345 TTL Synchronization Example with Adjusted Query and Debugging
# #
# # This example assumes:
# # - TTL levels are set using the ATTL command.
# # - The output is enabled with OUTP 1.
# # - The frequency is set using FREQ.
# #
# # The function generator is configured at the start of each run via
# # the init_function_generator() function.
# #
# # We query the DS345 using STAT? and now check bit 4 (mask 16) to 
# # indicate a TTL high state.
# ##################################################################

# import pyvisa as visa
# import numpy as np
# from time import sleep, time
# import pandas as pd
# import matplotlib.pyplot as plt
# import random
# from CalculatingFidelity import *
# import datetime
# import csv
# import serial

# # ---------------------------
# # Serial Connection to SRS DS345 (Function Generator)
# # ---------------------------
# ser = serial.Serial(
#     port='COM33',
#     baudrate=9600,
#     bytesize=serial.EIGHTBITS,
#     parity=serial.PARITY_NONE,
#     stopbits=serial.STOPBITS_TWO,
#     timeout=1
# )

# def init_function_generator():
#     """
#     Initialize the DS345 function generator at the start of each run:
#       - Set TTL levels using ATTL.
#       - Turn the output on with OUTP 1.
#       - Set the frequency.
#     """
#     # Set TTL levels (adjust as needed)
#     ser.write("ATTL\n".encode('utf-8'))
#     sleep(0.1)
    
#     # Turn on the output
#     ser.write("OUTP 1\n".encode('utf-8'))
#     sleep(0.1)
    
#     # Set frequency (e.g., 300 Hz) and verify setting
#     FREQ = 300  # Adjust frequency as needed
#     cmd = f'FREQ {FREQ};FREQ?\n'
#     ser.write(cmd.encode('utf-8'))
#     response = ser.read(100)
#     print("Function Generator Initialized:", response)

# # ---------------------------
# # TTL Query Functions for DS345
# # ---------------------------
# def query_ttl_state():
#     """
#     Query the DS345's status register via STAT? to determine the TTL state.
    
#     We now assume that bit 4 (mask 1<<4, i.e. 16) indicates TTL high.
#     If that bit is set, return 5; otherwise, return 0.
    
#     (You can try alternative queries if needed.)
#     """
#     ser.write("STAT?\n".encode('utf-8'))
#     response = ser.readline().decode().strip()
#     print("STAT? response:", response)  # Debug print
#     try:
#         stat_val = int(response)
#     except Exception as e:
#         print("Error converting STAT? response to int:", e)
#         return None
#     # Check bit 4 (mask = 1<<4, which is 16)
#     if stat_val & (1 << 4):
#         return 5
#     else:
#         return 0

# def wait_for_ttl_state(desired_state, timeout=1.0, poll_interval=0.01):
#     """
#     Poll the DS345 until its TTL output (as determined by our query function)
#     matches the desired state (5 for high or 0 for low).
#     """
#     start = time()
#     while time() - start < timeout:
#         current_state = query_ttl_state()
#         # Uncomment the next line to debug:
#         # print(f"Current TTL state: {current_state}")
#         if current_state == desired_state:
#             return True
#         sleep(poll_interval)
#     print(f"Timeout waiting for TTL state {desired_state}")
#     return False

# # ---------------------------
# # Utility: Wrap Voltage Values
# # ---------------------------
# def wrap_voltage(value, low=-5000, high=5000):
#     if low <= value <= high:
#         return value
#     else:
#         return ((value - low) % (high - low)) + low

# # ---------------------------
# # Custom EPC Optimizer Using DS345 TTL Synchronization
# # ---------------------------
# class CustomEPCOptimizer:
#     def __init__(self, detector_session, epc_feedback_session,
#                  max_iterations=400, initial_step_size=100, refinement_step_size=10, repeat_iterations=10):
#         self.epc_feedback_session = epc_feedback_session
#         self.detector_session = detector_session
#         self.SLEEP_TIME = 0.1
#         self.max_iterations = max_iterations
#         self.initial_step_size = initial_step_size
#         self.refinement_step_size = refinement_step_size
#         self.repeat_iterations = repeat_iterations
#         self.input_params = []
#         self.cost = []
#         self.fidelity = []
#         self.horizontal_power = []  # list of tuples: (port1, port2, port3, port4) percentages
#         self.diagonal_power = []    # for diagonal measurements
#         self.all_voltages = []
#         self.iteration = -1

#     # ---------------------------
#     # Detector Read Functions
#     # ---------------------------
#     def read_detector_dB(self):
#         sleep(self.SLEEP_TIME)
#         result = self.detector_session.query("READ? 0")
#         ports = result.split(",")
#         port1_dB = float(ports[0].split(":")[-1])
#         port2_dB = float(ports[1])
#         port3_dB = float(ports[2])
#         port4_dB = float(ports[3])
#         return (port1_dB, port2_dB, port3_dB, port4_dB)

#     def read_detector_mW(self):
#         ports_dB = self.read_detector_dB()
#         return tuple(10 ** (dB / 10) for dB in ports_dB)

#     def read_detector_percent(self):
#         ports_mW = self.read_detector_mW()
#         total = sum(ports_mW) + 1e-12
#         return tuple(val / total for val in ports_mW)

#     # ---------------------------
#     # Set EPC Feedback Voltages
#     # ---------------------------
#     def set_epc_feedback(self, voltages):
#         v1, v2, v3, v4 = voltages
#         for i, voltage in enumerate([v1, v2, v3, v4], start=1):
#             sleep(self.SLEEP_TIME)
#             self.epc_feedback_session.write(f"V{i},{int(voltage)}")
#         self.input_params.append(voltages)

#     # ---------------------------
#     # Compute Cost (using TTL-synchronized measurements)
#     # ---------------------------
#     def compute_cost(self, voltages):
#         self.set_epc_feedback(voltages)

#         # Wait until TTL indicates high (for H measurement)
#         if wait_for_ttl_state(5, timeout=1.0):
#             H_values = self.read_detector_mW()
#             horizontal_percent = tuple(val / (sum(H_values) + 1e-12) for val in H_values)
#         else:
#             H_values = (0.0, 0.0, 0.0, 0.0)
#             horizontal_percent = H_values
#         sleep(0.005)

#         # Wait until TTL indicates low (for D measurement)
#         if wait_for_ttl_state(0, timeout=0.5):
#             D_values = self.read_detector_mW()
#             diagonal_percent = tuple(val / (sum(D_values) + 1e-12) for val in D_values)
#         else:
#             D_values = (0.0, 0.0, 0.0, 0.0)
#             diagonal_percent = D_values
#         sleep(0.005)

#         # Extra readings for cost calculation:
#         Hvalues = self.read_detector_mW()
#         sleep(0.005)
#         Dvalues = self.read_detector_mW()
#         sleep(0.005)

#         self.horizontal_power.append(horizontal_percent)
#         self.diagonal_power.append(diagonal_percent)

#         standard_cost = (
#             (((Hvalues[0] / (Hvalues[0] + Hvalues[1] + 1e-12)) - 1) ** 2 +
#              ((Hvalues[1] / (Hvalues[0] + Hvalues[1] + 1e-12))) ** 2 +
#              ((Hvalues[2] / (Hvalues[2] + Hvalues[3] + 1e-12)) - 0.5) ** 2 +
#              ((Hvalues[3] / (Hvalues[2] + Hvalues[3] + 1e-12)) - 0.5) ** 2 +
#              ((Dvalues[0] / (Dvalues[0] + Dvalues[1] + 1e-12)) - 0.5) ** 2 +
#              ((Dvalues[1] / (Dvalues[0] + Dvalues[1] + 1e-12)) - 0.5) ** 2 +
#              ((Dvalues[2] / (Dvalues[2] + Dvalues[3] + 1e-12)) - 1) ** 2 +
#              ((Dvalues[3] / (Dvalues[2] + Dvalues[3] + 1e-12))) ** 2) / 5
#         )
#         total_cost = standard_cost
#         print(f"Iteration: {self.iteration}, Voltages: {voltages}, Cost: {total_cost:.5f}")
#         return total_cost

#     # ---------------------------
#     # Compute and Record Cost (for logging/plotting)
#     # ---------------------------
#     def compute_cost_record(self, voltages):
#         self.iteration += 1
#         self.set_epc_feedback(voltages)

#         if wait_for_ttl_state(5, timeout=1.0):
#             H_values = self.read_detector_mW()
#             horizontal_percent = tuple(val / (sum(H_values) + 1e-12) for val in H_values)
#         else:
#             H_values = (0.0, 0.0, 0.0, 0.0)
#             horizontal_percent = H_values
#         print("Horizontal:", horizontal_percent)
#         sleep(0.005)

#         if wait_for_ttl_state(0, timeout=0.5):
#             D_values = self.read_detector_mW()
#             diagonal_percent = tuple(val / (sum(D_values) + 1e-12) for val in D_values)
#         else:
#             D_values = (0.0, 0.0, 0.0, 0.0)
#             diagonal_percent = D_values
#         print("Diagonal:", diagonal_percent)
#         sleep(0.005)

#         Hvalues = self.read_detector_mW()
#         sleep(0.005)
#         Dvalues = self.read_detector_mW()
#         sleep(0.005)

#         self.horizontal_power.append(horizontal_percent)
#         self.diagonal_power.append(diagonal_percent)

#         standard_cost = (
#             (((Hvalues[0] / (Hvalues[0] + Hvalues[1] + 1e-12)) - 1) ** 2 +
#              ((Hvalues[1] / (Hvalues[0] + Hvalues[1] + 1e-12))) ** 2 +
#              ((Hvalues[2] / (Hvalues[2] + Hvalues[3] + 1e-12)) - 0.5) ** 2 +
#              ((Hvalues[3] / (Hvalues[2] + Hvalues[3] + 1e-12)) - 0.5) ** 2 +
#              ((Dvalues[0] / (Dvalues[0] + Dvalues[1] + 1e-12)) - 0.5) ** 2 +
#              ((Dvalues[1] / (Dvalues[0] + Dvalues[1] + 1e-12)) - 0.5) ** 2 +
#              ((Dvalues[2] / (Dvalues[2] + Dvalues[3] + 1e-12)) - 1) ** 2 +
#              ((Dvalues[3] / (Dvalues[2] + Dvalues[3] + 1e-12))) ** 2) / 5
#         )
#         total_cost = standard_cost
#         self.cost.append(total_cost)

#         Hmax = sum(Hvalues) + 1e-12
#         Dmax = sum(Dvalues) + 1e-12
#         print("HvalueH:", H_values[0] / Hmax)
#         print("HvalueD:", H_values[2] / Hmax)
#         sleep(0.005)
#         print("DvalueH:", Dvalues[0] / Dmax)
#         sleep(0.005)
#         theta, psi, lambdas = calculate_HBasis_new(
#             (Hvalues[0] / (Hvalues[0] + Hvalues[1] + 1e-12)),
#             (Hvalues[2] / (Hvalues[2] + Hvalues[3] + 1e-12)),
#             (Dvalues[0] / (Dvalues[0] + Dvalues[1] + 1e-12))
#         )
#         Fid = fidelity(theta, psi, lambdas)
#         self.fidelity.append(Fid)
#         self.all_voltages.append(voltages)

#         print(f"Iteration: {self.iteration}, Voltages: {voltages}, Cost: {total_cost:.5f}")
#         return total_cost

#     # ---------------------------
#     # Main Optimization Routine (MATT)
#     # ---------------------------
#     def MATT(self, initial_voltages):
#         current_voltages = np.array(initial_voltages, dtype=np.float64)
#         current_cost = self.compute_cost_record(current_voltages)
    
#         for i in range(self.max_iterations):
#             if self.cost and current_cost >= 0.1 and i >= 2 and abs(self.cost[-1] - self.cost[-2]) < 1e-3:
#                 step_size = self.initial_step_size
#                 current_voltages += np.random.uniform(-step_size, step_size, size=current_voltages.shape)
#                 current_voltages = np.array([wrap_voltage(v) for v in current_voltages])
#                 print("Stochastic perturbation applied.")
#             else:
#                 if current_cost < 0.035:
#                     print("Smallest stepsize")
#                     step_size = 20 * 3 * np.log((10 * current_cost) + (1 - 1E-8))
#                 elif current_cost < 0.038:
#                     print("Smaller stepsize")
#                     step_size = 25 * 3 * np.log((10 * current_cost) + (1 - 1E-8))
#                 elif current_cost < 0.04:
#                     print("Smaller stepsize")
#                     step_size = 50 * 3 * np.log((10 * current_cost) + (1 - 1E-8))
#                 else:
#                     step_size = 60 * 3 * np.log((10 * current_cost) + (1 - 1E-8))
#                     print("Larger stepsize")
#                 print("Step Size:", step_size)
#                 for j in range(len(current_voltages)):
#                     perturbed_voltages = np.copy(current_voltages)
#                     perturbed_voltages[j] += step_size
#                     perturbed_voltages = np.array([wrap_voltage(v) for v in perturbed_voltages])
#                     perturbed_cost = self.compute_cost(perturbed_voltages)
#                     if perturbed_cost < current_cost:
#                         current_cost = perturbed_cost
#                         current_voltages = perturbed_voltages
#             current_cost = self.compute_cost_record(current_voltages)
#             print(f"Current best cost: {current_cost:.5f}")
#         print(f"Final Voltages after Optimization: {current_voltages}, Final Cost: {current_cost:.5f}, Fidelity: {self.fidelity[-1]:.5f}")
#         return current_voltages

# # ---------------------------
# # Data Plotting Function
# # ---------------------------
# def plot_data(horizontal_power, diagonal_power, cost, fidelity, voltages, filename):
#     with open(filename, 'a', newline='') as file:
#         writer = csv.writer(file)
#         for i in range(len(horizontal_power)):
#             row = [
#                 horizontal_power[i][0], horizontal_power[i][1], horizontal_power[i][2], horizontal_power[i][3],
#                 diagonal_power[i][0], diagonal_power[i][1], diagonal_power[i][2], diagonal_power[i][3],
#                 cost[i], fidelity[i],
#                 voltages[i][0], voltages[i][1], voltages[i][2], voltages[i][3]
#             ]
#             writer.writerow(row)
#     data_H = pd.DataFrame(horizontal_power, columns=['Port 1', 'Port 2', 'Port 3', 'Port 4'])
#     data_D = pd.DataFrame(diagonal_power, columns=['Port 1', 'Port 2', 'Port 3', 'Port 4'])
#     cost_df = pd.Series(cost, name='Cost')
#     fidelity_df = pd.Series(fidelity, name='Fidelity')
#     voltages_df = pd.DataFrame(voltages, columns=['V1', 'V2', 'V3', 'V4'])

#     plt.figure(figsize=(10, 5))
#     plt.plot(data_H.index, data_H['Port 1'], label='Port 1 Horizontal Power')
#     plt.plot(data_H.index, data_H['Port 2'], label='Port 2 Horizontal Power')
#     plt.plot(data_H.index, data_H['Port 3'], label='Port 3 Horizontal Power')
#     plt.plot(data_H.index, data_H['Port 4'], label='Port 4 Horizontal Power')
#     plt.xlabel('Iteration')
#     plt.ylabel('Power Percentage')
#     plt.title('Horizontal Power Percentages Over Time')
#     plt.legend()
#     plt.grid(True)
#     plt.show()

#     plt.figure(figsize=(10, 5))
#     plt.plot(data_D.index, data_D['Port 1'], label='Port 1 Diagonal Power')
#     plt.plot(data_D.index, data_D['Port 2'], label='Port 2 Diagonal Power')
#     plt.plot(data_D.index, data_D['Port 3'], label='Port 3 Diagonal Power')
#     plt.plot(data_D.index, data_D['Port 4'], label='Port 4 Diagonal Power')
#     plt.xlabel('Iteration')
#     plt.ylabel('Power Percentage')
#     plt.title('Diagonal Power Percentages Over Time')
#     plt.legend()
#     plt.grid(True)
#     plt.show()

#     plt.figure(figsize=(10, 5))
#     plt.plot(cost_df.index, cost_df, label='Cost')
#     plt.xlabel('Iteration')
#     plt.ylabel('Cost')
#     plt.title('Cost Over Time')
#     plt.yscale("log")
#     plt.legend()
#     plt.grid(True)
#     plt.show()

#     plt.figure(figsize=(10, 5))
#     plt.plot(fidelity_df.index, fidelity_df, label='Fidelity')
#     plt.xlabel('Iteration')
#     plt.ylabel('Fidelity')
#     plt.title('Fidelity Over Iterations')
#     plt.yscale("log")
#     plt.legend()
#     plt.grid(True)
#     plt.show()

#     plt.figure(figsize=(10, 5))
#     plt.plot(voltages_df.index, voltages_df['V1'], label='Voltage 1')
#     plt.plot(voltages_df.index, voltages_df['V2'], label='Voltage 2')
#     plt.plot(voltages_df.index, voltages_df['V3'], label='Voltage 3')
#     plt.plot(voltages_df.index, voltages_df['V4'], label='Voltage 4')
#     plt.xlabel('Iteration')
#     plt.ylabel('Voltage')
#     plt.title('Voltages Over Time')
#     plt.legend()
#     plt.grid(True)
#     plt.show()

# # ---------------------------
# # Main Routine
# # ---------------------------
# def main():
#     # Initialize the function generator at the start of the run.
#     init_function_generator()
    
#     DETECTOR_RESOURCE_STR = "TCPIP0::10.0.3.17::5000::SOCKET"
#     EPC_FEEDBACK_RESOURCE_STR = "ASRL3::INSTR"
#     INITIAL_VOLTAGES = [-389, -531, 87, 277]
#     print("Initial Voltages:", INITIAL_VOLTAGES)
#     MAX_ITERATIONS = 100
#     REPEAT_ITERATIONS = 10

#     resource_manager = visa.ResourceManager()
#     detector_session = resource_manager.open_resource(DETECTOR_RESOURCE_STR)
#     detector_session.read_termination = "\n"
#     detector_session.write_termination = "\n"
#     epc_feedback_session = resource_manager.open_resource(EPC_FEEDBACK_RESOURCE_STR)
#     epc_feedback_session.read_termination = "\r\n"
#     epc_feedback_session.write_termination = "\r\n"
#     epc_feedback_session.query_termination = "\r\n"

#     epc_optimizer = CustomEPCOptimizer(detector_session, epc_feedback_session,
#                                        max_iterations=MAX_ITERATIONS, repeat_iterations=REPEAT_ITERATIONS)

#     optimized_voltages = epc_optimizer.MATT(INITIAL_VOLTAGES)
#     print("Optimized Voltages:", optimized_voltages)

#     timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
#     filename = fr'C:\Users\Alice1\Documents\Custom Algorithm Optimization\DO NOT TOUCH THESE FILES\BOB_Optimization_Values_{timestamp}.csv'
#     header = ["H1", "H2", "H3", "H4", "D1", "D2", "D3", "D4", "Cost", "Fidelity", "V1", "V2", "V3", "V4"]
#     with open(filename, 'w', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow(header)
    
#     plot_data(epc_optimizer.horizontal_power, epc_optimizer.diagonal_power, 
#               epc_optimizer.cost, epc_optimizer.fidelity, 
#               epc_optimizer.all_voltages, filename)
    
#     ser.close()

# if __name__ == "__main__":
#     main()
