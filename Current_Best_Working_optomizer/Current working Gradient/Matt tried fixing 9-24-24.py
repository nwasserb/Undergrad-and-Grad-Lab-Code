##################################################################
# Current Best code as of 9-26-24                                #
##################################################################


import pyvisa as visa
import numpy as np
from time import sleep
import vxi11
import pandas as pd
import matplotlib.pyplot as plt
import random
from CalculatingFidelity import * 
import datetime
import csv
# Connect to the power supply
supply = vxi11.Instrument("10.0.3.7")

# Function to wrap voltage
def wrap_voltage(value, low=-5000, high=5000):
    if(value >= low and value <= high):
        return value
    else:
        if (value // 2*high) == 1: 
            return low - ((-np.abs(value) % 5000) * np.sign(value))
        else:
            return (-np.abs(value) % 5000) * np.sign(value)
        # return -5000

class CustomEPCOptimizer:
        
    def __init__(self, detector_session, epc_feedback_session, max_iterations=400, initial_step_size=100, refinement_step_size=10, repeat_iterations=10):
        self.epc_feedback_session = epc_feedback_session
        self.detector_session = detector_session
        self.SLEEP_TIME = 0.05
        self.max_iterations = max_iterations
        self.initial_step_size = initial_step_size
        self.refinement_step_size = refinement_step_size
        self.repeat_iterations = repeat_iterations  # Number of times to repeat final voltages
        self.input_params = []
        self.cost = []
        self.fidelity = []
        self.horizontal_power = []
        self.diagonal_power = []
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
        ports_mW = tuple(float(pow(10, port_dB / 10)) for port_dB in ports_dB)
        return ports_mW

    def read_detector_percent(self):
        ports_mW = self.read_detector_mW()
        total_power = sum(ports_mW)
        if total_power == 0:
            return (0.0, 0.0, 0.0, 0.0)
        ports_percent = tuple(port_mW / total_power for port_mW in ports_mW)
        return ports_percent

    def set_epc_feedback(self, voltages):
        v1, v2, v3, v4 = voltages
        for i, inp_voltage in enumerate([v1, v2, v3, v4], 1):
            sleep(self.SLEEP_TIME)
            self.epc_feedback_session.write(f"V{i},{int(inp_voltage)}") #Changed query to write so code doesnt time out waiting for response - ms
        self.input_params.append(voltages)

    def compute_cost(self, voltages): 
        
        self.set_epc_feedback(voltages)
        supply.write("OUTP CH1,ON")
        sleep(.25)
        horizontal_ports_percent = self.read_detector_percent()
        sleep(.25)
        supply.write("OUTP CH1,OFF")
        sleep(.25)
        diagonal_ports_percent = self.read_detector_percent()

        # # self.horizontal_power.append(horizontal_ports_percent)
        # # self.diagonal_power.append(diagonal_ports_percent)

        # # Standard cost based on ideal values
        # standard_cost = (
        #     (horizontal_ports_percent[0] - 0.50) ** 2 + (horizontal_ports_percent[1]) ** 2 +
        #     (horizontal_ports_percent[2] - 0.25) ** 2 + (horizontal_ports_percent[3] - 0.25) ** 2 +
        #     (diagonal_ports_percent[0] - 0.25) ** 2 + (diagonal_ports_percent[1] - 0.25) ** 2 +
        #     (diagonal_ports_percent[2] - 0.50) ** 2 + (diagonal_ports_percent[3]) ** 2
        # )

        # Penalty to ensure ports 3 and 4 are similar and ports 2 and 3 are not
        # penalty_cost = (
        #     (horizontal_ports_percent[2] - horizontal_ports_percent[3]) ** 2 +  # Encourage ports 3 and 4 to be similar
        #     (horizontal_ports_percent[1] - horizontal_ports_percent[2]) ** 2 * 0.5  # Discourage ports 2 and 3 from being too similar
        # )



        supply.write("OUTP CH1,ON")
        sleep(.25)
        Hvalues = self.read_detector_mW()
        Hmaxvalues = sum(Hvalues)
        sleep(.25)
        supply.write("OUTP CH1,OFF")
        sleep(.25)
        Dvalues = self.read_detector_mW()
        Dmaxvalues = sum(Dvalues)

        
        self.horizontal_power.append(horizontal_ports_percent)
        self.diagonal_power.append(diagonal_ports_percent)

        # Standard cost based on ideal values
        standard_cost = (
                (((Hvalues[0]/(Hvalues[0]+Hvalues[1])) - 1) ** 2 + ((Hvalues[1]/(Hvalues[0]+Hvalues[1]))) ** 2 +
                ((Hvalues[2]/(Hvalues[2]+Hvalues[3])) - 0.5) ** 2 + ((Hvalues[3]/(Hvalues[2]+Hvalues[3])) - 0.5) ** 2 +
                ((Dvalues[0]/(Dvalues[0]+Dvalues[1])) - 0.5) ** 2 + ((Dvalues[1]/(Dvalues[0]+Dvalues[1])) - 0.5) ** 2 +
                ((Dvalues[2]/(Dvalues[2]+Dvalues[3])) - 1) ** 2 + ((Dvalues[3]/(Dvalues[2]+Dvalues[3]))) ** 2)/5
        )
        total_cost = standard_cost #+ penalty_cost
        # self.cost.append(total_cost)

        print(f"Iteration: {self.iteration}, Voltages: {voltages}, Cost: {total_cost:.5f}")
        return total_cost
    
    def compute_cost_record(self, voltages):
            self.iteration += 1
            self.set_epc_feedback(voltages)
            supply.write("OUTP CH1,ON")
            sleep(.25)
            horizontal_ports_percent = self.read_detector_percent()
            sleep(.25)
            supply.write("OUTP CH1,OFF")
            sleep(.25)
            diagonal_ports_percent = self.read_detector_percent()

            # self.horizontal_power.append(horizontal_ports_percent)
            # self.diagonal_power.append(diagonal_ports_percent)

            # # Standard cost based on ideal values
            # standard_cost = (
            #     (horizontal_ports_percent[0] - 0.50) ** 2 + (horizontal_ports_percent[1]) ** 2 +
            #     (horizontal_ports_percent[2] - 0.25) ** 2 + (horizontal_ports_percent[3] - 0.25) ** 2 +
            #     (diagonal_ports_percent[0] - 0.25) ** 2 + (diagonal_ports_percent[1] - 0.25) ** 2 +
            #     (diagonal_ports_percent[2] - 0.50) ** 2 + (diagonal_ports_percent[3]) ** 2
            # )

            supply.write("OUTP CH1,ON")
            sleep(.25)
            Hvalues = self.read_detector_mW()
            Hmaxvalues = sum(Hvalues)
            sleep(.25)
            supply.write("OUTP CH1,OFF")
            sleep(.25)
            Dvalues = self.read_detector_mW()
            Dmaxvalues = sum(Dvalues)

            
            self.horizontal_power.append(horizontal_ports_percent)
            self.diagonal_power.append(diagonal_ports_percent)

            # Standard cost based on ideal values 1 + 1 +
            standard_cost = (
                (((Hvalues[0]/(Hvalues[0]+Hvalues[1])) - 1) ** 2 + ((Hvalues[1]/(Hvalues[0]+Hvalues[1]))) ** 2 +
                ((Hvalues[2]/(Hvalues[2]+Hvalues[3])) - 0.5) ** 2 + ((Hvalues[3]/(Hvalues[2]+Hvalues[3])) - 0.5) ** 2 +
                ((Dvalues[0]/(Dvalues[0]+Dvalues[1])) - 0.5) ** 2 + ((Dvalues[1]/(Dvalues[0]+Dvalues[1])) - 0.5) ** 2 +
                ((Dvalues[2]/(Dvalues[2]+Dvalues[3])) - 1) ** 2 + ((Dvalues[3]/(Dvalues[2]+Dvalues[3]))) ** 2)/5
            )
            # standard_cost = (
            #     ((((Hvalues[0]/(Hvalues[0]+Hvalues[1])) - 1) + ((Hvalues[1]/(Hvalues[0]+Hvalues[1])))) ** 2 +
            #    ( ((Hvalues[2]/(Hvalues[2]+Hvalues[3])) - 0.5)  + ((Hvalues[3]/(Hvalues[2]+Hvalues[3])) - 0.5)) ** 2 +
            #     (((Dvalues[0]/(Dvalues[0]+Dvalues[1])) - 0.5)  + ((Dvalues[1]/(Dvalues[0]+Dvalues[1])) - 0.5)) ** 2 +
            #     (((Dvalues[2]/(Dvalues[2]+Dvalues[3])) - 1)  + ((Dvalues[3]/(Dvalues[2]+Dvalues[3])))) ** 2)/3
            # )
            # Penalty to ensure ports 3 and 4 are similar and ports 2 and 3 are not
            # penalty_cost = (
            #     (horizontal_ports_percent[2] - horizontal_ports_percent[3]) ** 2 +  # Encourage ports 3 and 4 to be similar
            #     (horizontal_ports_percent[1] - horizontal_ports_percent[2]) ** 2 * 0.5  # Discourage ports 2 and 3 from being too similar
            # )

            
            ################################
            #Cost function with percentages#
            ################################
            total_cost = standard_cost #+ penalty_cost
            self.cost.append(total_cost)
            ################################
            # Cost function with Fidelity  #
            ################################      


            supply.write("OUTP CH1,ON")
            sleep(.25)
            Hvalues = self.read_detector_mW()
            Hmaxvalues = sum(Hvalues)
            print("HvalueH: ", Hvalues[0]/Hmaxvalues)
            print("HvalueD: ", Hvalues[2]/Hmaxvalues)

            supply.write("OUTP CH1,OFF")
            sleep(.25)
            Dvalues = self.read_detector_mW()
            Dmaxvalues = sum(Dvalues)
            print("DvalueH: ", Dvalues[0]/Dmaxvalues)
            # theta, psi, lambdas = calculate_HBasis(Hvalues[0]/Hmaxvalues, Hvalues[2]/Hmaxvalues, Dvalues[1]/Dmaxvalues)
            theta, psi, lambdas = calculate_HBasis_new((Hvalues[0]/(Hvalues[0]+Hvalues[1])), (Hvalues[2]/(Hvalues[2]+Hvalues[3])), (Dvalues[0]/(Dvalues[0]+Dvalues[1]))) # p_H in H direction, p_D in H direction, p_H in D direction
            Fid  = fidelity(theta, psi, lambdas)  
            self.fidelity.append(Fid)
            self.all_voltages.append(voltages)

            print(f"Iteration: {self.iteration}, Voltages: {voltages}, Cost: {total_cost:.5f}")
            return total_cost #two cost functions???
    def custom_optimization(self, initial_voltages):
        current_voltages = np.array(initial_voltages, dtype=np.float64)
        current_cost = self.compute_cost_record(current_voltages)
    
        # Optimization loop
        for i in range(self.max_iterations):
            # if i < self.max_iterations // 2:
            #     step_size = self.initial_step_size
            # If no improvement is made, introduce stochastic perturbation
            if  self.cost and current_cost >= .1 and i >= 2 and abs(self.cost[-1] - self.cost[-2]) < 1e-3: #changed from 1e -5
                current_voltages += np.random.uniform(-step_size, step_size, size=current_voltages.shape)
                current_voltages = np.array([wrap_voltage(v) for v in current_voltages])
                print("Stochastic")
            else:

                ## add stepsize values sequentially from top to bottom (smallest to largest)
                if current_cost < .035:
                    print("smallest stepsize")
                    step_size = 20 * 3*np.log(( (10*current_cost) +(1 - 1E-8)))
                elif current_cost < .038:#changed from .038
                    print("smaller stepsize")
                    step_size = 25 * 3*np.log(( (10*current_cost) +(1 - 1E-8)))
                elif current_cost < .04: #changed from .04
                    print("smaller stepsize")
                    step_size = 50 * 3*np.log(( (10*current_cost) +(1 - 1E-8)))
                    # step_size = 1
                else:
                    step_size = 60 * 3*np.log(( (10*current_cost) +(1 - 1E-8)))
                    print("Larger stepsize")
                print("Step Size: " + str(step_size))
                for j in range(len(current_voltages)):
                    perturbed_voltages = np.copy(current_voltages)
                    perturbed_voltages[j] += step_size

                    perturbed_voltages = np.array([wrap_voltage(v) for v in perturbed_voltages])

                    perturbed_cost = self.compute_cost(perturbed_voltages)

                    if perturbed_cost < current_cost:
                        current_cost = perturbed_cost
                        current_voltages = perturbed_voltages
                

            
            current_cost = self.compute_cost_record(current_voltages)
            print(f"Current best cost: {current_cost:.5f}") #fix this it prints local best not global
        print(f"Final Voltages after Custom Optimization: {current_voltages}, Final Cost: {current_cost:.5f}, Fidelity: {self.fidelity[-1]:.5f}")

        # # Repeat final voltages for consistency check
        # for _ in range(self.repeat_iterations):
        #     self.compute_cost(current_voltages)
        #     print(f"Repeating Final Voltages: {current_voltages}, Cost: {self.cost[-1]:.5f}, Fidelity: {self.fidelity[-1]:}")

        return current_voltages

def plot_data(horizontal_power, diagonal_power, cost, fidelity, voltages,name):
    
    with open(name, 'a', newline='') as file:
        writer = csv.writer(file)
        for i in range(len(horizontal_power[0])):
            row = [
                horizontal_power[0][i], horizontal_power[1][i], horizontal_power[2][i], horizontal_power[3][i],
                diagonal_power[0][i], diagonal_power[1][i], diagonal_power[2][i], diagonal_power[3][i],
                cost[i], fidelity[i],
                voltages[0][i], voltages[1][i], voltages[2][i], voltages[3][i]
            ]
            writer.writerow(row)
    # Convert lists to DataFrames for easier plotting
    data_H = pd.DataFrame(horizontal_power, columns=['Port 1', 'Port 2', 'Port 3', 'Port 4'])
    data_D = pd.DataFrame(diagonal_power, columns=['Port 1', 'Port 2', 'Port 3', 'Port 4'])
    cost_df = pd.Series(cost, name='Cost')
    fidelity_df = pd.Series(fidelity, name='fidelity')
    voltages_df = pd.DataFrame(voltages, columns=['V1', 'V2', 'V3', 'V4'])

    # Plot horizontal power
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

    # Plot diagonal power
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

    # # Plot cost
    # plt.figure(figsize=(10, 5))
    # plt.plot(cost_df.index, cost_df, label='Cost')
    # plt.xlabel('Iteration')
    # plt.ylabel('Cost')
    # plt.title('Cost Over Time')
    # plt.ylim(0, 1)
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    # # Plot fidelity
    # plt.figure(figsize=(10, 5))
    # plt.plot(fidelity_df.index, fidelity_df, label='fidelity')
    # plt.xlabel('Iteration')
    # plt.ylabel('Fidelity')
    # plt.title('Fidelity Over Iterations')
    # plt.ylim(0, 1)
    # plt.legend()
    # plt.grid(True)
    # plt.show()

        # Plot cost
    plt.figure(figsize=(10, 5))
    plt.plot(cost_df.index, cost_df, label='Cost')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title('Cost Over Time')
    plt.yscale("log")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot fidelity
    plt.figure(figsize=(10, 5))
    plt.plot(fidelity_df.index, fidelity_df, label='fidelity')
    plt.xlabel('Iteration')
    plt.ylabel('Fidelity')
    plt.title('Fidelity Over Iterations')
    plt.yscale("log")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot voltages
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


    pass    
    

def main():
    DETECTOR_RESOURCE_STR = "TCPIP0::10.0.3.17::5000::SOCKET"
    EPC_FEEDBACK_RESOURCE_STR = "ASRL3::INSTR"
    #INITIAL_VOLTAGES = [-550, -650, 250, 350]
    
    #INITIAL_VOLTAGES = [5000, 5000, 5000,5000]

    INITIAL_VOLTAGES = [random.randint(-5000,5000), random.randint(-5000,5000), random.randint(-5000,5000) , random.randint(-5000,5000)]
    #INITIAL_VOLTAGES = [566.48054275, 425.28570077, 425.28570077, 425.28570077]
    #INITIAL_VOLTAGES = [-2257,   1572.42724783, -2510.48196718,  2575.15759365]
    print(INITIAL_VOLTAGES)
    MAX_ITERATIONS = 15 #Value to change for number of iterations
    REPEAT_ITERATIONS = 10  # Repeat final voltages for consistency check

    resource_manager = visa.ResourceManager()
    detector_session = resource_manager.open_resource(DETECTOR_RESOURCE_STR)
    detector_session.read_termination = "\n"
    detector_session.write_termination = "\n"
    epc_feedback_session = resource_manager.open_resource(EPC_FEEDBACK_RESOURCE_STR)
    epc_feedback_session.read_termination = "\r\n"
    epc_feedback_session.write_termination = "\r\n"
    epc_feedback_session.query_termination = "\r\n"

    epc_optimizer = CustomEPCOptimizer(detector_session, epc_feedback_session, max_iterations=MAX_ITERATIONS, repeat_iterations=REPEAT_ITERATIONS)
    
    # Start the custom optimization process
    optimized_voltages = epc_optimizer.custom_optimization(INITIAL_VOLTAGES)
    # Format the current datetime as a string
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    # Define the file path with the formatted timestamp
    name = fr'C:\Users\Alice1\Documents\RandomScripts\Opto_data\Optimization_Values_{timestamp}.csv'

    # Header for the CSV file
    header = ["H1", "H2", "H3", "H4", "D1", "D2", "D3", "D4", "Cost", "Fidelity", "V1", "V2", "V3", "V4"]

    # Write header to CSV file
    with open(name, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        

    # Plot data after optimization is complete
    plot_data(epc_optimizer.horizontal_power, epc_optimizer.diagonal_power, epc_optimizer.cost, epc_optimizer.fidelity, epc_optimizer.all_voltages, name)

    supply.close()

if __name__ == "__main__":
    main()
