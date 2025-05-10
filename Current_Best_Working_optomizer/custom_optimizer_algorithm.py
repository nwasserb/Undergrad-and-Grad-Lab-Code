# Given that traditional gradient descent methods may not perform well with noisy, non-linear data,
# try to implement a Simulated Annealing optimizer. 
# This method is robust to noise and can escape local minima, making it suitable for this problem.
# custom_optimizer.py

# Simulated Annealing is a probabilistic technique for approximating the global optimum of a given function.
# The algorithm explores the solution space by moving to neighboring solutions.
# It occasionally accepts worse solutions to escape local minima, 
# with the probability of accepting worse solutions decreasing over time.
# The "temperature" parameter controls this probability, gradually decreasing according to the "cooling rate".

##############################################################################################################
# To use this optimizer:
# Cost Function: Define a cost_function that uses existing compute_cost_record method.
# Parameters: Adjust max_iterations, initial_temperature, and cooling_rate as needed.
# Output: The optimizer returns the best voltages and cost found.

# from custom_optimizer import CustomOptimizer

# class CustomEPCOptimizer:
#     # ... (existing methods)

    # def custom_optimization(self, initial_voltages):
    #     def cost_function(voltages):
    #         return self.compute_cost_record(voltages)

        # optimizer = MATT(
        #     cost_function=cost_function,
        #     initial_voltages=initial_voltages,
        #     max_iterations=self.max_iterations,
        #     initial_temperature=1000, # adjust this, higher is slower but explores more of solution space
        #     cooling_rate=0.95 # the rate at which the probability of a guess decreases, closer to 1 is slower rate
        #     adaptive=True
        #     min_temperature = 1e-3
        # )

        # best_voltages, best_cost = optimizer.optimize()
        # print(f"Final Voltages after Custom Optimization: {best_voltages},
        #        Final Cost: {best_cost:.5f},
        #          Fidelity: {self.fidelity[-1]:.5f}")

        # return best_voltages

##############################################################################################################

# def main():
#     # ... (existing code)
    
#     # Example initial voltages
#     INITIAL_VOLTAGES = [random.randint(-5000, 5000) for _ in range(4)]
#     MAX_ITERATIONS = 100  # Adjust as needed
#     REPEAT_ITERATIONS = 10  # As before

#     # ... (resource setup code)

#     epc_optimizer = CustomEPCOptimizer(detector_session,
#                                         epc_feedback_session,
#                                           max_iterations=MAX_ITERATIONS,
#                                             repeat_iterations=REPEAT_ITERATIONS)
    
#     # Start the custom optimization process
#     optimized_voltages = epc_optimizer.custom_optimization(INITIAL_VOLTAGES)
    
#     # ... (rest of the code)


##############################################################################################################

# initial_temperature: Higher values allow the algorithm to explore more of the solution space initially.
# cooling_rate: Values closer to 1 slow down the cooling process, allowing more exploration over time.
# max_iterations: Increasing this value allows the optimizer more time to find the global minimum but will take longer.

# The custom optimizer works as follows:

# Initialization:

# Starts with an initial set of voltages.
# Sets the initial temperature for the annealing process.
# Iteration Loop:

# Generates a new candidate solution by randomly perturbing the current voltages within a range dictated by the current temperature.
# Evaluates the cost of the new candidate solution.
# Decides whether to accept the new solution:
# If the new cost is lower, it always accepts.
# If the new cost is higher, it may still accept based on a probability that decreases with higher cost and lower temperature.
# Updates the best solution found so far if the current solution is better.
# Decreases the temperature according to the cooling rate.
# Termination:

# After the maximum number of iterations, it returns the best voltages found.


##############################################################################################################

# Algorithm Name: Modified Annealing Tuning Technique (MATT)

##############################################################################################################

# Version 1: Fixed temperature and cooling rate
# Version 2: Adaptive temperature rate and re-annealing mechanism

import numpy as np

class MATT:
    """
    MATTOptimizer: Modified Annealing Tuning Technique.

    This optimizer uses a customized Simulated Annealing algorithm tailored for optimizing EPC voltages.
    It includes adaptive temperature control, re-annealing mechanisms, and noise handling to adapt
    to disturbances and noise in the cost function.
    """

    def __init__(self, cost_function, initial_voltages, max_iterations=100, initial_temperature=1000,
                 cooling_rate=0.95, min_temperature=1e-3, adaptive=True):
        """
        Initialize the MATTOptimizer.

        Parameters:
            cost_function: Callable that accepts voltages and returns a cost.
            initial_voltages: Initial guess for the voltages.
            max_iterations: Maximum number of iterations.
            initial_temperature: Starting temperature for the annealing process.
            cooling_rate: Rate at which the temperature decreases (between 0 and 1).
            min_temperature: Minimum temperature to prevent it from becoming too low.
            adaptive: Whether to use adaptive temperature control.
        """
        self.cost_function = cost_function
        self.initial_voltages = np.array(initial_voltages, dtype=np.float64)
        self.max_iterations = max_iterations
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.min_temperature = min_temperature
        self.adaptive = adaptive

        # Variables for adaptive control
        self.acceptance_history = []
        self.temperature = initial_temperature

    def optimize(self):
        """
        Run the optimization process.

        Returns:
            best_voltages: The best voltages found during optimization.
            best_cost: The cost corresponding to the best voltages.
        """
        # Initialize current solution
        current_voltages = self.initial_voltages.copy()
        current_cost = self.cost_function(current_voltages)
        best_voltages = current_voltages.copy()
        best_cost = current_cost

        # Initialize temperature
        temperature = self.initial_temperature

        # Stagnation counter for re-annealing
        stagnation_counter = 0

        for iteration in range(self.max_iterations):
            # Generate random perturbations within a range based on current temperature
            perturbations = np.random.uniform(-temperature, temperature, size=current_voltages.shape)
            candidate_voltages = current_voltages + perturbations

            # Ensure voltages are within allowed range [-5000, 5000]
            candidate_voltages = np.clip(candidate_voltages, -5000, 5000)

            # Evaluate the cost function
            # Implement noise handling by averaging multiple cost evaluations
            num_samples = 3  # Number of samples to average for noise reduction
            costs = []
            for _ in range(num_samples):
                cost = self.cost_function(candidate_voltages)
                costs.append(cost)
            candidate_cost = np.mean(costs)

            # Decide whether to accept the new solution
            delta_cost = candidate_cost - current_cost
            accept = False
            if delta_cost < 0:
                # If the candidate cost is better, accept it
                accept = True
            else:
                # If the candidate cost is worse, accept it with a probability
                probability = np.exp(-delta_cost / temperature)
                if probability > np.random.rand():
                    accept = True

            if accept:
                # Accept the candidate solution
                current_voltages = candidate_voltages
                current_cost = candidate_cost
                self.acceptance_history.append(1)  # Accepted move
                stagnation_counter = 0  # Reset stagnation counter

                # Update the best solution found so far
                if current_cost < best_cost:
                    best_voltages = current_voltages.copy()
                    best_cost = current_cost
            else:
                # Do not accept the candidate solution
                self.acceptance_history.append(0)  # Rejected move
                stagnation_counter += 1  # Increment stagnation counter

            # Adaptive temperature control
            if self.adaptive and iteration % 10 == 0 and iteration > 0:
                # Calculate acceptance ratio over the last 10 iterations
                acceptance_ratio = sum(self.acceptance_history[-10:]) / 10
                # Adjust temperature based on acceptance ratio
                if acceptance_ratio > 0.6:
                    # If acceptance ratio is high, decrease temperature faster
                    temperature *= 0.9
                elif acceptance_ratio < 0.4:
                    # If acceptance ratio is low, decrease temperature slower
                    temperature *= 1.1
                # Ensure temperature doesn't fall below minimum
                temperature = max(temperature, self.min_temperature)
            else:
                # Standard cooling schedule
                temperature *= self.cooling_rate
                temperature = max(temperature, self.min_temperature)

            # Re-annealing condition
            if stagnation_counter > 20:
                # If no improvement for 20 iterations, reset temperature
                temperature = self.initial_temperature
                stagnation_counter = 0
                print(f"Re-annealing at iteration {iteration}, resetting temperature to {temperature}")

            # Optionally, print progress
            print(f"Iteration {iteration}: Best Cost = {best_cost:.5f}, Current Cost = {current_cost:.5f}, Temperature = {temperature:.2f}")

        return best_voltages, best_cost

    # Function to wrap voltage within the allowed range [-5000, 5000]
    def wrap_voltage(self, value, low=-5000, high=5000):
        if(value >= low and value <= high):
            return value
        else:
            if (value // (2 * high)) == 1:
                return low - ((-np.abs(value) % 5000) * np.sign(value))
            else:
                return (-np.abs(value) % 5000) * np.sign(value)
