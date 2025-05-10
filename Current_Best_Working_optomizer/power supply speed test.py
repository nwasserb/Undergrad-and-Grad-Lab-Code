import vxi11
import time
import csv
import datetime

def test_power_supply_toggle_batch(supply, iterations=100):
    """
    Tests the speed of toggling the power supply on and off using batch commands.

    Args:
        supply: The power supply object (vxi11.Instrument).
        iterations: The number of on/off toggle cycles to test.

    Returns:
        A list of toggle times in seconds.
    """
    toggle_times = []

    # Start the timer for total execution
    total_start_time = time.time()

    # Prepare batch commands for toggling ON and OFF
    batch_commands = "\n".join(["OUTP CH1,ON", "OUTP CH1,OFF"])

    for i in range(iterations):
        start_time = time.time()
        # Send both commands in a single write
        supply.write(batch_commands)
        end_time = time.time()
        
        # Record the elapsed time for the toggle operation
        toggle_times.append(end_time - start_time)
        
        print(f"Iteration {i+1}/{iterations}: Toggle time = {toggle_times[-1]:.6f} seconds")

    # End the timer for total execution
    total_end_time = time.time()
    total_time = total_end_time - total_start_time

    return toggle_times, total_time

def main():
    # Connect to the power supply
    supply = vxi11.Instrument("10.0.3.7")
    
    print("Testing power supply toggle speed with batch commands...")

    # Run the batch toggle test
    iterations = 100
    toggle_times, total_time = test_power_supply_toggle_batch(supply, iterations=iterations)
    
    # Calculate the average toggle time
    avg_time = sum(toggle_times) / len(toggle_times)
    print(f"Average toggle time over {iterations} iterations: {avg_time:.6f} seconds")
    print(f"Total time for {iterations} iterations: {total_time:.6f} seconds")
    
    # Save results to CSV
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    filename = f"Power_Supply_Batch_Toggle_Test_{timestamp}.csv"
    with open(filename, "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["Iteration", "Toggle Time (s)"])
        for i, toggle_time in enumerate(toggle_times, 1):
            csvwriter.writerow([i, toggle_time])
        csvwriter.writerow([])
        csvwriter.writerow(["Average Toggle Time", avg_time])
        csvwriter.writerow(["Total Time for Iterations", total_time])
    
    print(f"Results saved to {filename}")

    # Close the connection to the power supply
    supply.close()

if __name__ == "__main__":
    main()
# import vxi11
# import time
# import csv
# import datetime

# def test_power_supply_toggle_batch_iterations(supply, iterations=100):
#     """
#     Tests the speed of toggling the power supply on and off using batch commands for all iterations.

#     Args:
#         supply: The power supply object (vxi11.Instrument).
#         iterations: The number of on/off toggle cycles to test.

#     Returns:
#         Total time taken and average toggle time.
#     """
#     # Start the timer for total execution
#     total_start_time = time.time()

#     # Build a single batch command for all iterations
#     batch_iterations = "\n".join(["OUTP CH1,ON\nOUTP CH1,OFF"] * iterations)

#     # Send all commands in a single write operation
#     supply.write(batch_iterations)

#     # End the timer for total execution
#     total_end_time = time.time()
#     total_time = total_end_time - total_start_time

#     # Calculate average time per toggle
#     avg_time = total_time / iterations

#     return total_time, avg_time

# def main():
#     # Connect to the power supply
#     supply = vxi11.Instrument("10.0.3.7")
    
#     print("Testing power supply toggle speed with full batch iterations...")

#     # Run the batch toggle test
#     iterations = 100
#     total_time, avg_time = test_power_supply_toggle_batch_iterations(supply, iterations=iterations)
    
#     print(f"Average toggle time over {iterations} iterations: {avg_time:.6f} seconds")
#     print(f"Total time for {iterations} iterations: {total_time:.6f} seconds")
    
#     # Save results to CSV
#     timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
#     filename = f"Power_Supply_Full_Batch_Toggle_Test_{timestamp}.csv"
#     with open(filename, "w", newline="") as csvfile:
#         csvwriter = csv.writer(csvfile)
#         csvwriter.writerow(["Total Iterations", "Total Time (s)", "Average Time per Toggle (s)"])
#         csvwriter.writerow([iterations, total_time, avg_time])
    
#     print(f"Results saved to {filename}")

#     # Close the connection to the power supply
#     supply.close()

# if __name__ == "__main__":
#     main()
