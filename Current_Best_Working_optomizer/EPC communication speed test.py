import pyvisa as visa
import numpy as np
from time import sleep, time
import csv
import datetime  # Import datetime module

# Function to wrap voltage
def wrap_voltage(value, low=-5000, high=5000):
    if low <= value <= high:
        return value
    return (-np.abs(value) % (high - low)) * np.sign(value)

# Test EPC Speed
def test_epc_speed(epc_feedback_session, iterations=100, max_voltage=5000):
    """
    Tests EPC speed by toggling voltages one at a time and then all at once.

    Args:
        epc_feedback_session: The EPC session object.
        iterations: The number of cycles (1, 2, 3, 4, all).
        max_voltage: The range of voltages (-max_voltage to +max_voltage).

    Returns:
        A dictionary containing timings and results for single and simultaneous switches.
    """
    single_channel_times = []
    simultaneous_times = []

    # Start the timer for total execution
    total_start_time = time()

    for i in range(iterations):
        # Test single-channel switching (1, 2, 3, 4)
        for channel in range(1, 5):
            voltage = np.random.randint(-max_voltage, max_voltage)
            start_time = time()
            epc_feedback_session.write(f"V{channel},{int(wrap_voltage(voltage))}")
            sleep(0.01)  # Allow time for communication
            end_time = time()
            single_channel_times.append(end_time - start_time)

        # Test simultaneous switching (all four channels)
        voltages = [
            np.random.randint(-max_voltage, max_voltage),
            np.random.randint(-max_voltage, max_voltage),
            np.random.randint(-max_voltage, max_voltage),
            np.random.randint(-max_voltage, max_voltage),
        ]
        start_time = time()
        for channel, voltage in enumerate(voltages, 1):
            epc_feedback_session.write(f"V{channel},{int(wrap_voltage(voltage))}")
        sleep(0.01)  # Allow time for communication
        end_time = time()
        simultaneous_times.append(end_time - start_time)

    # End the timer for total execution
    total_end_time = time()
    total_time = total_end_time - total_start_time

    # Calculate averages
    avg_single_time = sum(single_channel_times) / len(single_channel_times)
    avg_simultaneous_time = sum(simultaneous_times) / len(simultaneous_times)

    results = {
        "single_channel_times": single_channel_times,
        "simultaneous_times": simultaneous_times,
        "avg_single_time": avg_single_time,
        "avg_simultaneous_time": avg_simultaneous_time,
        "total_time": total_time,
    }
    return results

def main():
    EPC_FEEDBACK_RESOURCE_STR = "ASRL3::INSTR"

    resource_manager = visa.ResourceManager()
    epc_feedback_session = resource_manager.open_resource(EPC_FEEDBACK_RESOURCE_STR)
    epc_feedback_session.read_termination = "\r\n"
    epc_feedback_session.write_termination = "\r\n"

    print("Testing EPC speed...")

    # Run the test
    iterations = 100
    results = test_epc_speed(epc_feedback_session, iterations=iterations)

    # Display results
    print(f"Average time for single-channel switching: {results['avg_single_time']:.6f} seconds")
    print(f"Average time for simultaneous switching: {results['avg_simultaneous_time']:.6f} seconds")
    print(f"Total time for {iterations} cycles: {results['total_time']:.6f} seconds")

    # Save results to CSV
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    filename = f"EPC_Speed_Test_Results_{timestamp}.csv"
    with open(filename, "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["Iteration", "Single-Channel Time (s)", "Simultaneous Time (s)"])
        for i in range(iterations):
            csvwriter.writerow([
                i + 1,
                results["single_channel_times"][i * 4:(i + 1) * 4],
                results["simultaneous_times"][i],
            ])
        csvwriter.writerow([])
        csvwriter.writerow(["Average Single-Channel Time", results["avg_single_time"]])
        csvwriter.writerow(["Average Simultaneous Time", results["avg_simultaneous_time"]])
        csvwriter.writerow(["Total Time for All Iterations", results["total_time"]])

    print(f"Results saved to {filename}")

    epc_feedback_session.close()

if __name__ == "__main__":
    main()
