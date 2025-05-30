import matplotlib.pyplot as plt
import pandas as pd

# Read data from file
data = pd.read_csv("matrix_multiply_times.txt")

# Extract columns
sizes = data["Size"]
serial_times = data["SerialTime"]
parallel_times = data["ParallelTime"]

# Plot
plt.figure(figsize=(10, 6))
plt.plot(sizes, serial_times, marker='o', label='Serial', color='blue')
plt.plot(sizes, parallel_times, marker='s', label='Parallel (OpenMP)', color='red')
plt.xlabel('Matrix Size (N x N)')
plt.ylabel('Execution Time (seconds)')
plt.title('Matrix Multiplication: Serial vs Parallel (OpenMP)')
plt.grid(True)
plt.legend()
plt.savefig('matrix_multiply_performance.png')
plt.close()