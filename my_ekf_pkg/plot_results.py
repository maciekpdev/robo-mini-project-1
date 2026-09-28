import numpy as np
import matplotlib.pyplot as plt
import csv
import os

#Define file path for the evaluation CSV
file_path = os.path.expanduser("~/ros2_ws/ekf_evaluation.csv")

#Check if file exists before attempting to load it
if not os.path.exists(file_path):
    print(f"File {file_path} does not exist. Please run the EKF node first to generate the evaluation data.")
    exit()

#Load numerical data from CSV file, skipping the header row
data = np.loadtxt(file_path, delimiter=',', skiprows=1)

#Extract time and error metrics from the loaded data
t = data[:, 0] - data[0, 0] #Normalize time to start from 0
error_dist = data[:, 1]
err_x = data[:, 2]
err_y = data[:, 3]

#Compute evaluation metrics: RMSE, mean error, and max error
rmse = np.sqrt(np.mean(error_dist**2)) #Root mean square error
mean_err = np.mean(error_dist) #Average distance error
max_err = np.max(error_dist) #Maximum recorded position error

#Print evaluation metrics to the console
print("=== EVALUATION METRICS (Task 1a) ===")
print(f"Root Mean Square Error (RMSE): {rmse:.4f} m")
print(f"Mean Position Error:           {mean_err:.4f} m")
print(f"Max Position Error:            {max_err:.4f} m")

# Plot 
plt.figure(figsize=(10, 5))
plt.plot(t, error_dist, label='Euclidean Position Error $e_k$', color='b', linewidth=1.5)
plt.axhline(y=rmse, color='r', linestyle='--', label=f'RMSE ({rmse:.3f} m)')

plt.title('EKF Positioning Error over Time (1 Hz Ground Truth Update)')
plt.xlabel('Time [s]')
plt.ylabel('Error [m]')
plt.grid(True)
plt.legend()
plt.tight_layout()

#Save the plot as a PNG file in the user's home directory
output_png = os.path.expanduser("~/ros2_ws/ekf_error_plot.png")
plt.savefig(output_png, dpi=300)
print(f"diagram saved successfully: {output_png}")


plt.show()