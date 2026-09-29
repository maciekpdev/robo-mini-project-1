"""
Plot the EKF error at the 1 Hz ground-truth instants from the CSV.

Usage: ``python3 plot_results.py [csv_path]`` (default
~/ros2_ws/ekf_evaluation.csv, written by ekf_node on shutdown or by
run_offline --csv). The errors are those of the prediction just BEFORE each
correction, i.e. the drift accumulated over one second.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np


def main(argv=None):
    """Load the CSV, print metrics and save/show the error plot."""
    argv = sys.argv[1:] if argv is None else argv
    file_path = os.path.expanduser(argv[0] if argv else '~/ros2_ws/ekf_evaluation.csv')

    # Check if file exists before attempting to load it
    if not os.path.exists(file_path):
        print(f'File {file_path} does not exist. '
              'Please run the EKF node first to generate the evaluation data.')
        sys.exit(1)

    # Load numerical data from CSV file, skipping the header row
    data = np.atleast_2d(np.loadtxt(file_path, delimiter=',', skiprows=1))

    # Extract time and error metrics (columns: time,error_dist,err_x,err_y,...)
    t = data[:, 0] - data[0, 0]  # Normalize time to start from 0
    error_dist = data[:, 1]

    # Compute evaluation metrics: RMSE, mean error, and max error
    rmse = np.sqrt(np.mean(error_dist ** 2))  # Root mean square error
    mean_err = np.mean(error_dist)  # Average distance error
    max_err = np.max(error_dist)  # Maximum recorded position error

    print('=== EVALUATION METRICS (Task 1a, error before each 1 Hz correction) ===')
    print(f'Root Mean Square Error (RMSE): {rmse:.4f} m')
    print(f'Mean Position Error:           {mean_err:.4f} m')
    print(f'Max Position Error:            {max_err:.4f} m')

    plt.figure(figsize=(10, 5))
    plt.plot(t, error_dist, label='Euclidean position error $e_k$', color='#2a78d6',
             linewidth=1.5)
    plt.axhline(y=rmse, color='#2b2b2b', linestyle='--', label=f'RMSE ({rmse:.3f} m)')
    if data.shape[1] > 5:
        # Filter's own 2-sigma position bound (column sigma_pos)
        plt.plot(t, 2.0 * data[:, 5], color='#2a78d6', linestyle=':',
                 label='2 sigma_pos (filter)')

    plt.title('EKF position error before each 1 Hz ground-truth update')
    plt.xlabel('Time [s]')
    plt.ylabel('Error [m]')
    plt.grid(True, color='#d9d9d9')
    plt.legend()
    plt.tight_layout()

    # Save the plot next to the CSV file
    output_png = os.path.join(os.path.dirname(file_path), 'ekf_error_plot.png')
    plt.savefig(output_png, dpi=300)
    print(f'Plot saved to {output_png}')
    plt.show()


if __name__ == '__main__':
    main()
