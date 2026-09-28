import numpy as np

class EKF:
    def __init__(self, x0, P0, Q, R):
        #Current state estimate
        self.x = np.array(x0, dtype = float)

        #Current state covariance estimate
        self.P = np.array(P0, dtype = float)

        #Process noise covariance and measurement noise covariance
        self.Q = np.array(Q, dtype = float)
        self.R = np.array(R, dtype = float)

    #Normalizes every angle to be in the range [-pi, pi)
    def normalize_angle(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    #Run the prediction step of the EKF using the last received velocity and angular velocity
    def predict(self, v, omega, dt):
        theta = self.x[2]

        #Motion model equations for a differential drive robot
        dx = v * dt * np.cos(theta)
        dy = v * dt * np.sin(theta)
        dtheta = omega * dt

        self.x[0] += dx
        self.x[1] += dy
        self.x[2] = self.normalize_angle(theta + dtheta)

        #Compute the Jacobian of the motion model with respect to the state
        F = np.array([
            [1.0, 0.0, -v * dt * np.sin(theta)],
            [0.0, 1.0, v * dt * np.cos(theta)],
            [0.0, 0.0, 1.0]
        ])

        #Propagate the uncertainty through the motion model and add process noise Q
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z):
        #Links the state to the measurement
        H = np.eye(3)

        #Innovation: Difference between measurement and prediction
        y = z - self.x
        y[2] = self.normalize_angle(y[2])

        #Innovation covariance: Update the uncertainty in the innovation
        S = H @ self.P @ H.T + self.R

        #Compute the Kalman gain
        #Close to 1: trust the measurement more, Close to 0: trust the prediction more
        K = self.P @ H.T @ np.linalg.inv(S)

        #Correct the state estimate with the measurement
        self.x += K @ y
        self.x[2] = self.normalize_angle(self.x[2])

        #Update the state covariance estimate
        I = np.eye(3)
        self.P = (I - K @ H) @ self.P

