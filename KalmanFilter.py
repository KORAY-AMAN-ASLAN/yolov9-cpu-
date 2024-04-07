import numpy as np


class KalmanFilter:
    """
    Initializes the Kalman Filter for a simple dynamic system with position and velocity.

    The Kalman Filter estimates the state of a linear dynamic system from a series of noisy measurements.

    Parameters:
    - dt (float): Time step between state updates, affects the state transition model.
    - u (np.array): External motion, control input to the system, typically acceleration.
    - std_acc (float): Standard deviation of the acceleration (process noise), modeling the process uncertainty.
    - std_meas (float): Standard deviation of the measurement noise, representing sensor uncertainty.

    Attributes:
    - A (np.array): State transition matrix defining how the state evolves from one time step to the next without any control input.
    - B (np.array): Control input model, mapping the control input vector `u` into the state space.
    - H (np.array): Observation model, mapping the state vector into the measurement space.
    - Q (np.array): Process noise covariance matrix, representing the uncertainty in the system dynamics.
    - R (np.array): Measurement noise covariance matrix, representing the uncertainty in sensor measurements.
    - P (np.array): Estimate error covariance matrix, representing the uncertainty of the state estimate.
    - x (np.array): State vector, representing the estimated state of the system. Initially set to zeros.
    - u (np.array): Control input vector, typically acceleration, affecting the state at each time step.
    """

    def __init__(self, dt, u, std_acc, std_meas):
        self.A = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])  # State transition matrix
        self.B = np.array([[0.5 * dt ** 2],
                           [0.5 * dt ** 2],
                           [dt],
                           [dt]])  # Control input model
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])  # Observation model
        self.Q = np.eye(self.A.shape[0]) * std_acc ** 2  # Process noise covariance
        self.R = np.eye(self.H.shape[0]) * std_meas ** 2  # Measurement noise covariance
        self.P = np.eye(self.A.shape[0])  # Initial estimate error covariance
        self.x = np.zeros((self.A.shape[1], 1))  # Initial state vector
        self.u = u  # Control input vector

    def predict(self):
        """
        Predicts the next state of the system using the state transition model and updates the estimate error covariance.

        The prediction incorporates the control input through the control input model.

        Returns:
        - The predicted state vector (np.array).
        """
        # Predict the next state
        self.x = np.dot(self.A, self.x) + np.dot(self.B, self.u)
        # Update the error covariance matrix
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        return self.x

    def update(self, z):
        """
        Updates the state estimate based on a new measurement.

        This method computes the Kalman Gain, then uses the measurement to correct the predicted state,
        and finally updates the estimate error covariance matrix.

        Parameters:
        - z (np.array): The measurement vector.

        Returns:
        - None
        """
        # Compute the Kalman Gain
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        # Update the state estimate using the measurement
        self.x = self.x + np.dot(K, (z - np.dot(self.H, self.x)))
        # Update the error covariance matrix
        I = np.eye(self.A.shape[1])
        self.P = np.dot(I - np.dot(K, self.H), self.P)
