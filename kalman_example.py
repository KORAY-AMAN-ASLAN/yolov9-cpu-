import numpy as np

class KalmanFilter:
    def __init__(self, F=None, Q=None, H=None, R=None, P=None, x=None):
        """
        Initialize the Kalman Filter with matrices defining the system.
        :param F: State transition matrix
        :param Q: Process noise covariance
        :param H: Observation matrix
        :param R: Measurement noise covariance
        :param P: Estimate error covariance
        :param x: Initial state estimate
        """
        self.F = F if F is not None else np.eye(2)  # Default to identity if not provided
        self.Q = Q if Q is not None else np.eye(2)
        self.H = H if H is not None else np.eye(2)
        self.R = R if R is not None else np.eye(1)
        self.P = P if P is not None else np.eye(2)
        self.x = x if x is not None else np.zeros((2, 1))  # State vector [position, velocity]

    def predict(self):
        # Predict the next state
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(self.F, np.dot(self.P, self.F.T)) + self.Q

    def update(self, z):
        # Update the state with measurement z
        Y = z - np.dot(self.H, self.x)  # Measurement residual
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R  # Residual covariance
        K = np.dot(self.P, np.dot(self.H.T, np.linalg.inv(S)))  # Kalman gain
        self.x = self.x + np.dot(K, Y)
        I = np.eye(self.H.shape[1])
        self.P = np.dot(I - np.dot(K, self.H), self.P)

# Example: Vehicle Tracking

# Time step
dt = 1.0

# State transition matrix
F = np.array([[1, dt],
              [0, 1]])

# Measurement matrix
H = np.array([[1, 0]])

# Process noise covariance
Q = np.array([[1, 0],
              [0, 1]])

# Measurement noise covariance
R = np.array([[10]])  # High value indicates high noise

# Initial estimate error covariance
P = np.array([[1000, 0],
              [0, 1000]])

# Initial state
x = np.array([[0],  # position
              [0]])  # velocity

kf = KalmanFilter(F=F, Q=Q, H=H, R=R, P=P, x=x)

# Simulated GPS measurements (position) with noise
measurements = [10, 20, 30, 40, 50]  # Actual positions: 10, 20, 30, 40, 50

for z in measurements:
    kf.predict()
    kf.update(np.array([[z]]))

    print(f"Updated state (position, velocity): {kf.x.flatten()}")

