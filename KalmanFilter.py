import numpy as np

class KalmanFilter:
    def __init__(self, dt, u, std_acc, std_meas):
        self.A = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.B = np.array([[0.5 * dt ** 2], [0.5 * dt ** 2], [dt], [dt]])
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        self.Q = np.eye(self.A.shape[0]) * std_acc**2
        self.R = np.eye(self.H.shape[0]) * std_meas**2
        self.P = np.eye(self.A.shape[0])
        self.x = np.zeros((self.A.shape[1], 1))
        self.u = u

    def predict(self):
        self.x = np.dot(self.A, self.x) + self.B * self.u
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        return self.x

    def update(self, z):
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, (z - np.dot(self.H, self.x)))
        I = np.eye(self.A.shape[1])
        self.P = np.dot(I - np.dot(K, self.H), self.P)
#