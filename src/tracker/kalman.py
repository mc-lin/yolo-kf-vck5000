"""
Constant-velocity Kalman Filter for one bounding box.

State       x = [cx, cy, w, h, vx, vy]   center, size, center velocity (px/frame)
Measurement z = [cx, cy, w, h]           from a DPU detection box

Predict:  x = F x            P = F P F^T + Q
Update:   y = z - H x        S = H P H^T + R
          K = P H^T S^-1     x = x + K y        P = (I - K H) P
"""
import numpy as np


def box_to_z(box):
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], dtype=float)


def z_to_box(z):
    cx, cy, w, h = z[:4]
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


class KalmanBoxFilter:
    def __init__(self, box, dt=1.0,
                 meas_std=(5.0, 5.0, 10.0, 10.0),  # detector noise (px): cx, cy, w, h
                 accel_std=1.0,                     # how much velocity can change per frame
                 size_std=2.0,                      # how much box size drifts per frame
                 init_vel_std=10.0):                # unknown velocity at track start
        # F: motion model. New position = old position + velocity * dt; size and velocity carry over.
        self.F = np.eye(6)
        self.F[0, 4] = dt
        self.F[1, 5] = dt
        # H: which state entries the detector measures (cx, cy, w, h; not velocity).
        self.H = np.zeros((4, 6))
        self.H[:, :4] = np.eye(4)
        # R: measurement noise (how much we trust each detection).
        self.R = np.diag(np.square(meas_std))
        # Q: process noise (how much the real motion can deviate from constant velocity).
        #    Diagonal approximation of the constant-velocity / random-acceleration model.
        q = np.zeros(6)
        q[0:2] = (0.5 * accel_std * dt ** 2) ** 2
        q[2:4] = size_std ** 2
        q[4:6] = (accel_std * dt) ** 2
        self.Q = np.diag(q)
        # Initial state: position/size from the first detection, velocity unknown (0, high uncertainty).
        self.x = np.zeros(6)
        self.x[:4] = box_to_z(box)
        self.P = np.diag(np.concatenate([np.square(meas_std), [init_vel_std ** 2] * 2]))

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.x[2] = max(self.x[2], 1.0)   # keep width/height positive
        self.x[3] = max(self.x[3], 1.0)
        return self.box()

    def update(self, box):
        z = box_to_z(box)
        y = z - self.H @ self.x                    # innovation: detection minus prediction
        S = self.H @ self.P @ self.H.T + self.R    # its expected uncertainty
        K = self.P @ self.H.T @ np.linalg.inv(S)   # Kalman gain: how far to move toward the detection
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P
        return self.box()

    def box(self):
        return z_to_box(self.x)

    def velocity(self):
        return float(self.x[4]), float(self.x[5])
