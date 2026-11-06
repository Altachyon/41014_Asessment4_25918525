"""Path following with LQR + integral action (LQI) on the cross-track and heading error."""
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.linalg import solve_discrete_are
from model import T, wrap
from world import WAYPOINTS

V = 0.15                                             # cruise speed [m/s]


class Path:
    """Smooth path through the waypoints, stored as points with heading, curvature and arc length."""
    def __init__(self, start_heading, goal_heading=0.0):
        pts = np.array(WAYPOINTS)
        s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
        cs = CubicSpline(s, pts, bc_type=((1, [np.cos(start_heading), np.sin(start_heading)]),
                                          (1, [np.cos(goal_heading), np.sin(goal_heading)])))
        u = np.arange(0, s[-1], 0.005)
        d1, d2, p = cs(u, 1), cs(u, 2), cs(u)
        self.x, self.y = p[:, 0], p[:, 1]
        self.th = np.arctan2(d1[:, 1], d1[:, 0])
        self.kappa = (d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]) / np.hypot(d1[:, 0], d1[:, 1]) ** 3
        self.s = np.r_[0, np.cumsum(np.hypot(np.diff(self.x), np.diff(self.y)))]
        self.end = self.s[-1]

    def errors(self, pose, start=0, window=120):
        """Cross-track error e_y (left of the path is positive), heading error, curvature, arc length, index."""
        seg = slice(start, min(start + window, len(self.x)))
        i = start + int(np.argmin(np.hypot(self.x[seg] - pose[0], self.y[seg] - pose[1])))
        e_y = -np.sin(self.th[i]) * (pose[0] - self.x[i]) + np.cos(self.th[i]) * (pose[1] - self.y[i])
        return e_y, wrap(pose[2] - self.th[i]), self.kappa[i], self.s[i], i

    def nearest(self, pose):
        """Evaluation only: cross-track error of the TRUE pose against the whole path."""
        i = int(np.argmin(np.hypot(self.x - pose[0], self.y - pose[1])))
        return -np.sin(self.th[i]) * (pose[0] - self.x[i]) + np.cos(self.th[i]) * (pose[1] - self.y[i])


def model(v=V):
    """Error model e = [e_y, e_theta] (straight path): e_y' = v e_theta, e_theta' = omega correction. Exact ZOH discretisation."""
    Ad = np.array([[1, v * T], [0, 1]])
    Bd = np.array([[v * T ** 2 / 2], [T]])
    return Ad, Bd


def design(integral=True, ey_max=0.10, eth_max=np.deg2rad(10), dw_max=1.0, z_max=0.15):
    """LQR gain using Bryson's rule: weight = 1 / (largest acceptable value)^2."""
    Ad, Bd = model()
    Q = np.diag([1 / ey_max ** 2, 1 / eth_max ** 2])
    if integral:                                     # add the integrator state z+ = z + T e_y
        Ad = np.block([[Ad, np.zeros((2, 1))], [np.array([[T, 0]]), np.ones((1, 1))]])
        Bd = np.vstack([Bd, [[0]]])
        Q = np.diag([1 / ey_max ** 2, 1 / eth_max ** 2, 1 / z_max ** 2])
    R = np.array([[1 / dw_max ** 2]])
    P = solve_discrete_are(Ad, Bd, Q, R)
    return np.linalg.solve(R + Bd.T @ P @ Bd, Bd.T @ P @ Ad), Ad, Bd


class Controller:
    def __init__(self, integral=True):
        self.integral, self.z = integral, 0.0
        self.K = design(integral)[0].ravel()

    def step(self, e_y, e_th, kappa, freeze=False):
        e = np.array([e_y, e_th, self.z]) if self.integral else np.array([e_y, e_th])
        raw = V * kappa - self.K @ e                 # feedforward v*kappa + feedback
        w = float(np.clip(raw, -2.84, 2.84))
        if self.integral and w == raw and not freeze:   # anti-windup: no integrating while clipped or while avoiding
            self.z += T * e_y
        return w


def avoid(scan, angles, v, w, trigger=0.35, stop=0.18):
    """LiDAR layer: if something is closer than `trigger` metres ahead (+-35 deg), slow down and steer to the freer side."""
    front = np.abs(angles) < np.deg2rad(35)
    d = scan[front].min()
    if d >= trigger:
        return v, w, False
    v *= max(0.25, (d - stop) / (trigger - stop))
    w += 1.2 * np.sign(scan[(angles > 0) & front].mean() - scan[(angles < 0) & front].mean())
    return v, w, True
