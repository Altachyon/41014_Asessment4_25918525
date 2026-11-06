"""Motion model, measurement model and EKF for the TurtleBot (state x = [x, y, theta])."""
import numpy as np

T = 0.1                                              # sample time [s]
CAM = 0.05                                           # the camera sits 5 cm ahead of the robot centre (x forward)


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def f(x, u):
    """Motion model: robot moves along an arc with speed v and turn rate w for T seconds."""
    v, w = u
    th = x[2]
    th_new = th + w * T
    if abs(w) < 1e-6:                                # straight line
        dx, dy = v * T * np.cos(th), v * T * np.sin(th)
    else:
        dx, dy = v / w * (np.sin(th_new) - np.sin(th)), v / w * (np.cos(th) - np.cos(th_new))
    return np.array([x[0] + dx, x[1] + dy, wrap(th_new)])


def jacobian(fun, x, angle_rows=(), eps=1e-6):
    """Numerical Jacobian of fun at x (central differences), so there is no algebra to get wrong.
    angle_rows lists the outputs that are angles (their differences are wrapped)."""
    cols = []
    for i in range(len(x)):
        d = np.zeros(len(x)); d[i] = eps
        diff = np.array(fun(x + d) - fun(x - d), float)
        for k in angle_rows:
            diff[k] = wrap(diff[k])
        cols.append(diff / (2 * eps))
    return np.array(cols).T


def camera_position(x):
    """Frame change: camera position in the world = robot centre + CAM metres along the robot's heading."""
    return np.array([x[0] + CAM * np.cos(x[2]), x[1] + CAM * np.sin(x[2])])


def h(x, m):
    """Measurement model: range and bearing of a marker at m = (mx, my), measured BY THE CAMERA (not the robot centre)."""
    d = np.asarray(m) - camera_position(x)
    return np.array([np.hypot(*d), wrap(np.arctan2(d[1], d[0]) - x[2])])


def predict(x, P, u, sigma_v, sigma_w, drift):
    """EKF prediction. Odometry noise (sigma_v, sigma_w) is turned into state noise through G = df/du."""
    F = jacobian(lambda s: f(s, u), x, angle_rows=(2,))
    G = jacobian(lambda uu: f(x, uu), np.array(u, float), angle_rows=(2,))
    Q = G @ np.diag([sigma_v ** 2, sigma_w ** 2]) @ G.T + np.diag(drift)
    return f(x, u), F @ P @ F.T + Q


def update(x, P, z, m, R):
    """EKF update with one range/bearing measurement z of the marker at m."""
    H = jacobian(lambda s: h(s, m), x, angle_rows=(1,))
    innov = z - h(x, m)
    innov[1] = wrap(innov[1])
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)
    x = x + K @ innov
    x[2] = wrap(x[2])
    I_KH = np.eye(3) - K @ H
    P = I_KH @ P @ I_KH.T + K @ R @ K.T              # Joseph form keeps P symmetric and positive
    return x, P
