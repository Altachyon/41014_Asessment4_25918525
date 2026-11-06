"""Run everything:  python run.py        (about 3 minutes)
Prints the analysis, runs the experiments in MuJoCo, saves results.txt and results.png."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import chi2
from model import T, wrap, predict, update, jacobian, f, h
from world import Robot, calibrate, MARKERS, OBSTACLES, LIDAR_ANGLES
from control import Path, Controller, design, avoid, V

N_TRIALS = 20
SEED0 = 25918525                                     # base random seed (use your team's agreed number)
TOLD_START = np.array([-2.2, -2.0, 0.5])            # where the robot is TOLD it starts
P0_STD = np.array([0.2, 0.2, np.deg2rad(8)])         # how unsure it is (the real start is drawn from this)
FILTER_ODOM = (0.02, 0.05)                           # odometry noise the EKF assumes (a bit more than the truth)
DRIFT = (1e-6, 1e-6, np.deg2rad(0.5) ** 2)           # extra heading noise per step for what odometry cannot see
DOCK = np.array([2.4, 0.0, 0.0])

sigma_r, sigma_phi = calibrate()                     # Part B: learned from stationary readings
PATH0 = Path(TOLD_START[2])
_i = int(0.55 * len(PATH0.x))                        # a surprise obstacle ON the route, 30 cm to its left, NOT on the planner's map
_n = np.array([-np.sin(PATH0.th[_i]), np.cos(PATH0.th[_i])])
SURPRISE = (PATH0.x[_i] + 0.30 * _n[0], PATH0.y[_i] + 0.30 * _n[1], 0.20)

CONDITIONS = {
    "nominal":    dict(),
    "lqr_only":   dict(integral=False),              # no integral action
    "true_state": dict(use_truth=True),              # controller cheats: sees the true pose (upper bound)
    "noise_x3":   dict(noise_scale=3.0),             # sensors 3x noisier than the filter was told
    "R_x0.1":     dict(r_scale=0.1),                 # filter trusts the camera 10x too much
    "blackout":   dict(blackout=((15.0, 30.0),)),    # camera off for 15 s
    "surprise":   dict(surprise=SURPRISE),           # unmapped obstacle: only the LiDAR can see it
    "no_lidar":   dict(surprise=SURPRISE, lidar=False),   # same obstacle, LiDAR layer switched off
}


def run_trial(seed, integral=True, use_truth=False, noise_scale=1.0, r_scale=1.0, blackout=(), surprise=None, lidar=True):
    rng = np.random.default_rng(seed + 1000)
    x_hat, P = TOLD_START.copy(), np.diag(P0_STD ** 2)
    robot = Robot(TOLD_START + rng.normal(0, P0_STD), seed, noise_scale=noise_scale, blackout=blackout, surprise=surprise)
    path, ctrl = Path(TOLD_START[2]), Controller(integral)
    log = {k: [] for k in ("t", "true", "est", "P", "ey", "nees", "avoid")}
    u, idx, hold, reached, avoiding = (0.0, 0.0), 0, 0, False, False
    for _ in range(int(150 / T)):
        odom, readings, scan = robot.step(u)
        x_hat, P = predict(x_hat, P, odom, *FILTER_ODOM, DRIFT)
        for i, z in readings.items():
            R = r_scale * np.diag([sigma_r(z[0]) ** 2, sigma_phi ** 2])
            x_hat, P = update(x_hat, P, z, MARKERS[i], R)
        # ---- control uses the ESTIMATE (the true pose only in the 'true_state' baseline) ----
        e_y, e_th, kappa, s, idx = path.errors(robot.pose if use_truth else x_hat, idx)
        if path.end - s < 0.03:
            hold += 1; u = (0.0, 0.0)
            if hold > 10: reached = True; break
        else:
            v, w = V, ctrl.step(e_y, e_th, kappa, freeze=avoiding)
            if lidar:
                v, w, avoiding = avoid(scan, LIDAR_ANGLES, v, w)       # LiDAR layer: slow down and steer around close obstacles
            u = (v, w)
        # ---- logging: the true pose is read here, for evaluation only ----
        err = robot.pose - x_hat; err[2] = wrap(err[2])
        for k, val in (("t", robot.t), ("true", robot.pose.copy()), ("est", x_hat.copy()), ("P", P.copy()),
                       ("ey", path.nearest(robot.pose)), ("nees", err @ np.linalg.solve(P, err)), ("avoid", avoiding)):
            log[k].append(val)
    L = {k: np.array(v) for k, v in log.items()}
    pos_err = np.hypot(*(L["true"][-1, :2] - DOCK[:2])); th_err = abs(wrap(L["true"][-1, 2] - DOCK[2]))
    e = L["true"] - L["est"]; e[:, 2] = wrap(e[:, 2])
    collided = robot.min_clearance < 0
    m = dict(success=reached and pos_err < 0.05 and th_err < 0.10 and not collided, collided=collided,
             dock_cm=100 * pos_err, dock_deg=np.degrees(th_err),
             rms_ey_cm=100 * np.sqrt(np.mean(L["ey"][50:] ** 2)), mean_ey_cm=100 * np.mean(L["ey"][50:]),
             pos_rmse_cm=100 * np.sqrt(np.mean(np.sum(e[:, :2] ** 2, axis=1))),
             anees=L["nees"].mean(), nees_in=np.mean(L["nees"] <= chi2.ppf(0.95, 3)), avoid_pct=100 * L["avoid"].mean(), time=L["t"][-1])
    return m, L, path


def analysis():
    """Parts A and C: controllability, controller eigenvalues, EKF observability."""
    out = ["ANALYSIS", f"calibrated camera noise: sigma_r(1 m)={sigma_r(1):.3f} m, sigma_r(5 m)={sigma_r(5):.3f} m, sigma_phi={sigma_phi:.4f} rad"]
    for integral in (False, True):
        K, Ad, Bd = design(integral)
        ctrb = np.hstack([np.linalg.matrix_power(Ad, k) @ Bd for k in range(Ad.shape[0])])
        eig = np.abs(np.linalg.eigvals(Ad - Bd @ K))
        out.append(f"{'LQI' if integral else 'LQR'}: controllability rank {np.linalg.matrix_rank(ctrb)}/{Ad.shape[0]}, "
                   f"K = {np.round(K.ravel(), 2)}, closed-loop |eig| = {np.round(np.sort(eig)[::-1], 3)} (all < 1 means stable)")
    x = np.array([-1.0, -1.0, 0.4])
    for name, u, marks in (("one marker, robot stopped", (0, 0), [3]), ("one marker, robot moving", (V, 0), [3]),
                           ("two markers, robot stopped", (0, 0), [3, 8])):
        F = jacobian(lambda s: f(s, u), x, angle_rows=(2,))
        H = np.vstack([jacobian(lambda s: h(s, MARKERS[i]), x, angle_rows=(1,)) for i in marks])
        O = np.vstack([H @ np.linalg.matrix_power(F, k) for k in range(10)])
        sv = np.linalg.svd(O, compute_uv=False)
        out.append(f"EKF observability, {name}: rank {int(np.sum(sv > 1e-6 * sv[0]))}/3")
    return "\n".join(out)


if __name__ == "__main__":
    text = analysis(); print(text)
    rows, first = {}, {}
    for name, kw in CONDITIONS.items():
        ms = []
        for k in range(N_TRIALS):
            m, L, path = run_trial(SEED0 + k, **kw)
            ms.append(m)
            if k == 0: first[name] = (L, path)
        rows[name] = {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}
        print("done", name, flush=True)
    head = f"{'condition':11s} {'success':>7s} {'collide':>7s} {'dock cm':>8s} {'dock deg':>8s} {'rms ey cm':>9s} {'mean ey cm':>10s} {'pos rmse cm':>11s} {'ANEES(3)':>9s} {'NEES in 95%':>11s} {'avoid %':>7s}"
    table = [head] + [f"{n:11s} {r['success']:7.2f} {r['collided']:7.2f} {r['dock_cm']:8.1f} {r['dock_deg']:8.1f} {r['rms_ey_cm']:9.1f} {r['mean_ey_cm']:10.1f} "
                      f"{r['pos_rmse_cm']:11.1f} {r['anees']:9.1f} {r['nees_in']:11.2f} {r['avoid_pct']:7.1f}" for n, r in rows.items()]
    result = f"{text}\n\nRESULTS over {N_TRIALS} trials per condition (MuJoCo)\n" + "\n".join(table)
    open("results.txt", "w").write(result); print("\n" + "\n".join(table))
    fig, ax = plt.subplots(2, 2, figsize=(13, 10))
    for col, name in enumerate(("nominal", "surprise")):
        L, path = first[name]; a = ax[0, col]
        a.plot(path.x, path.y, "g:", label="planned path"); a.plot(L["true"][:, 0], L["true"][:, 1], "k", label="true"); a.plot(L["est"][:, 0], L["est"][:, 1], "r--", label="estimate")
        a.plot(L["true"][L["avoid"], 0], L["true"][L["avoid"], 1], "m.", ms=4, label="LiDAR avoiding")
        for cx, cy, r in OBSTACLES: a.add_patch(plt.Circle((cx, cy), r, color="0.6"))
        if name == "surprise": a.add_patch(plt.Circle(SURPRISE[:2], SURPRISE[2], color="tab:red", alpha=.6))
        a.plot(MARKERS[:, 0], MARKERS[:, 1], "s", color="tab:orange"); a.plot(*DOCK[:2], "g*", ms=14); a.set_aspect("equal"); a.legend(fontsize=8); a.set_title(name)
    L, _ = first["nominal"]
    err = np.hypot(*(L["true"][:, :2] - L["est"][:, :2]).T); sig = np.sqrt(L["P"][:, 0, 0] + L["P"][:, 1, 1])
    ax[1, 0].plot(L["t"], 100 * err, label="position error"); ax[1, 0].plot(L["t"], 200 * sig, "--", label="2 sigma from P"); ax[1, 0].legend(); ax[1, 0].set_title("Nominal: estimate error and claimed uncertainty [cm]")
    ax[1, 1].semilogy(L["t"], L["nees"]); ax[1, 1].axhline(chi2.ppf(0.95, 3), color="r", ls="--"); ax[1, 1].set_title(f"Nominal: NEES (should be about 3), ANEES = {L['nees'].mean():.1f}")
    for a in ax[1]: a.set_xlabel("t [s]"); a.grid(alpha=.4)
    fig.tight_layout(); fig.savefig("results.png", dpi=110); print("saved results.txt and results.png")
