# Group Aidan : Track A, TurtleBot3 navigation and ArUco docking (MuJoCo)

**Members:** <Aidan 25918525>   **Video:** <YouTube unlisted link>

## Run it
```
pip install -r requirements.txt
python run.py      # about 3 minutes: prints analysis, saves results.txt and results.png
```

## Files 
| File | Purpose |
|---|---|
| `model.py` | motion model `f`, camera model `h` (camera 5 cm ahead of the robot centre), EKF `predict`/`update` |
| `world.py` | MuJoCo room and robot, wheel encoders, 360-beam LiDAR (`mj_multiRay`), camera range/bearing, noise calibration |
| `control.py` | smooth path, error model, LQR/LQI (Bryson's rule), LiDAR avoidance layer |
| `run.py` | experiments, analysis, figure |

**Platform note:** this is my own TurtleBot3-style MuJoCo model, NOT the course scene. The camera returns range/bearing to markers
directly (no image rendering). Re-point `world.py` to the course scene when available.
Ground truth (`Robot.pose`) is read only in `run.py`'s logging block and in the `true_state` baseline.

## Design
* **Model (A):** state `[x, y, theta]`, input `(v, w)`. Error model `e_y' = v e_theta`, `e_theta' = w - v kappa`. Controllability and observability ranks are in `results.txt`.
* **Sensing (B):** calibrated by standing still at known ranges: range sigma 1.5 cm at 1 m to 11 cm at 5 m, bearing 0.01 rad. Frame change: the camera is offset 5 cm from the robot centre, handled in `camera_position`.
* **Estimation (C):** EKF, odometry from wheel-encoder increments (0.02 m/s, 0.05 rad/s assumed) plus a small heading drift term for the worn left wheel the encoders cannot see.
* **Control (D):** LQI on `[e_y, e_theta, z]`, Bryson limits 10 cm, 10 deg, 1 rad/s, 0.15 m s; feedforward `v kappa`; integrator frozen when clipped or avoiding.
* **LiDAR:** if anything is within 0.35 m ahead the robot slows and steers to the freer side; the planner's map does not contain the `surprise` obstacle.

## Results (20 trials per condition, seeds from 25918525)
```
condition   success collide  dock cm dock deg rms ey cm mean ey cm pos rmse cm  ANEES(3) NEES in 95% avoid %
nominal        1.00    0.00      2.1      1.0       2.1       -0.1         6.1       3.1        0.94     0.1
lqr_only       1.00    0.00      2.1      1.1       3.2        1.0         6.2       3.3        0.93     0.0
true_state     1.00    0.00      1.8      1.2       1.2       -0.2         4.7       5.8        0.93     0.0
noise_x3       0.95    0.00      2.6      1.1       3.2        0.5         7.7      22.0        0.22     0.1
R_x0.1         1.00    0.00      1.9      1.1       2.6        0.1         6.8      20.0        0.30     0.1
blackout       1.00    0.00      2.1      2.3      17.9        7.8        17.7       9.6        0.70     0.1
surprise       1.00    0.00      2.2      1.0       2.6       -0.4         6.1       3.2        0.93     7.4
no_lidar       0.05    0.95      2.1      1.0       2.1       -0.2         6.2       3.0        0.95     0.0
```
* `nominal`: docks every time, about 2 cm. ANEES 3.1 (ideal 3): the filter is honest.
* `noise_x3`, `R_x0.1`: still dock, but ANEES about 20, so the filter is overconfident.
* `blackout` (camera off 15 s): cross-track error about 18 cm; recovers when markers return.
* `surprise` vs `no_lidar`: an unmapped obstacle 30 cm off the path. With LiDAR 100% success; without 5%, 95% collide.
* **Limit found:** when the obstacle is closer to the path (12-25 cm) the reactive layer fails or squeezes by with ~1 cm margin, because it fights the path tracker. Explain why in your words.
* <Explain in your words why LQI barely beats LQR here: many markers keep correcting position.>

## Contributions
| Aidan | Coded everything |

## Generative AI declaration
Google Gemini was used to generalise and explain the thought process behind the production on the assesment in general. It did not write a line of code.
