"""The MuJoCo world: a TurtleBot3-style robot in a room with obstacles and markers.
Sensors: wheel encoders, a 360-beam LiDAR (mj_multiRay) and a camera that returns range/bearing to markers."""
import numpy as np
import mujoco
from model import T, wrap, camera_position

MARKERS = np.array([[3, 0], [0, 3], [-3, 1.5], [0, -3], [3, -2], [-2, 3], [2, 3], [-3, -1.5], [-2, -3], [2, -3], [3, 2]], float)
NORMALS = np.array([np.pi, -np.pi / 2, 0, np.pi / 2, np.pi, -np.pi / 2, -np.pi / 2, 0, np.pi / 2, np.pi / 2, np.pi])  # direction each marker faces
OBSTACLES = [(-0.5, 0.8, 0.35), (0.8, -0.9, 0.40), (-1.5, -1.2, 0.30), (1.2, 1.5, 0.30)]   # x, y, radius: on the planner's map
WAYPOINTS = [(-2.2, -2.0), (-1.2, -1.9), (0.0, -0.4), (1.4, 0.2), (1.8, 0.0), (2.4, 0.0)]  # last point = dock (marker 0)
FOV, MAX_RANGE, LIDAR_MAX = np.deg2rad(90), 5.0, 3.5
WHEEL_R, WHEEL_SEP, Z0 = 0.033, 0.160, 0.033
LIDAR_ANGLES = np.linspace(-np.pi, np.pi, 360, endpoint=False)


def true_range_sigma(r): return 0.01 + 0.004 * r ** 2       # TRUE camera noise (the filter learns it in calibrate())
BEARING_SIGMA = 0.01
ODOM_SIGMA = (0.01, 0.02)


def build_xml(pose, wear, surprise):
    """Room, obstacles, markers and the robot. `wear` makes the LEFT wheel physically smaller (encoders cannot see it)."""
    rl = WHEEL_R * (1 - wear)
    obs = list(OBSTACLES) + ([surprise] if surprise else [])
    geoms = [f'<geom type="box" pos="{x} {y} .25" size="{sx} {sy} .25" rgba=".9 .9 .9 1"/>'
             for x, y, sx, sy in [(0, -3.025, 3.05, .025), (0, 3.025, 3.05, .025), (-3.025, 0, .025, 3.05), (3.025, 0, .025, 3.05)]]
    geoms += [f'<geom type="cylinder" pos="{x} {y} .2" size="{r} .2" rgba=".5 .5 .55 1"/>' for x, y, r in obs]
    for m, nu in zip(MARKERS, NORMALS):                      # flat orange squares on the walls (visual only)
        zM = np.array([np.cos(nu), np.sin(nu), 0]); xM = np.cross([0, 0, 1], zM); p = [m[0] + .003 * zM[0], m[1] + .003 * zM[1], .15]
        geoms.append(f'<geom type="plane" pos="{p[0]} {p[1]} {p[2]}" size=".15 .15 .001" xyaxes="{xM[0]} {xM[1]} {xM[2]} 0 0 1" '
                     f'rgba="1 .5 0 1" contype="0" conaffinity="0"/>')
    wheel = '<joint name="{n}" type="hinge" axis="0 1 0" damping="0.0005" armature="0.0005"/>' \
            '<geom type="cylinder" size="{r} .0105" euler="90 0 0" mass=".03" friction="1.5 .005 .0001" group="2" rgba=".05 .05 .05 1"/>'
    return f"""<mujoco><compiler angle="degree"/><option timestep="0.002" integrator="implicitfast"/>
<worldbody><light pos="0 0 4" dir="0 0 -1"/><geom name="floor" type="plane" size="5 5 .1" rgba=".8 .8 .8 1"/>{''.join(geoms)}
<body name="base" pos="{pose[0]} {pose[1]} {Z0}" euler="0 0 {np.degrees(pose[2])}"><freejoint/>
<geom type="cylinder" size=".07 .03" pos="0 0 .03" mass=".8" group="2"/>
<geom type="sphere" size=".0125" pos=".05 0 {-(Z0 - .0125) + .0008}" mass=".02" condim="1" priority="1" group="2"/>
<geom type="sphere" size=".0125" pos="-.05 0 {-(Z0 - .0125) + .0008}" mass=".02" condim="1" priority="1" group="2"/>
<site name="lidar" pos="0 0 .117" size=".005"/>
<body pos="0 {WHEEL_SEP / 2} {-(WHEEL_R - rl)}">{wheel.format(n="wheel_left_joint", r=rl)}</body>
<body pos="0 {-WHEEL_SEP / 2} 0">{wheel.format(n="wheel_right_joint", r=WHEEL_R)}</body></body></worldbody>
<actuator><velocity name="left" joint="wheel_left_joint" kv=".05" ctrlrange="-30 30" forcerange="-.08 .08"/>
<velocity name="right" joint="wheel_right_joint" kv=".05" ctrlrange="-30 30" forcerange="-.08 .08"/></actuator></mujoco>"""


def sense(cam_pos, heading, marker, rng, scale=1.0):
    """One noisy range/bearing reading of a marker, from the camera position."""
    d = marker - cam_pos
    r = np.hypot(*d)
    return np.array([r + rng.normal(0, scale * true_range_sigma(r)), wrap(np.arctan2(d[1], d[0]) - heading) + rng.normal(0, scale * BEARING_SIGMA)])


class Robot:
    def __init__(self, pose, seed, wear=0.04, noise_scale=1.0, blackout=(), surprise=None):
        self.model = mujoco.MjModel.from_xml_string(build_xml(pose, wear, surprise))
        self.data = mujoco.MjData(self.model)
        self.rng, self.scale, self.blackout, self.t = np.random.default_rng(seed), noise_scale, blackout, 0.0
        self.obstacles = list(OBSTACLES) + ([surprise] if surprise else [])
        m = self.model
        self.jl, self.jr = m.joint("wheel_left_joint").qposadr[0], m.joint("wheel_right_joint").qposadr[0]
        self.base, self.site = m.body("base").id, m.site("lidar").id
        mujoco.mj_forward(m, self.data)
        self.phi = np.array([self.data.qpos[self.jl], self.data.qpos[self.jr]])
        self.min_clearance = np.inf

    @property
    def pose(self):
        """TRUE pose [x, y, theta] from MuJoCo. Only the evaluation code may read this."""
        R = self.data.xmat[self.base].reshape(3, 3)
        return np.array([*self.data.xpos[self.base][:2], np.arctan2(R[1, 0], R[0, 0])])

    def lidar(self):
        """360 ranges in the robot frame (3.5 = no return). Rays ignore the robot's own geoms (group 2)."""
        pnt = self.data.site_xpos[self.site].copy()
        a = self.pose[2] + LIDAR_ANGLES
        vec = np.column_stack([np.cos(a), np.sin(a), np.zeros(360)]).ravel()
        gid, dist = np.zeros(360, np.int32), np.zeros(360)
        mujoco.mj_multiRay(self.model, self.data, pnt, vec, np.array([1, 0, 0, 0, 0, 0], np.uint8), 1, -1, gid, dist, None, 360, LIDAR_MAX)
        hit = (dist > 0) & (dist < LIDAR_MAX)
        return np.where(hit, dist + self.rng.normal(0, 0.01, 360), LIDAR_MAX)

    def camera(self):
        """Markers in view: in front of the camera, within range, and not hidden behind an obstacle (ray test)."""
        if any(a <= self.t < b for a, b in self.blackout):
            return {}
        x = self.pose
        cam = camera_position(x)
        seen = {}
        for i, (m, nu) in enumerate(zip(MARKERS, NORMALS)):
            d = m - cam
            r, phi = np.hypot(*d), wrap(np.arctan2(d[1], d[0]) - x[2])
            if r > MAX_RANGE or abs(phi) > FOV / 2 or np.cos(nu) * -d[0] + np.sin(nu) * -d[1] < 0.34 * r:   # marker must face the camera
                continue
            gid = np.zeros(1, np.int32)
            hit = mujoco.mj_ray(self.model, self.data, np.r_[cam, .15], np.r_[d / r, 0], np.array([1, 0, 0, 0, 0, 0], np.uint8), 1, -1, gid)
            if 0 <= hit < r - 0.02:
                continue                                         # something is in the way
            seen[i] = sense(cam, x[2], m, self.rng, self.scale)
        return seen

    def step(self, u):
        v, w = np.clip(u[0], 0, 0.22), np.clip(u[1], -2.84, 2.84)
        self.data.ctrl[0] = (v - w * WHEEL_SEP / 2) / WHEEL_R      # nominal wheel speeds: what the controller believes
        self.data.ctrl[1] = (v + w * WHEEL_SEP / 2) / WHEEL_R
        for _ in range(int(round(T / self.model.opt.timestep))):
            mujoco.mj_step(self.model, self.data)
        self.t += T
        x = self.pose
        cl = min(np.hypot(x[0] - cx, x[1] - cy) - r - 0.11 for cx, cy, r in self.obstacles)
        self.min_clearance = min(self.min_clearance, cl)
        phi = np.array([self.data.qpos[self.jl], self.data.qpos[self.jr]])
        dphi, self.phi = phi - self.phi, phi                        # encoder increments over this step
        odom = (WHEEL_R * (dphi[0] + dphi[1]) / (2 * T) + self.rng.normal(0, self.scale * ODOM_SIGMA[0]),
                WHEEL_R * (dphi[1] - dphi[0]) / (WHEEL_SEP * T) + self.rng.normal(0, self.scale * ODOM_SIGMA[1]))
        return odom, self.camera(), self.lidar()


def calibrate(seed=0, n=300):
    """Part B: stand still at known ranges, measure the spread of the readings, fit sigma_r(r) and sigma_phi."""
    rng = np.random.default_rng(seed)
    ranges = np.arange(1.0, 5.1, 0.5)
    sr, sp = [], []
    for r in ranges:
        cam = MARKERS[1] - np.array([0, r]); z = np.array([sense(cam, np.pi / 2, MARKERS[1], rng) for _ in range(n)])
        sr.append(z[:, 0].std()); sp.append(z[:, 1].std())
    coef = np.polyfit(ranges, sr, 2)
    return (lambda r: np.polyval(coef, r)), float(np.mean(sp))
