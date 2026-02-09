import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ============================================================
# 1) Build exact T-shaped 3D body (mm)
# ============================================================

pts = []

def block_centers(xmin, xmax, ymin, ymax, zmin, zmax, h):
    xs = np.arange(xmin + h/2, xmax, h)
    ys = np.arange(ymin + h/2, ymax, h)
    zs = np.arange(zmin + h/2, zmax, h)
    for x in xs:
        for y in ys:
            for z in zs:
                pts.append([x,y,z])

h = 5.0
zmin, zmax = -20, 20

# vertical bar (40 x 200 x 40)
block_centers(-20, 20, -100, 100, zmin, zmax, h)

# horizontal bar (50 x 40 x 40), attached to right face
block_centers(20, 70, -20, 20, zmin, zmax, h)

x = np.array(pts)
m = np.ones(len(x))
print("Particles:", len(x))

# ============================================================
# 2) COM and full inertia tensor
# ============================================================

M = m.sum()
x_cm = (m[:,None] * x).sum(axis=0) / M
r = x - x_cm

I = np.zeros((3,3))
for i in range(len(r)):
    xi, yi, zi = r[i]
    mi = m[i]
    I += mi * np.array([
        [yi*yi + zi*zi, -xi*yi,       -xi*zi],
        [-yi*xi,        xi*xi + zi*zi, -yi*zi],
        [-zi*xi,        -zi*yi,        xi*xi + yi*yi]
    ])

eigvals, eigvecs = np.linalg.eigh(I)
Q = eigvecs                    # body -> world
r_body = (Q.T @ r.T).T         # world -> body
I_body = Q.T @ I @ Q
I_diag = np.array([I_body[0,0], I_body[1,1], I_body[2,2]])

print("Principal moments:", I_diag)

# ============================================================
# 3) Quaternion helpers
# ============================================================

def quat_mul(q, p):
    w0,x0,y0,z0 = q
    w1,x1,y1,z1 = p
    return np.array([
        w0*w1 - x0*x1 - y0*y1 - z0*z1,
        w0*x1 + x0*w1 + y0*z1 - z0*y1,
        w0*y1 - x0*z1 + y0*w1 + z0*x1,
        w0*z1 + x0*y1 - y0*x1 + z0*w1
    ])

def quat_to_R(q):
    q0,q1,q2,q3 = q
    return np.array([
        [q0*q0+q1*q1-q2*q2-q3*q3, 2*(q1*q2-q0*q3),       2*(q1*q3+q0*q2)],
        [2*(q1*q2+q0*q3),       q0*q0-q1*q1+q2*q2-q3*q3, 2*(q2*q3-q0*q1)],
        [2*(q1*q3-q0*q2),       2*(q2*q3+q0*q1),       q0*q0-q1*q1-q2*q2+q3*q3]
    ])

# ============================================================
# 4) Euler rigid body RHS (torque = 0)
# ============================================================

def rhs(w):
    wx, wy, wz = w
    Ix, Iy, Iz = I_diag
    return np.array([
        ((Iz - Iy)*wy*wz)/Ix,
        ((Ix - Iz)*wz*wx)/Iy,
        ((Iy - Ix)*wx*wy)/Iz
    ])

def dqdt(q, w):
    return 0.5 * quat_mul(q, np.array([0.0, *w]))

# ============================================================
# 5) RK4 integrator
# ============================================================

def rk4_step(w, q, dt):
    k1 = rhs(w)
    k2 = rhs(w + 0.5*dt*k1)
    k3 = rhs(w + 0.5*dt*k2)
    k4 = rhs(w + dt*k3)
    w_new = w + dt/6*(k1 + 2*k2 + 2*k3 + k4)

    k1q = dqdt(q, w)
    k2q = dqdt(q + 0.5*dt*k1q, w_new)
    k3q = dqdt(q + 0.5*dt*k2q, w_new)
    k4q = dqdt(q + dt*k3q, w_new)

    q_new = q + dt/6*(k1q + 2*k2q + 2*k3q + k4q)
    q_new /= np.linalg.norm(q_new)

    return w_new, q_new

# ============================================================
# 6) Initial condition (paper)
# ============================================================

idx = np.argsort(I_diag)
mid = idx[1]

w_body = np.zeros(3)
w_body[mid] = 100.0
w_body[idx[0]] = 0.01
w_body[idx[2]] = -0.01

q = np.array([1.0,0.0,0.0,0.0])
dt = 1e-4

# ============================================================
# 7) Run + store frames
# ============================================================

frames = []
for n in range(12000):
    w_body, q = rk4_step(w_body, q, dt)
    if n % 50 == 0:
        R = quat_to_R(q)
        pts_w = (R @ r_body.T).T
        frames.append(pts_w.copy())

print("Stored frames:", len(frames))

# ============================================================
# 8) Write VTK files
# ============================================================

def write_vtk_points(fname, pts):
    with open(fname, "w") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Dzhanibekov rigid body\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")
        f.write(f"POINTS {len(pts)} float\n")
        for p in pts:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")


def write_csv_points(fname, pts):
    with open(fname, "w") as f:
        for p in pts:
            f.write(f"{p[0]},{p[1]},{p[2]}\n")

for i, pts in enumerate(frames):
    # write_vtk_points(f"t_body_{i:04d}.vtk", pts)
    write_csv_points(f"t_body_{i:04d}.csv", pts)

print("VTK files written.")

# ============================================================
# 9) Python animation
# ============================================================

# fig = plt.figure(figsize=(6,6))
# ax = fig.add_subplot(111, projection="3d")
# sc = ax.scatter(frames[0][:,0], frames[0][:,1], frames[0][:,2], s=5)

# ax.set_xlim(-120,120)
# ax.set_ylim(-120,120)
# ax.set_zlim(-120,120)
# ax.set_box_aspect([1,1,1])

# def update(i):
#     sc._offsets3d = (frames[i][:,0], frames[i][:,1], frames[i][:,2])
#     return sc,

# ani = FuncAnimation(fig, update, frames=len(frames), interval=40)
# plt.show()
