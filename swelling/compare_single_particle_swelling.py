import numpy as np
import matplotlib.pyplot as plt

# ── Digitized experimental data from figure ────────────────────────────────────
# (time [min], d/d0)
exp_data = np.array([
    [0,    1.00],
    [1,    1.90],
    [2,    2.25],
    [5,    2.80],
    [10,   3.20],
    [15,   3.45],
    [20,   3.70],
    [25,   3.75],
    [30,   3.85],
    [35,   3.90],
    [40,   4.00],
    [50,   4.15],
    [60,   4.25],
    [75,   4.30],
    [90,   4.35],
    [100,  4.40],
    [130,  4.50],
    [150,  4.50],
    [180,  4.60],
    [210,  4.65],
    [240,  4.70],
    [270,  4.75],
])

# ── Digitized macroscopic model curve from paper (red dash-dot) ────────────────
# (time [min], d/d0)
paper_macro = np.array([
    [0,    1.00],
    [1,    2.00],
    [2,    2.50],
    [5,    3.50],
    [10,   3.90],
    [15,   4.05],
    [20,   4.15],
    [30,   4.28],
    [40,   4.35],
    [50,   4.40],
    [75,   4.48],
    [100,  4.52],
    [150,  4.60],
    [200,  4.65],
    [250,  4.70],
    [300,  4.73],
])

# ── Physical parameters ────────────────────────────────────────────────────────
rho_s   = 1486.0
rho_w   = 1000.0
c_wl    = 992.0
D_0     = 4e-7
delta   = 1.81
c_max   = c_wl
v_0     = 3e-7        # swelling front velocity for microscopic model [m/s]

r0      = 0.95e-3
V_p0    = (4.0 / 3.0) * np.pi * r0**3
m_p0    = rho_s * V_p0

dt      = 1.0
t_end   = 300.0 * 60
steps   = int(t_end / dt)
t_hist  = np.zeros(steps + 1)
t_hist[0] = 0.0

# ── Macroscopic model ──────────────────────────────────────────────────────────
m_w  = 0.0;  V_p = V_p0;  r_p = r0;  m_p = m_p0
r_hist_macro = np.zeros(steps + 1)
r_hist_macro[0] = r_p

for i in range(steps):
    S_p     = 4.0 * np.pi * r_p**2
    c_wp    = (m_p - m_p0) / V_p
    D       = D_0 * np.exp(-delta * c_wp / c_max)
    m_w_dot = S_p * D * (c_wl - c_wp) / r_p
    m_w    += m_w_dot * dt
    m_p     = m_p0 + m_w
    V_p     = V_p0 + m_w / rho_w
    r_p     = (3.0 * V_p / (4.0 * np.pi))**(1.0 / 3.0)
    t_hist[i+1]        = (i + 1) * dt
    r_hist_macro[i+1]  = r_p

# ── Microscopic model ──────────────────────────────────────────────────────────
m_w  = 0.0;  r_p = r0;  r_c = r0
V_c  = V_p0; V_s = 0.0;  V_p = V_p0;  m_p = m_p0
core_present = True
r_hist_micro = np.zeros(steps + 1)
r_hist_micro[0] = r_p

for i in range(steps):
    S_p = 4.0 * np.pi * r_p**2

    if core_present:
        c_wp    = (m_p - m_p0) / V_s if V_s > 0.0 else 0.0
        D       = D_0 * np.exp(-delta * c_wp / c_max)
        m_w_dot = S_p * D * (c_wl - c_wp) / r_p
        m_w    += m_w_dot * dt
        m_p     = m_p0 + m_w
        dV_d    = m_w_dot * dt / rho_w

        r_c_old = r_c
        r_c     = r_c - v_0 * dt
        if r_c <= 0.0:
            dV_c = (4.0 / 3.0) * np.pi * r_c_old**3
            r_c  = 0.0;  V_c = 0.0;  core_present = False
        else:
            V_c  = (4.0 / 3.0) * np.pi * r_c**3
            dV_c = (4.0 / 3.0) * np.pi * (r_c_old**3 - r_c**3)

        V_s += dV_d + dV_c
        V_p  = V_s + V_c
    else:
        c_wp    = (m_p - m_p0) / V_p
        D       = D_0 * np.exp(-delta * c_wp / c_max)
        m_w_dot = S_p * D * (c_wl - c_wp) / r_p
        m_w    += m_w_dot * dt
        m_p     = m_p0 + m_w
        V_p     = V_p0 + m_w / rho_w

    r_p = (3.0 * V_p / (4.0 * np.pi))**(1.0 / 3.0)
    r_hist_micro[i+1] = r_p

# ── Plot comparison ────────────────────────────────────────────────────────────
t_min        = t_hist / 60.0
d_d0_macro   = 2.0 * r_hist_macro / (2.0 * r0)
d_d0_micro   = 2.0 * r_hist_micro / (2.0 * r0)

fig, ax = plt.subplots()
ax.errorbar(exp_data[:, 0], exp_data[:, 1],
            fmt='ko', markersize=5, capsize=3, label="Experiment (digitized)")
ax.plot(paper_macro[:, 0], paper_macro[:, 1], 'r-.', label="Macroscopic model (paper)")
ax.plot(t_min, d_d0_macro, 'b--',  label="Macroscopic model (ours)")
ax.plot(t_min, d_d0_micro, 'g-',   label="Microscopic model (ours)")
ax.set_xlabel("Time [min]")
ax.set_ylabel(r"$d / d_0$")
ax.set_title("Single particle swelling — macroscopic vs microscopic")
ax.set_xlim(0, 300)
ax.set_ylim(1, 5)
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.savefig("compare_single_particle_swelling.pdf")
plt.show()
