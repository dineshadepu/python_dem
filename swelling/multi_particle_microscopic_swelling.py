import numpy as np
import matplotlib.pyplot as plt

# ── Physical parameters ────────────────────────────────────────────────────────
rho_s   = 1486.0      # solid (dry) density              [kg/m^3]
rho_w   = 1000.0      # water density                    [kg/m^3]
c_wl    = 992.0       # water concentration in liquid    [kg/m^3]
D_0     = 4e-7        # reference diffusion coefficient  [m^2/s]
delta   = 1.81        # diffusion sensitivity parameter  [-]
c_max   = c_wl        # maximum concentration            [kg/m^3]
v_0     = 3e-7        # swelling front (inward) velocity [m/s]

# ── Particles (SoA: index j = particle j) ─────────────────────────────────────
n   = 3
r0  = np.array([0.95e-3, 0.95e-3, 0.95e-3])   # initial radii [m]

V_p0 = (4.0 / 3.0) * np.pi * r0**3
m_p0 = rho_s * V_p0

# ── Time integration ───────────────────────────────────────────────────────────
dt    = 1.0
t_end = 300.0 * 60   # 300 minutes
steps = int(t_end / dt)

# ── State variables (SoA) ─────────────────────────────────────────────────────
m_w  = np.zeros(n)
r_p  = r0.copy()
r_c  = r0.copy()          # core radius per particle
V_c  = V_p0.copy()        # core volume per particle
V_s  = np.zeros(n)        # swollen zone volume per particle
V_p  = V_p0.copy()
m_p  = m_p0.copy()
core = np.ones(n, dtype=bool)   # core_present flag per particle

# r_hist[j] = radius history of particle j
t_hist = np.zeros(steps + 1)
r_hist = np.zeros((n, steps + 1))
t_hist[0] = 0.0;  r_hist[:, 0] = r_p

for i in range(steps):
    S_p = 4.0 * np.pi * r_p**2

    # ── Particles with core still present ─────────────────────────────────────
    if np.any(core):
        idx = core

        V_s_safe    = np.where(V_s[idx] > 0.0, V_s[idx], 1.0)
        c_wp        = np.where(V_s[idx] > 0.0, (m_p[idx] - m_p0[idx]) / V_s_safe, 0.0)
        D           = D_0 * np.exp(-delta * c_wp / c_max)
        m_w_dot     = S_p[idx] * D * (c_wl - c_wp) / r_p[idx]

        m_w[idx]   += m_w_dot * dt
        m_p[idx]    = m_p0[idx] + m_w[idx]
        dV_d        = m_w_dot * dt / rho_w

        r_c_old     = r_c[idx].copy()
        r_c[idx]   -= v_0 * dt

        # Particles whose core just vanished
        gone        = idx & (r_c <= 0.0)
        alive       = idx & (r_c > 0.0)

        if np.any(gone):
            dV_c_gone       = (4.0 / 3.0) * np.pi * r_c_old[gone[idx]]**3
            V_s[gone]      += dV_d[gone[idx]] + dV_c_gone
            r_c[gone]       = 0.0
            V_c[gone]       = 0.0
            V_p[gone]       = V_s[gone]
            core[gone]      = False

        if np.any(alive):
            dV_c_alive      = (4.0 / 3.0) * np.pi * (r_c_old[alive[idx]]**3 - r_c[alive]**3)
            V_c[alive]      = (4.0 / 3.0) * np.pi * r_c[alive]**3
            V_s[alive]     += dV_d[alive[idx]] + dV_c_alive
            V_p[alive]      = V_s[alive] + V_c[alive]

    # ── Particles with no core — macroscopic update ────────────────────────────
    if np.any(~core):
        idx = ~core

        c_wp        = (m_p[idx] - m_p0[idx]) / V_p[idx]
        D           = D_0 * np.exp(-delta * c_wp / c_max)
        m_w_dot     = S_p[idx] * D * (c_wl - c_wp) / r_p[idx]

        m_w[idx]   += m_w_dot * dt
        m_p[idx]    = m_p0[idx] + m_w[idx]
        V_p[idx]    = V_p0[idx] + m_w[idx] / rho_w

    r_p = (3.0 * V_p / (4.0 * np.pi))**(1.0 / 3.0)

    t_hist[i+1]    = (i + 1) * dt
    r_hist[:, i+1] = r_p

# ── Plot ───────────────────────────────────────────────────────────────────────
t_min  = t_hist / 60.0
d_hist = 2.0 * r_hist
d0     = 2.0 * r0

fig, ax = plt.subplots()
for j in range(n):
    ax.plot(t_min, d_hist[j] / d0[j], label=f"Particle {j+1}")
ax.set_xlabel("Time [min]")
ax.set_ylabel(r"$d / d_0$")
ax.set_title("Multi-particle microscopic swelling")
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.savefig("multi_particle_microscopic_swelling.pdf")
plt.show()
