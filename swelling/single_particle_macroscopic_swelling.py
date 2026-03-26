import numpy as np
import matplotlib.pyplot as plt

# ── Physical parameters ────────────────────────────────────────────────────────
rho_s   = 1486.0      # solid (dry) density              [kg/m^3]
rho_w   = 1000.0      # water density                    [kg/m^3]
c_wl    = 992.0       # water concentration in liquid    [kg/m^3]
D_0     = 4e-7        # reference diffusion coefficient  [m^2/s]
delta   = 1.81        # diffusion sensitivity parameter  [-]
c_max   = c_wl        # maximum concentration            [kg/m^3]

r0      = 0.95e-3     # initial particle radius          [m]
V_p0    = (4.0 / 3.0) * np.pi * r0**3
m_p0    = rho_s * V_p0

# ── Time integration ───────────────────────────────────────────────────────────
dt      = 1.0
t_end   = 300.0 * 60   # 300 minutes
steps   = int(t_end / dt)

m_w = 0.0;  V_p = V_p0;  r_p = r0;  m_p = m_p0

t_hist = np.zeros(steps + 1)
r_hist = np.zeros(steps + 1)
t_hist[0] = 0.0;  r_hist[0] = r_p

for i in range(steps):
    S_p     = 4.0 * np.pi * r_p**2
    c_wp    = (m_p - m_p0) / V_p
    D       = D_0 * np.exp(-delta * c_wp / c_max)
    m_w_dot = S_p * D * (c_wl - c_wp) / r_p

    m_w    += m_w_dot * dt
    m_p     = m_p0 + m_w
    V_p     = V_p0 + m_w / rho_w
    r_p     = (3.0 * V_p / (4.0 * np.pi))**(1.0 / 3.0)

    t_hist[i+1] = (i + 1) * dt
    r_hist[i+1] = r_p

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots()
ax.plot(t_hist / 60.0, 2.0 * r_hist / (2.0 * r0))
ax.set_xlabel("Time [min]")
ax.set_ylabel(r"$d / d_0$")
ax.set_title("Single particle macroscopic swelling")
ax.grid(True)
plt.tight_layout()
plt.savefig("single_particle_macroscopic_swelling.pdf")
plt.show()
