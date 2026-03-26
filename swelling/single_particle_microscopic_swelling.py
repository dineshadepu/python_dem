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

r0      = 0.95e-3     # initial particle radius          [m]

# ── Derived initial quantities ─────────────────────────────────────────────────
V_p0    = (4.0 / 3.0) * np.pi * r0**3
m_p0    = rho_s * V_p0

# ── Time integration setup ─────────────────────────────────────────────────────
dt      = 1.0
t_end   = 300.0 * 60
steps   = int(t_end / dt)

# ── State variables ────────────────────────────────────────────────────────────
m_w     = 0.0
r_p     = r0
r_c     = r0          # core radius starts as full particle radius
V_c     = V_p0        # core volume
V_s     = 0.0         # swollen zone volume (initially zero)
V_p     = V_p0
m_p     = m_p0
core_present = True

# ── Storage ───────────────────────────────────────────────────────────────────
t_hist  = np.zeros(steps + 1)
r_hist  = np.zeros(steps + 1)
t_hist[0] = 0.0
r_hist[0] = r_p

# ── Time loop ──────────────────────────────────────────────────────────────────
for i in range(steps):
    S_p = 4.0 * np.pi * r_p**2

    if core_present:
        # Concentration in swollen zone only (avoid /0 at t=0)
        c_wp = (m_p - m_p0) / V_s if V_s > 0.0 else 0.0

        D       = D_0 * np.exp(-delta * c_wp / c_max)
        m_w_dot = S_p * D * (c_wl - c_wp) / r_p

        m_w  += m_w_dot * dt
        m_p   = m_p0 + m_w

        # Volume from diffusion
        dV_d = m_w_dot * dt / rho_w

        # Core shrinks inward
        r_c_old = r_c
        r_c     = r_c - v_0 * dt

        if r_c <= 0.0:
            # Core just disappeared — absorb remaining core volume
            dV_c = (4.0 / 3.0) * np.pi * r_c_old**3
            r_c  = 0.0
            V_c  = 0.0
            core_present = False
        else:
            V_c  = (4.0 / 3.0) * np.pi * r_c**3
            dV_c = (4.0 / 3.0) * np.pi * (r_c_old**3 - r_c**3)

        V_s += dV_d + dV_c
        V_p  = V_s + V_c

    else:
        # ── Macroscopic model once core is gone ────────────────────────────
        c_wp    = (m_p - m_p0) / V_p
        D       = D_0 * np.exp(-delta * c_wp / c_max)
        m_w_dot = S_p * D * (c_wl - c_wp) / r_p

        m_w    += m_w_dot * dt
        m_p     = m_p0 + m_w
        V_p     = V_p0 + m_w / rho_w

    r_p = (3.0 * V_p / (4.0 * np.pi))**(1.0 / 3.0)

    t_hist[i+1] = (i + 1) * dt
    r_hist[i+1] = r_p

# ── Plot ───────────────────────────────────────────────────────────────────────
t_min = t_hist / 60.0
d_d0  = 2.0 * r_hist / (2.0 * r0)

fig, ax = plt.subplots()
ax.plot(t_min, d_d0, 'g-', label="Microscopic model (our simulation)")
ax.set_xlabel("Time [min]")
ax.set_ylabel(r"$d / d_0$")
ax.set_title("Single particle — microscopic swelling model")
ax.set_xlim(0, 300)
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.savefig("single_particle_microscopic_swelling.pdf")
plt.show()
