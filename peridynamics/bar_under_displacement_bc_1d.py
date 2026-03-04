import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Geometry
# -----------------------------

L = 1.0
N = 50
dx = L/(N-1)

x = np.linspace(0,L,N)

# -----------------------------
# Material
# -----------------------------

E = 200e9
rho = 7800
delta = 3*dx
volume = dx

c = E/delta

# -----------------------------
# Time parameters
# -----------------------------

dt = 0.2*np.sqrt(rho*dx/E)
steps = 20000

# -----------------------------
# Boundary condition
# -----------------------------

U = 1e-5

# -----------------------------
# State variables
# -----------------------------

u = np.zeros(N)
v = np.zeros(N)
F = np.zeros(N)

# -----------------------------
# Neighbor list
# -----------------------------

neighbors = []

for i in range(N):
    neigh=[]
    for j in range(N):
        if i==j: continue
        if abs(x[j]-x[i]) < delta:
            neigh.append(j)
    neighbors.append(neigh)

# -----------------------------
# Time integration
# -----------------------------

for step in range(steps):

    # displacement BC
    u[0] = 0
    v[0] = 0

    u[-1] = U
    v[-1] = 0

    F[:] = 0

    for i in range(N):
        for j in neighbors[i]:

            xi = x[i]
            xj = x[j]

            yi = xi + u[i]
            yj = xj + u[j]

            r0 = abs(xj-xi)
            r  = abs(yj-yi)

            stretch = r - r0
            direction = np.sign(yj-yi)

            fij = c * stretch * direction * volume

            F[i] += fij

    # integrate
    a = F/(rho*volume)
    v += a*dt
    u += v*dt

    # damping
    v *= 0.01

    print("max velocity:", np.max(np.abs(v)))


# -----------------------------
# Analytical solution
# -----------------------------

u_exact = U*x/L

strain = np.diff(u)/dx
print(strain)

# -----------------------------
# Plot
# -----------------------------

plt.plot(x,u,label="Peridynamics")
plt.plot(x,u_exact,'--',label="Analytical")
plt.xlabel("x")
plt.ylabel("displacement")
plt.legend()
plt.show()

# plt.plot(x[:-1], strain)
# plt.xlabel("x")
# plt.ylabel("strain")
# plt.show()
