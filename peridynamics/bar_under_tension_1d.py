import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Material and geometry
# -----------------------------

L = 1.0
N = 50
dx = L/(N-1)

E = 200e9
A = 1.0
rho = 7800

delta = 3.01*dx
# dt = 1e-6
dt = 0.2 * np.sqrt(rho * dx / E)
print("dt is", dt)
steps = 2000

P = 1e6

# -----------------------------
# Particle arrays
# -----------------------------

x = np.linspace(0,L,N)
u = np.zeros(N)
v = np.zeros(N)
F = np.zeros(N)

volume = dx*A

# bond stiffness for 1D PD
# c = 2*E*A/delta**2.
c = E*A/delta

# -----------------------------
# neighbor list
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
# Normalization
# -----------------------------
m = np.zeros(N)

for i in range(N):
    for j in neighbors[i]:
        r0 = abs(x[j] - x[i])
        m[i] += r0**2 * volume
print(m)


# -----------------------------
# time stepping
# -----------------------------

for step in range(steps):

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
            # F[i] += fij / m[i]

    # external force
    F[-1] += P

    # integrate
    a = F/(rho*volume)
    v += a*dt
    v *= 0.999
    u += v*dt

    # fixed boundary
    # F[0] = 0
    u[0] = 0
    v[0] = 0
    # print(np.sum(F), "Force sum")

# -----------------------------
# analytical solution
# -----------------------------
print(len(neighbors[0]), len(neighbors[N//2]))
u_exact = P*x/(E*A)

# -----------------------------
# plot
# -----------------------------

plt.plot(x,u,label="Peridynamics")
plt.plot(x,u_exact,'--',label="Analytical")
plt.xlabel("x")
plt.ylabel("displacement")
plt.legend()
plt.show()
