import numpy as np
import matplotlib.pyplot as plt

# geometry
L = 1.0
N = 50
dx = L/(N-1)
x = np.linspace(0,L,N)

# material
E = 200e9
delta = 3*dx
volume = dx

c = E/delta

# stiffness matrix
K = np.zeros((N,N))

# neighbor list
neighbors = []

for i in range(N):
    for j in range(N):
        if i==j: continue
        if abs(x[j]-x[i]) < delta:

            k = c*volume

            K[i,i] += k
            K[i,j] -= k

# displacement BC
U = 1e-5

# RHS
f = np.zeros(N)

# apply BC
K[0,:] = 0
K[0,0] = 1
f[0] = 0

K[-1,:] = 0
K[-1,-1] = 1
f[-1] = U

# solve
u = np.linalg.solve(K,f)

# analytical
u_exact = U*x/L

plt.plot(x,u,label="PD")
plt.plot(x,u_exact,'--',label="Analytical")
plt.legend()
plt.xlabel("x")
plt.ylabel("displacement")
plt.show()
