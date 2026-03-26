import numpy as np
import matplotlib.pyplot as plt

D_0 = 1.0
delta = 1.81
c_max = 1000.0

c = np.linspace(0, c_max, 500)
D = D_0 * np.exp(-delta * c / c_max)

fig, ax = plt.subplots()
ax.plot(c, D)
ax.set_xlabel("c (concentration)")
ax.set_ylabel("D / D_0")
ax.set_title(r"$D = D_0 \, e^{-\delta \, c / c_{\max}}$, $\delta = 1.81$, $c_{\max} = 1000$")
ax.grid(True)
plt.tight_layout()
plt.savefig("diffusion_coefficient_variation.pdf")
plt.show()
