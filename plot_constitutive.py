#%%
import numpy as np
import matplotlib.pyplot as plt

import packaged._pile_and_soil as _pile_and_soil

pile = _pile_and_soil.Pile(R=0.6, L=30, reinforcement_ratio=0.023)

strain = np.linspace(-0.001, 0.001, 10000)
force1 = pile.steel_E * pile.A * pile.F_over_AEs(strain, None, do_tension_concrete=True)
force2 = pile.steel_E * pile.A * pile.F_over_AEs(strain, None, do_tension_concrete=False)

# plot the stress-strain curve
plt.plot(strain, force1/1000, label="Concrete has some tensile strength")
plt.plot(strain, force2/1000, label="Concrete has no tensile strength", linestyle="--")
# plt.scatter(strain, stress)
plt.xlabel("Strain")
plt.ylabel("Axial Force(kN)")
plt.title("Stress-Strain Curve for Pile")
plt.legend()
plt.grid()
plt.show()
# %%
