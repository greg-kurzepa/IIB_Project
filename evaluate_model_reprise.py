#%%
import numpy as np
import pymc as pm
import pandas as pd
import copy
from tqdm import tqdm
import matplotlib.pyplot as plt
import json

import packaged._pile_and_soil as _pile_and_soil
import packaged._model_springs as _model_springs

import packaged_old._pile_and_soil_nocrack as _pile_and_soil_nocrack
import packaged_old._model_springs_nocrack as _model_springs_nocrack

# Pile
pile = _pile_and_soil.Pile(R=0.3, L=30)
pile_nocrack = _pile_and_soil_nocrack.Pile(R=0.3, L=30, E=35e9)
print(f"{pile_nocrack.E:.4e}")

# Define the other variables (loading, number of elements, model noise variance)
P = 3.6e6 # top axial load
N = 100 # number of nodes along pile
z = np.linspace(0, pile.L, N)
z_midpoints = 0.5 * (z[:-1] + z[1:])

# according to API, beta for soils should be scaled by 1.25 for plugged piles
plug_factor=1.25

#%%

# in this test, there are two soil layers. each has its own uniform set of ground truth parameters
# Soil 1
# gamma_d_l = np.array([19912.162155588052, 13379.549241894718])
gamma_d_l = np.array([15000, 17000])
beta_l = np.array([0.2675, 0.575])
layer1 = _pile_and_soil.SandLayer(gamma_d=gamma_d_l[0], e=0.689, N_q=8, beta=beta_l[0], shaft_pressure_limit=47.8e3, end_pressure_limit=1.9e6, base_depth=12.5)
layer2 = _pile_and_soil.SandLayer(gamma_d=gamma_d_l[1], e=0.441, N_q=40, beta=beta_l[1], shaft_pressure_limit=96e3, end_pressure_limit=10e6, base_depth=pile.L)
soil = _pile_and_soil.Soil([layer1, layer2])

res = _model_springs.solve_springs4(pile, soil, P, 3, N)
res2 = _model_springs_nocrack.solve_springs4(pile_nocrack, soil, 4.6e6, 3, N)

#%%

plt.plot(z, res.F, label="1")
plt.plot(z, res2.F, label="2", linestyle="--")
plt.title("Force")
plt.grid()
plt.legend()
plt.show()

#%%

plt.plot(z, res.d, label="1")
plt.plot(z, res2.d, label="2", linestyle="--")
plt.title("Displacement")
plt.grid()
plt.legend()
plt.show()

#%%

plt.plot(z, res.strain, label="1")
plt.plot(z, res2.strain, label="2", linestyle="--")
plt.title("Strain")
plt.grid()
plt.legend()
plt.show()

#%%

#%% plot a nice profile of tau_ult, tau_cap and tau

fig, ax = plt.subplots(1,2 , figsize=(12, 6), sharey=True, sharex=True)

ax[0].plot(z_midpoints, res.tau_ult/1e6, label="$\\tau_{{ult}}$", linestyle="--", color="blue")
ax[0].plot(z_midpoints, res.shaft_pressure_limit/1e6, label="$\\tau_{{cap}}$", linestyle=":", color="red")
ax[0].plot(z_midpoints, -res.tau_ult/1e6, linestyle="--", color="blue")
ax[0].plot(z_midpoints, -res.shaft_pressure_limit/1e6, linestyle=":", color="red")
ax[0].plot(z_midpoints, res.tau/1e6, color="green", label="Mobilised Shear Stress $\\tau$")
ax[0].set_xlabel("Depth $z$ (m)")
ax[0].set_ylabel("Shear Stress (MPa)")
ax[0].set_title("$P=3.6$MN")
ax[0].legend()
ax[0].grid()

ax[1].plot(z_midpoints, res2.tau_ult/1e6, label="$\\tau_{{ult}}$", linestyle="--", color="blue")
ax[1].plot(z_midpoints, res2.shaft_pressure_limit/1e6, label="$\\tau_{{cap}}$", linestyle=":", color="red")
ax[1].plot(z_midpoints, -res2.tau_ult/1e6, linestyle="--", color="blue")
ax[1].plot(z_midpoints, -res2.shaft_pressure_limit/1e6, linestyle=":", color="red")
ax[1].plot(z_midpoints, res2.tau/1e6, color="green", label="Mobilised Shear Stress $\\tau$")
ax[1].set_xlabel("Depth $z$ (m)")
# ax[1].set_ylabel("Shear Stress (MPa)")
ax[1].set_title("$P=4.6$MN")
# ax[1].legend()
ax[1].grid()

plt.tight_layout()
plt.show()

#%% plot effective stress profile

plt.plot(z_midpoints, res.eff_stress/1e6)
plt.xlabel("Depth $z$ (m)")
plt.ylabel("Effective Stress (MPa)")
plt.grid()

plt.tight_layout()
plt.show()

#%%

import pickle
import time
timenow = time.strftime("%Y-%m-%d_%H-%M-%S")
with open(f"results\\SolveData_{timenow}.pkl", "wb") as f:
    pickle.dump(res, f, pickle.HIGHEST_PROTOCOL)
# %%

import pandas as pd
res=res2

random_seed = 716743
rng = np.random.default_rng(random_seed)

sigma = 20e-6
obs = res.strain + sigma * rng.normal(size=N)
df_forces = pd.DataFrame({"z":z, "True Strain":res.strain, "Observed Strain":obs})
df_forces.to_csv("results\\strains_B.csv")
# %%

plt.plot(z, res.strain)
plt.scatter(z, obs, color='red', label='Observed')
plt.show()

#%%