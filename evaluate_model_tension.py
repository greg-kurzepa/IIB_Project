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
pile_nocrack = _pile_and_soil_nocrack.Pile(R=0.3, L=30, E=pile.equivalent_compressive_E)
print(f"{pile_nocrack.E:.4e}")

# Define the other variables (loading, number of elements, model noise variance)
P = -1.8e6 # top axial load
N = 100 # number of nodes along pile
z = np.linspace(0, pile.L, N)
z_midpoints = 0.5 * (z[:-1] + z[1:])

# according to API, beta for soils should be scaled by 1.25 for plugged piles
plug_factor=1.25

#%%

# in this test, there are two soil layers. each has its own uniform set of ground truth parameters
# Soil 1
gamma_d_l = np.array([15000, 17000])
beta_l = np.array([0.2675, 0.575])
layer1 = _pile_and_soil.SandLayer(gamma_d=gamma_d_l[0], e=0.689, N_q=8, beta=beta_l[0], shaft_pressure_limit=47.8e3, end_pressure_limit=1.9e6, base_depth=12.5)
layer2 = _pile_and_soil.SandLayer(gamma_d=gamma_d_l[1], e=0.441, N_q=40, beta=beta_l[1], shaft_pressure_limit=96e3, end_pressure_limit=10e6, base_depth=pile.L)
soil = _pile_and_soil.Soil([layer1, layer2])

#%%
res = _model_springs.solve_springs4(pile, soil, P, 3, N, do_Afck=True)
#%%
res2 = _model_springs.solve_springs4(pile, soil, 0.001*P, 3, N, do_Afck=False)
#%%
res3 = _model_springs_nocrack.solve_springs4(pile_nocrack, soil, P, 3, N)

#%%

plt.plot(z, res.F, label="springs")
plt.plot(z, res2.F, label="springs_nocrack", linestyle="--")
plt.plot(z, res3.F, label="springs_uniform", linestyle=":")
plt.title("Force")
if P < 0: plt.gca().invert_yaxis()
plt.grid()
plt.legend()
plt.show()

#%%

plt.plot(z, res.strain, label="springs")
plt.plot(z, res2.strain, label="springs_nocrack", linestyle="--")
plt.plot(z, res3.strain, label="springs_uniform", linestyle=":")
plt.title("Strain")
if P < 0: plt.gca().invert_yaxis()
plt.grid()
plt.legend()
plt.show()

#%%

plt.plot(z, res.d, label="springs")
plt.plot(z, res2.d, label="springs_nocrack", linestyle="--")
plt.plot(z, res3.d, label="springs_uniform", linestyle=":")
plt.title("Displacement")
if P < 0: plt.gca().invert_yaxis()
plt.grid()
plt.legend()
plt.show()

#%% See if shear stresses go above soil limits

plt.plot(z_midpoints, res.shaft_pressure_limit, linestyle="--", c="b", label="tau limit")
plt.plot(z_midpoints, -res.shaft_pressure_limit, linestyle="--", c="b")
plt.plot(z_midpoints, res.tau_ult, linestyle="--", c="r", label="tau ult")
plt.plot(z_midpoints, -res.tau_ult, linestyle="--", c="r")
plt.plot(z_midpoints, res.tau, c="g", label="shear stress")
if P < 0: plt.gca().invert_yaxis()
plt.grid()
plt.legend()
plt.show()

#%%

plt.plot(pile.F_over_AEs(res.strain, res.d, nocrack=True), label="F/AEs")
if P < 0: plt.gca().invert_yaxis()
plt.grid()
plt.legend()
plt.show()
# %%

import pickle
import time
timenow = time.strftime("%Y-%m-%d_%H-%M-%S")
with open(f"results\\SolveData_{timenow}.pkl", "wb") as f:
    pickle.dump(res, f, pickle.HIGHEST_PROTOCOL)
# %%

import pandas as pd

random_seed = 716743
rng = np.random.default_rng(random_seed)

sigma = 0.05 * P
obs = res.F + sigma * rng.normal(size=N)
df_forces = pd.DataFrame({"z":z, "True Force":res.F, "Observed Force":obs})
df_forces.to_csv("results\\forces_B.csv")
# %%
