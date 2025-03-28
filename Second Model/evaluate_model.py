#%%
import numpy as np
import pymc as pm
import pandas as pd
import copy
from tqdm import tqdm
import matplotlib.pyplot as plt

import packaged._pile_and_soil as _pile_and_soil
import packaged._model_springs as _model_springs

# Pile
pile = _pile_and_soil.Pile(R=0.3, L=30, E=35e9)

# Define the other variables (loading, number of elements, model noise variance)
P = 1.8e6 # top axial load
N = 100 # number of nodes along pile
z = np.linspace(0, pile.L, N)
z_midpoints = 0.5 * (z[:-1] + z[1:])

# according to API, beta for soils should be scaled by 1.25 for plugged piles
plug_factor=1

#%%

# in this test, there are two soil layers. each has its own uniform set of ground truth parameters
# Soil 1
layer1 = _pile_and_soil.SandLayer(gamma_d=15e3, e=0.689, N_q=8, beta=plug_factor*0.214, shaft_friction_limit=47.8e3, end_bearing_limit=1.9e6, base_depth=12.5)
layer2 = _pile_and_soil.SandLayer(gamma_d=17e3, e=0.441, N_q=40, beta=plug_factor*0.46, shaft_friction_limit=96e3, end_bearing_limit=10e6, base_depth=pile.L)
soil = _pile_and_soil.Soil([layer1, layer2])
res1 = _model_springs.solve_springs4(pile, soil, P, 3, N)
# Soil 2
layer1 = _pile_and_soil.SandLayer(gamma_d=15e3, e=0.689, N_q=8, beta=plug_factor*0.214, shaft_friction_limit=47.8e3, end_bearing_limit=1.9e6, base_depth=12.5)
layer2 = _pile_and_soil.SandLayer(gamma_d=17e3, e=0.441, N_q=40, beta=plug_factor*0.46, shaft_friction_limit=96e3, end_bearing_limit=10e6, base_depth=pile.L)
soil2 = _pile_and_soil.Soil([layer1, layer2])
res2 = _model_springs.solve_springs4(pile, soil2, P, 3, N)

#%%

plt.plot(res1[0], z, label="1")
plt.plot(res2[0], z, label="2")
plt.xlim(left=0)
plt.gca().invert_yaxis()
plt.xlabel("F")
plt.ylabel("z")
plt.legend()
plt.grid()
plt.show()

#%%
plt.plot(res1[2], z, label="1")
plt.plot(res2[2], z, label="2")
plt.gca().invert_yaxis()
plt.xlabel("d")
plt.ylabel("z")
plt.legend()
plt.grid()
plt.show()
# %%

# testing head and tip settlement
P_array = np.linspace(0.1e6,6e6,20)
tip_settlements = []
head_settlements = []
for P in tqdm(P_array):
    tip_settlements.append(_model_springs.solve_springs4(pile, soil2, P, 3, N)[2][-1])
    head_settlements.append(_model_springs.solve_springs4(pile, soil2, P, 3, N)[2][0])

#%%
plt.plot(tip_settlements, P_array, label="tip settlement")
plt.grid()
plt.xlim(left=0, right=0.001)
plt.ylim(bottom=0)
plt.xlabel("tip settlement")
plt.ylabel("P")
plt.show()
plt.plot(head_settlements, P_array, label="head settlement")
plt.grid()
plt.xlim(left=0, right=0.025)
plt.ylim(bottom=0)
plt.xlabel("head settlement")
plt.ylabel("P")
plt.show()
#%%

# plot shear force profile with z and the shear limit. Also print the end bearing force and the end bearing limit.
tau, Q = res2[5], res2[6]
Q_over_Q_ult = _model_springs.Q_over_Q_ult(res2[2][-1])
print(f"Q_over_Q_ult: {Q_over_Q_ult:.4e}, Q: {Q:.4e}, end bearing limit: {soil2.layers[0].end_bearing_limit:.4e}")

tau_over_tau_ult = _model_springs.tau_over_tau_ult_sand(res2[3])

fig, ax = plt.subplots(1,2)
plt.tight_layout()
ax[0].plot(z_midpoints, tau_over_tau_ult)
ax[0].set_title("tau_over_tau_ult")
ax[1].plot(z_midpoints, tau)
ax[1].set_title("tau")
# plot lines of shear limits
prev_depth = 0
for layer in soil2.layers:
    if layer.layer_type == "sand":
        sfl = layer.shaft_friction_limit
        depth = layer.base_depth
        ax[1].plot([prev_depth, depth], [sfl, sfl], color="red", linestyle="--")
        ax[1].plot([prev_depth, depth], [-sfl, -sfl], color="red", linestyle="--")
        prev_depth = depth
    else:
        prev_depth = layer.base_depth
plt.show()

#%%