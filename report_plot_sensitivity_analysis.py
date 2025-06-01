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

import packaged_shooting._model_bvp as _model_bvp

# Pile
P = 4.6e6 # top axial load
N = 200 # number of nodes along pile
z_w = 3 # water table depth in m

# Define the pile variables
pile_D = np.full(N, 0.6)
pile_L = 30
pile_E = 35e9

# according to API, beta for sands should be scaled by 1.25 for plugged piles
plug_factor=1.25
# Define the soil variables
layer_type = np.array([1, 1]) # 0 for clay, 1 for sand
gamma_d = np.array([15e3, 17e3])
# gamma_d = np.array([19912.16215559, 13379.54924189])
# gamma_d = np.array([19912.162155588052, 13379.549241894718])
e = np.array([0.689, 0.441])
N_q = np.array([8.0, 40.0])
# beta = np.array([x*plug_factor if layer_type[idx]==0 else x for idx, x in enumerate([0.214, 0.46])])
beta = np.array([0.2675, 0.575])
shaft_pressure_limit = np.array([47.8e3, 96e3])
end_pressure_limit = np.array([1.9e6, 10e6])
base_depth = np.array([12.5, pile_L])

z = np.linspace(0, pile_L, N)

shooting_params = {
    "N": N,
    "P": P,
    "z_w": z_w,
    "pile_D": np.array(pile_D),
    "pile_L": pile_L,
    "pile_E": pile_E,
    "l_layer_type": np.array(layer_type),
    "l_gamma_d": np.array(gamma_d),
    "l_e": np.array(e),
    "l_c1": np.array(N_q),
    "l_c2": np.array(beta),
    "l_shaft_pressure_limit": np.array(shaft_pressure_limit),
    "l_end_pressure_limit": np.array(end_pressure_limit),
    "l_base_depth": np.array(base_depth),
}

#%%

fig, ax = plt.subplots(3, 2, figsize=(10, 15), sharex=True, sharey=True)
ax = ax.flatten()

cp = 0.25 # change proportion for each parameter

for i in range(6):
    params = copy.copy(shooting_params)
    res = _model_bvp.forward_api(**params)
    ax[i].plot(z, res.strain*1e6, label="Original")

    for j in range(2):
        if i == 0:
            ax[i].set_title("Sensitivity to $\\gamma_d$")
            params["l_gamma_d"] *= (1+cp if j == 0 else 1-cp)
        if i == 1:
            ax[i].set_title("Sensitivity to $e$")
            params["l_e"] *= (1+cp if j == 0 else 1-cp)
        if i == 2:
            ax[i].set_title("Sensitivity to $N_q$")
            params["l_c1"] *= (1+cp if j == 0 else 1-cp)
        if i == 3:
            ax[i].set_title("Sensitivity to $\\beta$")
            params["l_c2"] *= (1+cp if j == 0 else 1-cp)
        if i == 4:
            ax[i].set_title("Sensitivity to $\\tau_{{cap}}$")
            params["l_shaft_pressure_limit"] *= (1+cp if j == 0 else 1-cp)
        if i == 5:
            ax[i].set_title("Sensitivity to $Q_{{cap}}$")
            params["l_end_pressure_limit"] *= (1+cp if j == 0 else 1-cp)

        res = _model_bvp.forward_api(**params)
        ax[i].plot(z, res.strain*1e6,
                   linestyle = "--" if j == 0 else ":",
                   label=f"Increase by {100*cp:.0f}%" if j == 0 else f"Decrease by {100*cp:.0f}%")

    if i % 3 == 0: ax[i].set_ylabel("Microstrains")
    ax[i].set_xlabel("Depth $z$ (m)")
    ax[i].grid()
    ax[i].set_xlim(0, pile_L)
    if i == 0: ax[i].legend()

plt.tight_layout()
plt.subplots_adjust(hspace=0.3)
plt.show()

#%%