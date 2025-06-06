#%%
import numpy as np
import pymc as pm
import pandas as pd
import copy
from tqdm import tqdm
import matplotlib.pyplot as plt
import json

from packaged_shooting import _model_bvp

P = 3.6e6 # top axial load
N = 100 # number of nodes along pile
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
e = np.array([0.689, 0.441])
N_q = np.array([8, 40])
beta = np.array([x*plug_factor if layer_type[idx]==0 else x for idx, x in enumerate([0.214, 0.46])])
shaft_pressure_limit = np.array([47.8e3, 96e3])
end_pressure_limit = np.array([1.9e6, 10e6])
base_depth = np.array([12.5, pile_L])

bvp_params = {
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

P_ult = _model_bvp.forward_api(**bvp_params).P_cap
print("P_ult", P_ult/1e6, "MN")

#%%

forces = np.linspace(0, 0.99*P_ult, 50)[1:]

#%%

head_settlements = []
base_settlements = []
for force in tqdm(forces):
    bvp_params["P"] = force
    res = _model_bvp.forward_api(**bvp_params)
    head_settlements.append(res.u[0])
    base_settlements.append(res.u[-1])

#%% 

from matplotlib.lines import Line2D

legend_elements = [Line2D([0], [0], color='steelblue', ls='--',lw=2, label='API'),
                   Line2D([0], [0], color='k', ls=':',lw=2, label='P_ult'),
                   Line2D([0], [0], color='k', ls='-', lw=2, label='Theoretical (Poulos)'),
                   Line2D([0], [0], color='k', marker="x", lw=0, label='Empirical (Poulos)')]

fig, ax = plt.subplots(1,2, figsize=(12, 6))

ax[0].plot(np.array(head_settlements)*1000, forces/1e6, linestyle="--", label="API")
ax[0].set_xlabel("Head Settlement (mm)")
ax[0].set_ylabel("Force (MN)")
ax[0].axhline(P_ult/1e6, xmin=0, xmax=20, color="k", linestyle=":", label="P_ult")
ax[0].set_xlim(0,20)
# ax[0].legend()
ax[0].grid()

ax[1].plot(np.array(base_settlements)*1000, forces/1e6, linestyle="--", label="API")
ax[1].set_xlabel("Base Settlement (mm)")
ax[1].set_ylabel("Force (MN)")
ax[1].axhline(P_ult/1e6, xmin=0, xmax=8, color="k", linestyle=":", label="P_ult")
ax[1].set_xlim(0,8)
ax[1].legend(handles=legend_elements)
ax[1].grid()

plt.savefig("report_poulos_comparison_settlement.png", dpi=300, bbox_inches='tight')

plt.show()

#%%