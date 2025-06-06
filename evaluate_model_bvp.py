#%%
import numpy as np
import pymc as pm
import pandas as pd
import copy
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
from matplotlib.lines import Line2D

from packaged_shooting._model_ivp import shooting_api
from packaged_shooting._model_bvp import forward_api as bvp_api

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
beta = plug_factor*np.array([0.214, 0.46])
shaft_pressure_limit = np.array([47.8e3, 96e3])
end_pressure_limit = np.array([1.9e6, 10e6])
base_depth = np.array([12.5, pile_L])

# gamma_d = np.array([11441.96410478, 15184.46566306])
# e = np.array([0.95088726, 0.28129699])
# beta = np.array([0.26122883, 0.65903567])
# shaft_pressure_limit = np.array([14170.9607232, 134981.95198728])

# gamma_d = np.array([18605.37548914, 14776.27071572])
# e = np.array([0.46125496, 0.24476925])
# beta = np.array([0.18417854, 0.57434449])
# shaft_pressure_limit = np.array([35543.22431393, 55603.63328634])

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

res = shooting_api(u0_init=0, **shooting_params)
res2 = bvp_api(**shooting_params)
print(f"P_cap: {res.P_cap:.4e}, P_cap2: {res2.P_cap:.4e}")

#%%

for i in range(100):
    shooting_params["l_gamma_d"] += np.random.normal(0, 1e-8)
    shooting_params["l_e"] += np.random.normal(0, 1e-8)
    shooting_params["l_c2"] += np.random.normal(0, 1e-8)
    shooting_params["l_shaft_pressure_limit"] += np.random.normal(0, 1e-8)

    print(f"gamma_d: {shooting_params['l_gamma_d']}, e: {shooting_params['l_e']}, c2: {shooting_params['l_c2']}, shaft_pressure_limit: {shooting_params['l_shaft_pressure_limit']}")

    res2 = bvp_api(**shooting_params, max_nodes=200)
    print("strain", res2.strain[-1])

    print("\n")

#%%

legend_elements = [Line2D([0], [0], color='steelblue', ls='--',lw=2, label='API'),
                   Line2D([0], [0], color='k', ls='-', lw=2, label='Theoretical (Poulos)'),
                   Line2D([0], [0], color='k', marker="x", lw=0, label='Empirical (Poulos)')]

# plt.plot(z, res.F, label='Shooting API')
plt.plot(z, res2.F, label='BVP API', linestyle='--', color='steelblue')
plt.xlabel('Depth $z$ (m)')
plt.ylabel('Axial Force $F$ (N)')
plt.xlim(0,pile_L)
plt.ylim(bottom=0)
plt.legend(handles=legend_elements)
plt.grid()
plt.savefig("results\\axial_force_profile_3,6.png", dpi=300, bbox_inches='tight')
plt.show()

# %%

plt.plot(z, res.strain*1e6, label='Shooting API')
plt.plot(z, res2.strain*1e6, label='BVP API', linestyle='--')
plt.xlabel('Depth (m)')
plt.ylabel('Microstrains')
plt.xlim(0,pile_L)
plt.legend()
plt.grid()
plt.show()

#%%

import pickle
import time
import pandas as pd

timenow = time.strftime("%Y-%m-%d_%H-%M-%S")
with open(f"results\\SolveData_{timenow}.pkl", "wb") as f:
    pickle.dump(res2, f, pickle.HIGHEST_PROTOCOL)

random_seed = 716743
rng = np.random.default_rng(random_seed)

sigma = 20e-6
obs = res2.strain + sigma * rng.normal(size=N)
df_forces = pd.DataFrame({"z":z, "True Strain":res2.strain, "Observed Strain":obs})
df_forces.to_csv("results\\strains_B.csv")

#%%

plt.plot(z, res2.strain*1e6, label='BVP API')
plt.plot(z, obs*1e6, label='noisy', linestyle='--')
plt.xlabel('Depth (m)')
plt.ylabel('Microstrains')
plt.xlim(0,pile_L)
plt.legend()
plt.grid()
plt.show()

#%%