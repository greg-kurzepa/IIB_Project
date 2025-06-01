#%%
import numpy as np
import pymc as pm
import pandas as pd
import copy
from tqdm import tqdm
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import json

import packaged_shooting._model_ivp as _model_ivp
import packaged_shooting._model_bvp as _model_bvp
from packaged_old import _pile_and_soil_nocrack
from packaged_old import _model_springs_nocrack

# Pile
# pile = _pile_and_soil.Pile(R=0.3, L=30, E=35e9)

# Define the miscellaneous variables
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
# Define a different set
# layer_type = np.array([1]) # 0 for clay, 1 for sand
# gamma_d = np.array([15e3])
# e = np.array([0.689])
# N_q = np.array([8])
# beta = np.array([x*plug_factor if layer_type[idx]==0 else x for idx, x in enumerate([0.214])])
# shaft_pressure_limit = np.array([47.8e3])
# end_pressure_limit = np.array([1.9e6])
# base_depth = np.array([pile_L])

# Depth coordinate system
z = np.linspace(0, pile_L, N)
z_midpoints = 0.5 * (z[:-1] + z[1:])

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

# for non-shooting numpy method
pile = _pile_and_soil_nocrack.Pile(R=pile_D[0]/2, L=pile_L, E=pile_E)
layer1 = _pile_and_soil_nocrack.SandLayer(
    gamma_d=gamma_d[0],
    e=e[0],
    N_q=N_q[0],
    beta=beta[0],
    shaft_pressure_limit=shaft_pressure_limit[0],
    end_pressure_limit=end_pressure_limit[0],
    base_depth=base_depth[0]
)
layer2 = _pile_and_soil_nocrack.SandLayer(
    gamma_d=gamma_d[1],
    e=e[1],
    N_q=N_q[1],
    beta=beta[1],
    shaft_pressure_limit=shaft_pressure_limit[1],
    end_pressure_limit=end_pressure_limit[1],
    base_depth=base_depth[1]
)
soil = _pile_and_soil_nocrack.Soil(layers=[layer1, layer2])

#%%

# Initial run to obtain P_ult, the failure load of the pile
P_ult = _model_ivp.forward_api(**shooting_params, u0=0).P_cap

P_over_P_ult_arr = np.linspace(0, 0.99, 10)[1:]
results_np = []
results_shooting = []
results_bvp = []
for idx, P_over_P_ult in tqdm(enumerate(P_over_P_ult_arr)):
    # Update the input parameters for the current P/P_ult value
    shooting_params["P"] = P_over_P_ult * P_ult
    
    # Call the function with the updated parameters
    res_np = _model_springs_nocrack.solve_springs4(pile, soil, P_over_P_ult * P_ult, z_w=z_w, N=N)
    res_shooting = _model_ivp.shooting_api(**shooting_params, u0_init=0)
    res_bvp  = _model_bvp.forward_api(**shooting_params)

    # append to results lists
    results_np.append(res_np)
    results_shooting.append(res_shooting)
    results_bvp.append(res_bvp)
    
    # Extract displacement from the result
    disp_np = res_np.d
    disp_shooting = res_shooting.u
    disp_bvp = res_bvp.u
    
    # Plotting
    # plt.plot(z, disp_np, label=f'simultaneous')
    # plt.plot(z, disp_shooting, label=f'shooting')
    # plt.plot(z, disp_bvp, label=f'bvp')
    # plt.title(f"$P/P_{{ult}}$ = {P_over_P_ult:.2f}")
    # plt.legend()
    # plt.show()

#%% plot forces

plt.plot(z, results_np[-1].F*1e-6, label=f'simultaneous', linestyle='--')
plt.plot(z, results_bvp[-1].F*1e-6, label=f'bvp', linestyle="-.")
plt.plot(z, results_shooting[-1].F*1e-6, label=f'shooting', linestyle=':')
plt.title("$P/P_{{ult}} = 0.99$")
plt.xlabel("Depth $z$ (m)")
plt.ylabel("Force (MN)")
plt.xlim(left=0, right=pile_L)
plt.legend()
plt.grid()
plt.show()