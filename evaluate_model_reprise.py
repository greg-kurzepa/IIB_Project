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

import packaged._pile_and_soil as _pile_and_soil
import packaged._model_springs as _model_springs
import packaged._model_springs_jax as _model_springs_jax

jax.config.update('jax_enable_x64', True)

# Pile
pile = _pile_and_soil.Pile(R=0.3, L=30, E=35e9)

# Define the other variables (loading, number of elements, model noise variance)
P = 1.8e6 # top axial load
N = 100 # number of nodes along pile
z = np.linspace(0, pile.L, N)
z_midpoints = 0.5 * (z[:-1] + z[1:])

# according to API, beta for soils should be scaled by 1.25 for plugged piles
plug_factor=1.25

# P_cap = 6675646.359033609
# P_over_P_cap = 0.99
# P = P_over_P_cap * P_cap

#%%

# in this test, there are two soil layers. each has its own uniform set of ground truth parameters
# Soil 1
# gamma_d_l = np.array([19912.162155588052, 13379.549241894718])
gamma_d_l = np.array([15000, 17000])
beta_l = np.array([0.2675, 0.5750000000000001])
layer1 = _pile_and_soil.SandLayer(gamma_d=gamma_d_l[0], e=0.689, N_q=8, beta=beta_l[0], shaft_pressure_limit=47.8e3, end_pressure_limit=1.9e6, base_depth=12.5)
layer2 = _pile_and_soil.SandLayer(gamma_d=gamma_d_l[1], e=0.441, N_q=40, beta=beta_l[1], shaft_pressure_limit=96e3, end_pressure_limit=10e6, base_depth=pile.L)
soil = _pile_and_soil.Soil([layer1, layer2])

res = _model_springs.solve_springs4(pile, soil, P, 3, N)
# print(res[0])

#%%

# construct values needed for jax version
jax_pile_D = jnp.full_like(z, pile.D)
jax_pile_L = pile.L
jax_pile_E = pile.E
jax_layer_type = jnp.array([0 if layer.layer_type=="clay" else 1 for layer in soil.layers])
jax_gamma_d = jnp.array([layer.gamma_d for layer in soil.layers])
jax_e = jnp.array([layer.e for layer in soil.layers])
jax_c1 = jnp.array([layer.N_c if layer.layer_type=="clay" else layer.N_q for layer in soil.layers])
jax_c2 = jnp.array([layer.psi if layer.layer_type=="clay" else layer.beta for layer in soil.layers])
jax_shaft_pressure_limit = jnp.array([layer.shaft_pressure_limit for layer in soil.layers])
jax_end_pressure_limit = jnp.array([layer.end_pressure_limit for layer in soil.layers])
jax_base_depth = jnp.array([layer.base_depth for layer in soil.layers])

input_params = (jax_pile_D, jax_pile_L, jax_pile_E, jax_layer_type, jax_gamma_d, jax_e, jax_c1, jax_c2, jax_shaft_pressure_limit, jax_end_pressure_limit, jax_base_depth, P, 3, N)
res_jax = _model_springs_jax.solve_springs_api_jax(*input_params, throw=False)

#%%

plt.plot(z, res[0], label="springs")
plt.plot(z, res_jax[0], label="springs_jax", linestyle="--")
plt.title("Force")
plt.grid()
plt.legend()
plt.show()

#%%