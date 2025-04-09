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
import sys

sys.path.insert(1, r"C:\Users\gregk\Documents\MyDocuments\IIB\Project\Code\Second Model\packaged")

import _pile_and_soil as _pile_and_soil
import _model_springs as _model_springs
import _model_springs_jax as _model_springs_jax

jax.config.update('jax_enable_x64', True) #ESSENTIAL

# Pile
pile = _pile_and_soil.Pile(R=0.3, L=30, E=35e9)

# Define the other variables (loading, number of elements, model noise variance)
P = 1.8e6 # top axial load
N = 100 # number of nodes along pile
z = np.linspace(0, pile.L, N)
z_midpoints = 0.5 * (z[:-1] + z[1:])

# according to API, beta for soils should be scaled by 1.25 for plugged piles
plug_factor=1.25

# The mu, sigma parameters of a lognormal distribution are not the true mean and standard deviation of the distribution.
# This function takes the true mean and standard deviation and outputs the mu, sigma parameters of the lognormal distribution.
def get_lognormal_params(mean, stdev):
    mu_lognormal = np.log(mean**2 / np.sqrt(stdev**2 + mean**2))
    sigma_lognormal = np.sqrt(np.log(1 + (stdev**2 / mean**2)))
    return {"mean" : mu_lognormal, "sigma" : sigma_lognormal}

# gamma_d_means = np.array([8e3, 8e3])
# gamma_d_stdevs = np.array([5e3, 5e3])
gamma_d_means = np.array([15e3, 17e3])
gamma_d_stdevs = np.array([3e3, 3e3])

#%%

sample_n = 100
zeros_list = []
zeros_jax_list = []
res_list = []
res_jax_list = []
gamma_d_list = []
for i in range(sample_n):
    if i==0: gamma_d = np.array([6916.90781741, 7946.94530825])
    else: gamma_d = np.random.lognormal(**get_lognormal_params(gamma_d_means, gamma_d_stdevs), size=2)
    gamma_d_list.append(gamma_d.tolist())

    layer1 = _pile_and_soil.SandLayer(gamma_d=gamma_d[0], e=0.689, N_q=8, beta=plug_factor*0.214, shaft_friction_limit=47.8e3, end_bearing_limit=1.9e6, base_depth=12.5)
    layer2 = _pile_and_soil.SandLayer(gamma_d=gamma_d[1], e=0.441, N_q=40, beta=plug_factor*0.46, shaft_friction_limit=96e3, end_bearing_limit=10e6, base_depth=pile.L)
    soil = _pile_and_soil.Soil([layer1, layer2])

    # construct values needed for jax version
    jax_pile_D = jnp.full_like(z, pile.D)
    jax_pile_L = pile.L
    jax_pile_E = pile.E
    jax_layer_type = jnp.array([0 if layer.layer_type=="clay" else 1 for layer in soil.layers])
    jax_gamma_d = jnp.array([layer.gamma_d for layer in soil.layers])
    jax_e = jnp.array([layer.e for layer in soil.layers])
    jax_c1 = jnp.array([layer.N_c if layer.layer_type=="clay" else layer.N_q for layer in soil.layers])
    jax_c2 = jnp.array([layer.psi if layer.layer_type=="clay" else layer.beta for layer in soil.layers])
    jax_shaft_friction_limit = jnp.array([layer.shaft_friction_limit for layer in soil.layers])
    jax_end_bearing_limit = jnp.array([layer.end_bearing_limit for layer in soil.layers])
    jax_base_depth = jnp.array([layer.base_depth for layer in soil.layers])

    input_params = {"pile_D": jax_pile_D, "pile_L": jax_pile_L, "pile_E": jax_pile_E, "l_layer_type": jax_layer_type,
                    "l_gamma_d": jax_gamma_d, "l_e": jax_e, "l_c1": jax_c1, "l_c2": jax_c2,
                    "l_shaft_friction_limit": jax_shaft_friction_limit, "l_end_bearing_limit": jax_end_bearing_limit,
                    "l_base_depth": jax_base_depth, "P": P, "z_w": 3, "N": N}


    res_nondim = _model_springs.solve_springs4(pile, soil, P, 3, N, nondim=True)

    # jitted_solve_springs = jax.jit(_model_springs_jax.solve_springs_api_jax, static_argnames=("pile_L", "pile_E", "P", "z_w", "N", "t_res_clay"))
    jitted_solve_springs = _model_springs_jax.solve_springs_api_jax
    # res_jax_nondim = jitted_solve_springs(**input_params, throw=False, sol_verbose=frozenset({"step", "accepted", "loss", "step_size"}))
    res_jax_nondim = jitted_solve_springs(**input_params, throw=False)

    zeros = {
        "nondim jax" : np.abs(res_jax_nondim[4]).sum(),
        "nondim" : np.abs(res_nondim[4]).sum(),
    }
    zeros_list.append(zeros["nondim"])
    zeros_jax_list.append(zeros["nondim jax"])
    res_list.append(res_nondim)
    res_jax_list.append(res_jax_nondim)

    print(f"Sample number {i}, gamma_d: {gamma_d[0]:.4f}, {gamma_d[1]:.4f}, numpy zeros: {zeros['nondim jax']:.4e}, jax zeros: {zeros['nondim']:.4e}")
# %%

fig, ax = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
plt.tight_layout()
ax[0].scatter(*zip(gamma_d_list), c=zeros_list, label="numpy", cmap="viridis")
ax[1].scatter(*zip(gamma_d_list), c=zeros_jax_list, label="jax", cmap="viridis")
ax[0].colorbar()
ax[1].colorbar()
ax[0].xlabel("gamma_d 1")
ax[0].ylabel("gamma_d 2")
ax[0].legend()
ax[1].legend()
plt.title("Residuals of JAX and Numpy models")
plt.show()

# %%