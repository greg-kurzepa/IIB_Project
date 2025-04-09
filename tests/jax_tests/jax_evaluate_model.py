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
P = 4.6e6 # top axial load
N = 100 # number of nodes along pile
z = np.linspace(0, pile.L, N)
z_midpoints = 0.5 * (z[:-1] + z[1:])

# according to API, beta for soils should be scaled by 1.25 for plugged piles
plug_factor=1.25

# in this test, there are two soil layers. each has its own uniform set of ground truth parameters
# Soil 1
gamma_d = [15e3, 17e3]
# gamma_d = np.array([38863.71553916,  7728.92628868])
# gamma_d = np.array([6916.90781741, 7946.94530825])
layer1 = _pile_and_soil.SandLayer(gamma_d=gamma_d[0], e=0.689, N_q=8, beta=plug_factor*0.214, shaft_friction_limit=47.8e3, end_bearing_limit=1.9e6, base_depth=12.5)
layer2 = _pile_and_soil.SandLayer(gamma_d=gamma_d[1], e=0.441, N_q=40, beta=plug_factor*0.46, shaft_friction_limit=96e3, end_bearing_limit=10e6, base_depth=pile.L)
soil = _pile_and_soil.Soil([layer1, layer2])

res_nondim = _model_springs.solve_springs4(pile, soil, P, 3, N, nondim=True)

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
jax_shaft_friction_limit = jnp.array([layer.shaft_friction_limit for layer in soil.layers])
jax_end_bearing_limit = jnp.array([layer.end_bearing_limit for layer in soil.layers])
jax_base_depth = jnp.array([layer.base_depth for layer in soil.layers])

input_params = {"pile_D": jax_pile_D, "pile_L": jax_pile_L, "pile_E": jax_pile_E, "l_layer_type": jax_layer_type,
                "l_gamma_d": jax_gamma_d, "l_e": jax_e, "l_c1": jax_c1, "l_c2": jax_c2,
                "l_shaft_pressure_limit": jax_shaft_friction_limit, "l_end_pressure_limit": jax_end_bearing_limit,
                "l_base_depth": jax_base_depth, "P": P, "z_w": 3, "N": N}

# jitted_solve_springs = jax.jit(_model_springs_jax.solve_springs_api_jax, static_argnames=("pile_L", "pile_E", "P", "z_w", "N", "t_res_clay"))
jitted_solve_springs = _model_springs_jax.solve_springs_api_jax
res_jax_nondim = jitted_solve_springs(**input_params, throw=False, sol_verbose=frozenset({"step", "accepted", "loss", "step_size"}))

zeros = {
    "nondim jax" : np.abs(res_jax_nondim[4]).sum(),
    "nondim" : np.abs(res_nondim[4]).sum(),
}

print(zeros)
print(res_jax_nondim[8], res_jax_nondim[9])

#%%

def log_likelihood(data, sigma, forces):
    # Assuming additive gaussian white noise
    # If forces are all jnp.nan, set probability to zero
    logp = jax.lax.cond(
        jnp.all(jnp.isnan(forces)),
        lambda: -jnp.inf,
        lambda: (- jnp.log(sigma) - 0.5 * jnp.log(2 * jnp.pi) - 0.5 * ((data - forces) / sigma) ** 2).sum()
    )
    return logp

print(log_likelihood(res_jax_nondim[0], 0.05e6, res_jax_nondim[0]))

#%%

plt.plot(res_nondim[0], z, label="np nondim")
plt.plot(res_jax_nondim[0], z, label="jax nondim")
plt.xlim(left=0)
plt.gca().invert_yaxis()
plt.xlabel("F")
plt.ylabel("z")
# plt.xlim(right=2e6)
plt.legend()
plt.grid()
plt.show()

#%%

def wrapper_f(pile_D, pile_L, pile_E, l_layer_type, l_gamma_d, l_e, l_c1, l_c2,
              l_shaft_pressure_limit, l_end_pressure_limit, l_base_depth,
              P, z_w, N=100, t_res_clay=0.9, data=None):
    
    return _model_springs_jax.solve_springs_api_jax(pile_D, pile_L, pile_E, l_layer_type, l_gamma_d, l_e, l_c1, l_c2,
              l_shaft_pressure_limit, l_end_pressure_limit, l_base_depth,
              P, z_w, N, t_res_clay, data)[0].sum()

jitted_wrapper_f = jax.jit(wrapper_f, static_argnames=("pile_L", "pile_E", "P", "z_w", "N", "t_res_clay"))
vg = jax.value_and_grad(jitted_wrapper_f, argnums=7) # 7 corresponds to jax_c2
# g = jax.grad(wrapper_f, argnums=7)
# jax.make_jaxpr(g, static_argnums=[11,12,13])(*input_params)
print(vg(*input_params.values()))
# print(vg(**input_params)) # doesn't work
# %%
