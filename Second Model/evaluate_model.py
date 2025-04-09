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


res1 = _model_springs.solve_springs4(pile, soil2, P, 3, N, nondim=True)

zeros = {
    "nondim solver, nondim zeros" : np.abs(res1[4]).sum(),
    "nondim solver, dim zeros" : np.abs(res1[8]).sum(),
    "dim solver, dim zeros" : np.abs(res2[4]).sum(),
    "dim solver, nondim zeros" : np.abs(res2[8]).sum(),
}
#%%

# construct values needed for jax version
jax_pile_D = jnp.full_like(z, pile.D)
jax_pile_L = pile.L
jax_pile_E = pile.E
jax_layer_type = jnp.array([0 if layer.layer_type=="clay" else 1 for layer in soil2.layers])
jax_gamma_d = jnp.array([layer.gamma_d for layer in soil2.layers])
jax_e = jnp.array([layer.e for layer in soil2.layers])
jax_c1 = jnp.array([layer.N_c if layer.layer_type=="clay" else layer.N_q for layer in soil2.layers])
jax_c2 = jnp.array([layer.psi if layer.layer_type=="clay" else layer.beta for layer in soil2.layers])
jax_shaft_friction_limit = jnp.array([layer.shaft_friction_limit for layer in soil2.layers])
jax_end_bearing_limit = jnp.array([layer.end_bearing_limit for layer in soil2.layers])
jax_base_depth = jnp.array([layer.base_depth for layer in soil2.layers])

# jaxopt.ScipyRootFinding does NOT support jit compilation! I'd need to use Bryoden or smoothened-LBFGS instead but they have stability issues.
# jit_fn = jax.jit(_model_springs_jax.solve_springs_api_jax, static_argnames=("P", "z_w", "N"))
# res2_jax = jit_fn(jax_pile_D, jax_pile_L, jax_pile_E, jax_layer_type, jax_gamma_d, jax_e, jax_c1, jax_c2, jax_shaft_friction_limit, jax_end_bearing_limit, jax_base_depth, P, 3, N)

input_params = (jax_pile_D, jax_pile_L, jax_pile_E, jax_layer_type, jax_gamma_d, jax_e, jax_c1, jax_c2, jax_shaft_friction_limit, jax_end_bearing_limit, jax_base_depth, P, 3, N)
input_params2 = copy.deepcopy(input_params)
res3_jax = _model_springs_jax.solve_springs_api_jax(*input_params, nondim=True, data=np.concatenate((res1[2], res1[3])))
res2_jax = _model_springs_jax.solve_springs_api_jax(*input_params, nondim=False)

zeros_jax = {
    "nondim solver, nondim zeros" : np.abs(res3_jax[4]).sum(),
    "nondim solver, dim zeros" : np.abs(res3_jax[8]).sum(),
    "dim solver, dim zeros" : np.abs(res2_jax[4]).sum(),
    "dim solver, nondim zeros" : np.abs(res2_jax[8]).sum(),
    "numpy nondim solver, nondim zeros" : np.abs(res3_jax[9]).sum(),
}

# with open('disp_nondim.json', 'w') as fp:
#     json.dump(np.concatenate((res1[2], res1[3])).tolist(), fp)
# with open('disp_nondim_jax.json', 'w') as fp:
#     json.dump(np.concatenate((res3_jax[2], res3_jax[3])).tolist(), fp)

#%%
# def wrapper_f(*params):
#     return _model_springs_jax.solve_springs_api_jax(*params)[0].sum()
# vg = jax.value_and_grad(wrapper_f, argnums=7)
# g = jax.grad(wrapper_f, argnums=7)
# jax.make_jaxpr(g, static_argnums=[11,12,13])(*input_params)
# print(vg(*input_params))

#%%

plt.plot(res1[0], z, label="1")
plt.plot(res2[0], z, label="2")
plt.plot(res2_jax[0], z, label="jax", linestyle="--")
plt.plot(res3_jax[0], z, label="jax nondim")
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