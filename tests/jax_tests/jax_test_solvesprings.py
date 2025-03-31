#%%
import jax
import jax.numpy as jnp

import sys
import matplotlib.pyplot as plt

sys.path.append("C:\\Users\\gregk\\Documents\\MyDocuments\\IIB\\Project\\Code\\Second Model\\packaged")
import _model_springs_jax

L=15

# Pile
pile = jnp.array([0.15, L, 2.275e3, 20e9]) # (R, L, W, E)

# Soil
# in this test, there are two soil layers. each has its own uniform set of ground truth parameters
layer1 = jnp.array([0.4, 20e3, 9, 40e3, 3.9e3, 15]) # (alpha, gamma, N_c, s_u0, rho, base_depth)
# layer2 = _pile_and_soil.SoilLayer(alpha=0.7, gamma=20e3, N_c=9, s_u0=58e3, rho=5.5e3, base_depth=pile.L)
soil = jnp.vstack([layer1])
N_L = 1 # number of layers

# Define the other variables (loading, number of elements, model noise variance)
P = 200.0e3 # top axial load
N = 100 # number of nodes along pile
sigma_n = 10e3 # currently the noise variance is additive, i.e. the +- (a set value) instead of a precentage of the measurement. should change this when I know better
z = jnp.linspace(0, L, N)

#%%
# Some plots to visualise

# # plot the ground truth profile of force and displacement vs depth

F, strain, u = _model_springs_jax.solve_springs_jax(pile, soil, P, N)

#%%

# F2, strain2, u2 = _model_springs.solve_springs(pile, soil2, z, P, N)
plt.plot(F, z, label="1")
# plt.plot(F2, z, label="2")
plt.gca().invert_yaxis()
plt.xlabel("F")
plt.ylabel("z")
plt.legend()
plt.grid()
plt.show()
# %%
