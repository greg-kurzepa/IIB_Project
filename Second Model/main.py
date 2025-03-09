import numpy as np
import pymc as pm
import pandas as pd
import copy
import matplotlib.pyplot as plt

import packaged._pile_and_soil as _pile_and_soil
import packaged._model_springs as _model_springs
import packaged._unit_tests as _unit_tests

# _unit_tests.run_tests()

#%% --------------------------------------------------------------------------------------------
# Step 1: Specify priors the model sees. Define the system and ground truth parameters.

# Priors
# all soil layers will have the same priors
priors = {
    "alpha" : {"dist" : pm.LogNormal, "params" : {"mu" : 0.55, "sigma" : 0.25}}, # typically between 0.3-0.8
    "s_u0" : {"dist" : pm.LogNormal, "params" : {"mu" : 45e3, "sigma" : 10e3}},
}

# Pile
pile = _pile_and_soil.Pile(R=0.15, L=15, W=2.275e3, E=20e9)
# p = _pile_and_soil.Pile(0.15, 10, 2275, E=20e9) # https://ukrstarline.ua/en/reinforced-concrete-products/reinforced-concrete-pile-driven/reinforced-concrete-pile-driven-c-10030-10

# Soil
# in this test, there are two soil layers. each has its own uniform set of ground truth parameters
layer1 = _pile_and_soil.SoilLayer(alpha=0.4, gamma=20e3, N_c=9, s_u0=40e3, rho=3.9e3, base_depth=pile.L)
# layer2 = _pile_and_soil.SoilLayer(alpha=0.7, gamma=20e3, N_c=9, s_u0=58e3, rho=5.5e3, base_depth=pile.L)
# soil = _pile_and_soil.Soil([layer1, layer2])
soil = _pile_and_soil.Soil([layer1])
N_L = len(soil.layers) # number of layers

# Define the other variables (loading, number of elements, model noise variance)
P = 200e3 # top axial load
N = 100 # number of nodes along pile
sigma_n = 10e3 # currently the noise variance is additive, i.e. the +- (a set value) instead of a precentage of the measurement. should change this when I know better
z = np.linspace(0, pile.L, N)

# Some plots to visualise

# # plot the ground truth profile of s_u vs depth
# z = np.linspace(0, pile.L, N)
# s_u = _model_springs.vectorised_get_undrained_strength(soil, z)
# plt.plot(z, s_u)
# plt.xlabel("z")
# plt.ylabel("$s_u$")
# plt.grid()
# plt.show()

# # plot the ground truth profile of force and displacement vs depth
F, strain, u = _model_springs.solve_springs(pile, soil, z, P, N)
# # F2, strain2, u2 = _model_springs.solve_springs(pile, soil2, z, P, N)
# plt.plot(F, z, label="1")
# # plt.plot(F2, z, label="2")
# plt.gca().invert_yaxis()
# plt.xlabel("F")
# plt.ylabel("z")
# plt.legend()
# plt.grid()
# plt.show()

# CHECK AGAINST RSPILE TUTORIAL: https://www.youtube.com/watch?v=iOsDRqHUbA8 or try to find another one where we can see the values!

#%% --------------------------------------------------------------------------------------------
# Step 2: Generate synthetic data

RANDOM_SEED = 8928
np.random.seed(RANDOM_SEED)

synthetic_F = F + np.random.standard_normal(N) * sigma_n

#%% --------------------------------------------------------------------------------------------
# Step 3: Define the inference model

def inf_solve(inf_alpha, inf_s_u0): # this line has the inference parameters that must always be passed.
    for layer in range(N_L):
        inf_soil.layers[layer].alpha = inf_alpha
        inf_soil.layers[layer].s_u0 = inf_s_u0

    return _model_springs.solve_springs(inf_pile, inf_soil, P, N)[1] # 1th index is the force.

# coords = {"layers" : range(N_L)}
model = pm.Model()

with model:
    # Observed predictors and outcome
    depth = pm.Data("depth", z) # depth of each node
    measured_force = pm.Data("measured_force", synthetic_F) # measured noisy force

    # Priors, one for each layer
    # inf_alpha = priors["alpha"]["dist"]("inf_alpha", **priors["alpha"]["params"])#, dims="layers")
    # inf_s_u0 = priors["s_u0"]["dist"]("inf_s_u0", **priors["s_u0"]["params"])#, dims="layers")

    inf_alpha = pm.LogNormal("inf_alpha", mu=0.55, sigma=0.25), # typically between 0.3-0.8
    inf_s_u0 = pm.LogNormal("inf_s_u0", mu=45e3, sigma=10e3),

    # note, deterministic is not strictly necessary and can slow down sampling a lot.
    # if the sampling gets slow, try replacing this.
    force = pm.Deterministic(
        "force", inf_solve(inf_alpha, inf_s_u0)
    )

    obs = pm.Normal("obs", mu=force, sigma=sigma_n, observed=measured_force)

# pm.model_to_graphviz(model).render("model") # if not in jupyter notebook, add .render("model")

#%% --------------------------------------------------------------------------------------------
# Step 4: Prior predictive check

with model:
    idata = pm.sample_prior_predictive(draws=1000, random_seed=RANDOM_SEED)

# plot the force profile using the priors, as well as the prior predictive distribution which should be = that + the zero-mean noise.

#%% --------------------------------------------------------------------------------------------
# Step 5: Inference and MCMC diagnostics

#%% --------------------------------------------------------------------------------------------
# Step 6: Posterior predictive check
# %%
