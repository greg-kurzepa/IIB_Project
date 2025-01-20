#%%

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm

print(f"Running on PyMC v{pm.__version__}")

# %config InlineBackend.figure_format = 'retina'
az.style.use("arviz-darkgrid")

def plot_xY(x, Y, ax):
    quantiles = Y.quantile((0.025, 0.25, 0.5, 0.75, 0.975), dim=("chain", "draw")).transpose()

    az.plot_hdi(
        x,
        hdi_data=quantiles.sel(quantile=[0.025, 0.975]),
        fill_kwargs={"alpha": 0.25},
        smooth=False,
        ax=ax,
    )
    az.plot_hdi(
        x,
        hdi_data=quantiles.sel(quantile=[0.25, 0.75]),
        fill_kwargs={"alpha": 0.5},
        smooth=False,
        ax=ax,
    )
    ax.plot(x, quantiles.sel(quantile=0.5), color="C1", lw=3)

class Pile:
    def __init__(self, R, L, W):
        self.R = R # the outer radius of the pile
        self.L = L # the length of the pile
        self.W = W # the total weight of the pile

def g(pile, z, alpha, gamma, N_c, s_u0, rho):
    # base bearing load
    BEARING = np.pi * pile.R**2 * ( gamma*pile.L + N_c*( s_u0 + rho*pile.L ) )
    print(BEARING)

    # total vertical shear force below z
    SHEAR = 2*np.pi*pile.R*alpha * ( 0.5*rho*( pile.L**2 - z**2 ) + s_u0*( pile.L - z ) )

    # total pile weight below z
    W_FRAC = pile.W * (1 - z/pile.L)

    # vertical force in steel at that point
    F = BEARING + SHEAR - W_FRAC
    return F

# True parameter values (used to test the MCMC fit)
pile = Pile(0.15, 10, 2275) # https://ukrstarline.ua/en/reinforced-concrete-products/reinforced-concrete-pile-driven/reinforced-concrete-pile-driven-c-10030-10
alpha = 0.4 # shear utilisation, typically 0.3-0.8
gamma = 20e3 # unit weight of soil, typically 20-22 kn/m3 for London Clay
N_c = 9 # bearing utlisation, typically 9
s_u0 = 30e3 # surface undrained shear strength, representative of London Clay
rho = 4.8e3 # rate of increase of s_u with depth, representative of London Clay
sigma_n = 5e3 # model noise variance

size = 100 # size of validation dataset
z = np.linspace(0, pile.L, size)
F = g(pile, z, alpha, gamma, N_c, s_u0, rho)

# plot the theoretical load profile
# plt.plot(F/1000, z)
# plt.gca().invert_yaxis()
# plt.xlabel("F, kN")
# plt.ylabel("z")
# plt.title("Theoretical Vertical Load Profile in Pile at Soil Failure")
# plt.show()

#%%

# Initialize random number generator
RANDOM_SEED = 8927
rng = np.random.default_rng(RANDOM_SEED)

# Predictor variable
# F_noisy = F + np.random.randn(size) * sigma_n
F_noisy = F + rng.standard_normal(size) * sigma_n
# plt.hist(F_noisy, bins=20)
# plt.show()

# Display generated data
# plt.scatter(F_noisy/1000, z, alpha=0.6)
# plt.gca().invert_yaxis()
# plt.xlabel("F, kN")
# plt.ylabel("z")
# plt.title("Noisy (\"Measured\") Vertical Load Profile in Pile at Soil Failure")
# plt.show()

#%%

model = pm.Model()

with model:
    # Observed predictors and outcome
    depth = pm.MutableData("depth", z)
    measured_force = pm.MutableData("measured_force", F_noisy) # measured noisy force

    # Priors
    shear_utilisation = pm.LogNormal("shear_utilisation", mu=-0.6, sigma=0.3)
    undrained_strength = pm.LogNormal("undrained_strength", mu=10.4, sigma=0.11)
    noise_sigma = pm.Normal("noise_sigma", mu=4e3, sigma=0.5e3)

    force = pm.Deterministic(
        "force", g(pile, depth, shear_utilisation, gamma, N_c, undrained_strength, rho)
    )

    obs = pm.Normal("obs", mu=force, sigma=noise_sigma, observed=measured_force)

pm.model_to_graphviz(model)

#%%

with model:
    idata = pm.sample_prior_predictive(random_seed=RANDOM_SEED)

# plot prior on parameters
# az.plot_trace(idata, var_names=["undrained_strength"])

# plot prior predictive distribution
figsize = (10,5)
fig, ax = plt.subplots(figsize=figsize)

plot_xY(z, idata.prior_predictive["obs"], ax)
ax.scatter(z, F_noisy, label="observed", alpha=0.6, zorder=100)
ax.set(title="Prior predictive distribution")
# plt.gca().invert_yaxis()
plt.legend()
plt.show()

az.plot_ppc(idata, group="prior")
plt.show()

#%%

# draw from posterior
with model:
    idata.extend(pm.sample(random_seed=RANDOM_SEED, cores=1))

az.plot_trace(idata, var_names=["~force"])
plt.show()

#%%

az.plot_dist_comparison(idata, var_names=["undrained_strength"])
az.plot_dist_comparison(idata, var_names=["shear_utilisation"])
az.plot_dist_comparison(idata, var_names=["noise_sigma"])

#%%

with model:
    idata.extend(pm.sample_posterior_predictive(idata, random_seed=RANDOM_SEED))

fig, ax = plt.subplots(figsize=figsize)

az.plot_hdi(z, idata.posterior_predictive["obs"], hdi_prob=0.5, smooth=False)
az.plot_hdi(z, idata.posterior_predictive["obs"], hdi_prob=0.95, smooth=False)
ax.scatter(z, F_noisy, label="observed", alpha=0.6)
ax.set(title="Posterior predictive distribution")
ax.set_xlabel("depth z")
ax.set_ylabel("Force (N)")
plt.legend()
plt.show()
# %%
