import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm

print(f"Running on PyMC v{pm.__version__}")

# %config InlineBackend.figure_format = 'retina'
# Initialize random number generator
RANDOM_SEED = 8927
rng = np.random.default_rng(RANDOM_SEED)
az.style.use("arviz-darkgrid")

# True parameter values
alpha, sigma = 1, 1
beta = [1, 2.5]

# Size of dataset
size = 100

# Predictor variable
X1 = np.random.randn(size)
X2 = np.random.randn(size) * 0.2

# Simulate outcome variable
Y = alpha + beta[0] * X1 + beta[1] * X2 + rng.normal(size=size) * sigma

# Display generated data
# fig, axes = plt.subplots(1, 2, sharex=True, figsize=(10, 4))
# axes[0].scatter(X1, Y, alpha=0.6)
# axes[1].scatter(X2, Y, alpha=0.6)
# axes[0].set_ylabel("Y")
# axes[0].set_xlabel("X1")
# axes[1].set_xlabel("X2")

basic_model = pm.Model()

with basic_model:
    # Priors for unknown model parameters
    alpha = pm.Normal("alpha", mu=0, sigma=10)
    beta = pm.Normal("beta", mu=0, sigma=10, shape=2)
    sigma = pm.HalfNormal("sigma", sigma=1)

    # Expected value of outcome
    mu = alpha + beta[0] * X1 + beta[1] * X2

    # Likelihood (sampling distribution) of observations
    Y_obs = pm.Normal("Y_obs", mu=mu, sigma=sigma, observed=Y)

with basic_model:
    # draw 1000 posterior samples
    idata = pm.sample(cores=1)

print(idata.posterior["alpha"].sel(draw=slice(0, 4)))

# with basic_model:
#     # instantiate sampler
#     step = pm.Slice()

#     # draw 5000 posterior samples
#     slice_idata = pm.sample(5000, step=step, cores=1)

az.plot_trace(idata, combined=True)
plt.show()

print(az.summary(idata, round_to=2))