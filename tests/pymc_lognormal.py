#%%
import numpy as np
import pymc as pm

def get_lognormal_params(mean, stdev):
    # The mu, sigma parameters of a lognormal distribution are not the true mean and standard deviation of the distribution.
    # This function takes the true mean and standard deviation and outputs the mu, sigma parameters of the lognormal distribution.
    mu_lognormal = np.log(mean**2 / np.sqrt(stdev**2 + mean**2))
    sigma_lognormal = np.sqrt(np.log(1 + (stdev**2 / mean**2)))
    return {"mu" : mu_lognormal, "sigma" : sigma_lognormal}

with pm.Model() as no_grad_model:
    x = pm.LogNormal("gamma_d_1", **get_lognormal_params(15e3, 1.5e3))

#%%
x_draws = pm.draw(x, draws=10)
# %%