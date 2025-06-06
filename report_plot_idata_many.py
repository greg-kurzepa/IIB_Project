#%%
import matplotlib.pyplot as plt
import arviz as az
import cloudpickle
import numpy as np

linel_idata = cloudpickle.load(open("results_new\\api_idata_many.pkl", "rb"))

#%%

varnames = list(linel_idata.posterior.data_vars)
varnames = ["l_gamma_d", "l_e", "l_c2", "l_shaft_pressure_limit"]
varlatex = ["$\\gamma_d$", "$e$", "$\\beta$", "$\\tau_{{cap}}$"]

ground_truths = (
    np.array([15000, 17000]),
    np.array([0.8, 0.45]),
    1.25*np.array([0.214, 0.46]),
    np.array([47.8e3, 96e3]),
)

fig, ax = plt.subplots(4, 2, figsize=(7,14))

for idx, varname in enumerate(varnames):
    sf = 1
    if idx == 0 or idx == 3: sf = 1e3

    prior = np.array(linel_idata.prior[varname])
    prior_l1 = prior[:,:,0].flatten()/sf
    prior_l2 = prior[:,:,1].flatten()/sf

    posterior = np.array(linel_idata.posterior[varname])
    posterior_l1 = posterior[:,:,0].flatten()/sf
    posterior_l2 = posterior[:,:,1].flatten()/sf

    ax[idx, 0].hist(prior_l1, density=True, bins=50, alpha=0.5, hatch="/", label="prior samples")
    ax[idx, 0].hist(posterior_l1, density=True, bins=50, alpha=0.5, label="posterior samples")
    ax[idx, 0].get_yaxis().set_visible(False)
    ax[idx, 0].set_xlabel(f"{varlatex[idx]}, Top Layer{', kN/m' if idx==0 else (', kPa' if idx==3 else '')}")
    ax[idx, 0].axvline(ground_truths[idx][0]/sf, color='k', linestyle='--', label="ground truth")
    if idx==0: ax[idx, 0].legend()

    ax[idx, 1].hist(prior_l2, density=True, bins=50, alpha=0.5, hatch="/", label="prior samples")
    ax[idx, 1].hist(posterior_l2, density=True, bins=50, alpha=0.5, label="posterior samples")
    ax[idx, 1].get_yaxis().set_visible(False)
    ax[idx, 1].set_xlabel(f"{varlatex[idx]}, Bottom Layer{', kN/m' if idx==0 else (', kPa' if idx==3 else '')}")
    ax[idx, 1].axvline(ground_truths[idx][1]/sf, color='k', linestyle='--', label="ground truth")

plt.tight_layout()
plt.show()

#%%