#%%
import matplotlib.pyplot as plt
import arviz as az
import cloudpickle
import numpy as np

#%%

api_idata_1 = cloudpickle.load(open("results_new\\api_idata_single_3,6.pkl", "rb"))
api_idata_2 = cloudpickle.load(open("results_new\\api_idata_single_4,6.pkl", "rb"))
sf = 1000

prior_gamma_1 = np.array(api_idata_1.prior["l_gamma_d"])[:,:,0].flatten()/sf
prior_gamma_2 = np.array(api_idata_1.prior["l_gamma_d"])[:,:,1].flatten()/sf

posterior_gamma_1 = np.array(api_idata_1.posterior["l_gamma_d"])[:,:,0].flatten()/sf
posterior_gamma_2 = np.array(api_idata_1.posterior["l_gamma_d"])[:,:,1].flatten()/sf

prior_gamma_1_idat2 = np.array(api_idata_2.prior["l_gamma_d"])[:,:,0].flatten()/sf
prior_gamma_2_idat2 = np.array(api_idata_2.prior["l_gamma_d"])[:,:,1].flatten()/sf

posterior_gamma_1_idat2 = np.array(api_idata_2.posterior["l_gamma_d"])[:,:,0].flatten()/sf
posterior_gamma_2_idat2 = np.array(api_idata_2.posterior["l_gamma_d"])[:,:,1].flatten()/sf

#%%

fig,ax = plt.subplots(1, 2, figsize=(12, 6))

ax[0].hist(prior_gamma_1, density=True, bins=50, alpha=0.5, hatch="/", label="prior")
ax[0].hist(posterior_gamma_1, density=True, bins=50, alpha=0.5, label="posterior")
ax[0].get_yaxis().set_visible(False)
ax[0].set_xlabel("$\\gamma_d$, Top Layer, kN/m")
ax[0].legend()

ax[1].hist(prior_gamma_2, density=True, bins=50, alpha=0.5, hatch="/", label="prior")
ax[1].hist(posterior_gamma_2, density=True, bins=50, alpha=0.5, label="posterior")
ax[1].get_yaxis().set_visible(False)
ax[1].set_xlabel("$\\gamma_d$, Bottom Layer, kN/m")
# ax[1].legend()

plt.tight_layout()
plt.show()
# %%

fig = plt.figure(constrained_layout=True, figsize=(6,6))

subfigs = fig.subfigures(nrows=2, ncols=1)
for row, subfig in enumerate(subfigs):
    subfig.suptitle('$P=3.6MN$' if row==0 else '$P=4.6MN$', fontsize=16)

    # create 1x3 subplots per subfig
    ax = subfig.subplots(nrows=1, ncols=2)

    ax[0].hist(prior_gamma_1 if row==0 else prior_gamma_1_idat2, density=True, hatch="/", bins=50, alpha=0.5, label="prior samples")
    ax[0].hist(posterior_gamma_1 if row==0 else posterior_gamma_1_idat2, density=True, bins=50, alpha=0.5, label="posterior samples")
    ax[0].get_yaxis().set_visible(False)
    ax[0].set_xlabel("$\\gamma_d$, Top Layer, kN/m")
    ax[0].axvline(15, color='k', linestyle='--', label="ground truth")
    if row==0: ax[0].legend()

    ax[1].hist(prior_gamma_2 if row==0 else prior_gamma_2_idat2, density=True, hatch="/", bins=50, alpha=0.5, label="prior samples")
    ax[1].hist(posterior_gamma_2 if row==0 else posterior_gamma_2_idat2, density=True, bins=50, alpha=0.5, label="posterior samples")
    ax[1].get_yaxis().set_visible(False)
    ax[1].set_xlim(10, 30)
    ax[1].set_xlabel("$\\gamma_d$, Bottom Layer, kN/m")
    ax[1].axvline(17, color='k', linestyle='--', label="ground truth")

plt.show()

#%%