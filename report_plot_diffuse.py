#%%
import matplotlib.pyplot as plt
import arviz as az
import cloudpickle
import numpy as np

#%%

api_idata_1 = cloudpickle.load(open("results_new\\api_idata_single_3,6.pkl", "rb"))
api_idata_2 = cloudpickle.load(open("results_new\\api_idata_single_3,6_uniform.pkl", "rb"))
sf = 1000

prior_gamma_1 = np.array(api_idata_1.prior["l_gamma_d"])[:,:,0].flatten()/sf
prior_gamma_2 = np.array(api_idata_1.prior["l_gamma_d"])[:,:,1].flatten()/sf

posterior_gamma_1 = np.array(api_idata_1.posterior["l_gamma_d"])[:,:,0].flatten()/sf
posterior_gamma_2 = np.array(api_idata_1.posterior["l_gamma_d"])[:,:,1].flatten()/sf

prior_gamma_1_diffuse = np.array(api_idata_2.prior["l_gamma_d"])[:,:,0].flatten()/sf
prior_gamma_2_diffuse = np.array(api_idata_2.prior["l_gamma_d"])[:,:,1].flatten()/sf

posterior_gamma_1_diffuse = np.array(api_idata_2.posterior["l_gamma_d"])[:,:,0].flatten()/sf
posterior_gamma_2_diffuse = np.array(api_idata_2.posterior["l_gamma_d"])[:,:,1].flatten()/sf

#%%

fig, ax = plt.subplots(1, 2, figsize=(8, 4))

ax[0].hist(posterior_gamma_1, density=True, bins=50, alpha=0.5, label="posterior samples\ninformative prior", hatch="/", color="C2")
ax[0].hist(posterior_gamma_1_diffuse, density=True, bins=50, alpha=0.5, label="posterior samples\nuninformative prior", color="C4")
ax[0].get_yaxis().set_visible(False)
ax[0].set_xlabel("$\\gamma_d$, Top Layer, kN/m")
ax[0].axvline(15, color='k', linestyle='--', label="ground truth")
ax[0].legend()

ax[1].hist(prior_gamma_2, density=True, bins=50, alpha=0.5, label="posterior samples\ninformative prior", hatch="/", color="C2")
ax[1].hist(posterior_gamma_2_diffuse, density=True, bins=50, alpha=0.5, label="posterior samples\nuninformative prior", color="C4")
ax[1].get_yaxis().set_visible(False)
ax[1].set_xlabel("$\\gamma_d$, Bottom Layer, kN/m")
ax[1].axvline(17, color='k', linestyle='--', label="ground truth")
# ax[1].legend()

plt.tight_layout()
plt.show()

#%%