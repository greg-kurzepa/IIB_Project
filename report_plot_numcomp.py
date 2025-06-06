#%%
import matplotlib.pyplot as plt
import arviz as az
import cloudpickle
import numpy as np

#%%

api_idata_1 = cloudpickle.load(open("results_new\\api_idata_single_3,6.pkl", "rb"))
api_idata_2 = cloudpickle.load(open("results_new\\api_idata_many.pkl", "rb"))
sf = 1000

# %%

fig, ax = plt.subplots(1, 2, figsize=(6, 3))

for i in range(2):
    posterior_single = np.array(api_idata_1.posterior["l_gamma_d"])[:,:,i].flatten()/sf
    posterior_many = np.array(api_idata_2.posterior["l_gamma_d"])[:,:,i].flatten()/sf

    ax[i].hist(posterior_single, density=True, bins=50, alpha=0.5, hatch="/", label="posterior samples\nsingle variable inference", color="C2")
    ax[i].hist(posterior_many, density=True, bins=50, alpha=0.5, label="posterior samples\nmulti variable inference", color="C4")
    ax[i].get_yaxis().set_visible(False)
    ax[i].set_xlabel(f"$\\gamma_d$, Layer {i+1}, kN/m")
    ax[i].axvline(15 if i == 0 else 17, color='k', linestyle='--', label="ground truth")
    if i==0: ax[i].legend()

plt.tight_layout()
plt.show()

#%%