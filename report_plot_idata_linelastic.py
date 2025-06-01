#%%
import matplotlib.pyplot as plt
import arviz as az
import cloudpickle
import numpy as np

#%%

linel_idata_1 = cloudpickle.load(open("results\\linearelastic_idata_1.pkl", "rb"))
linel_idata_2 = cloudpickle.load(open("results\\linearelastic_idata_2.pkl", "rb"))

#%%
sf = 1e6
sf2 = 1e6

prior_Es_1 = np.array(linel_idata_1.prior["_E_soil"])[:,:,0].flatten()/sf
prior_Es_2 = np.array(linel_idata_1.prior["_E_soil"])[:,:,1].flatten()/sf
prior_Kb = np.array(linel_idata_1.prior["_Kb"]).flatten()/sf2

posterior_Es_1 = np.array(linel_idata_1.posterior["_E_soil"])[:,:,0].flatten()/sf
posterior_Es_2 = np.array(linel_idata_1.posterior["_E_soil"])[:,:,1].flatten()/sf
posterior_Kb = np.array(linel_idata_1.posterior["_Kb"]).flatten()/sf2

prior_Es_1_idat2 = np.array(linel_idata_2.prior["_E_soil"])[:,:,0].flatten()/sf
prior_Es_2_idat2 = np.array(linel_idata_2.prior["_E_soil"])[:,:,1].flatten()/sf
prior_Kb_idat2 = np.array(linel_idata_2.prior["_Kb"]).flatten()/sf2

posterior_Es_1_idat2 = np.array(linel_idata_2.posterior["_E_soil"])[:,:,0].flatten()/sf
posterior_Es_2_idat2 = np.array(linel_idata_2.posterior["_E_soil"])[:,:,1].flatten()/sf
posterior_Kb_idat2 = np.array(linel_idata_2.posterior["_Kb"]).flatten()/sf2

# %%

fig = plt.figure(constrained_layout=True, figsize=(10,6))

subfigs = fig.subfigures(nrows=2, ncols=1)
for row, subfig in enumerate(subfigs):
    subfig.suptitle('$P=3.6MN$' if row==0 else '$P=4.6MN$', fontsize=16)

    # create 1x3 subplots per subfig
    ax = subfig.subplots(nrows=1, ncols=3)

    ax[0].hist(posterior_Es_1 if row==0 else posterior_Es_1_idat2, density=True, bins=50, alpha=0.5, label="posterior samples")
    ax[0].get_yaxis().set_visible(False)
    ax[0].set_xlabel("$E_s$, Top Layer, MPa")
    ax[0].set_xlim(0, 2)
    if row==0: ax[0].legend()

    ax[1].hist(posterior_Es_2 if row==0 else posterior_Es_2_idat2, density=True, bins=50, alpha=0.5, label="posterior samples")
    ax[1].get_yaxis().set_visible(False)
    ax[1].set_xlabel("$E_s$, Bottom Layer, MPa")
    ax[1].set_xlim(0, 30)

    ax[2].hist(posterior_Kb if row==0 else posterior_Kb_idat2, density=True, bins=50, alpha=0.5, label="posterior samples")
    ax[2].get_yaxis().set_visible(False)
    ax[2].set_xlabel("$K_b$, MN/m")
    ax[2].set_xlim(0, 200)

plt.show()
# %%
