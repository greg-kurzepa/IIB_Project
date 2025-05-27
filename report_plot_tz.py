#%%
# import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import packaged._model_springs as _model_springs

# az.style.use("arviz-darkgrid")

z_over_D_tau = np.linspace(0, 0.025, 100)
ptau_clay1 = _model_springs.tau_over_tau_ult_clay(z_over_D_tau/_model_springs.m_to_in)
ptau_clay2 = _model_springs.tau_over_tau_ult_clay(z_over_D_tau/_model_springs.m_to_in, t_res=0.7)
ptau_sand = _model_springs.tau_over_tau_ult_sand(z_over_D_tau/_model_springs.m_to_in)

z_over_D_Q = np.linspace(0, 0.12, 100)
pQ = _model_springs.Q_over_Q_ult(z_over_D_Q/_model_springs.m_to_in)

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
fontsize = 16

ax[0].plot(z_over_D_tau, ptau_clay1, label="clay, $t_{res}=0.9$", linestyle="--")
ax[0].plot(z_over_D_tau, ptau_clay2, label="clay, $t_{res}=0.7$", linestyle=":")
ax[0].plot(z_over_D_tau, ptau_sand, label="sand")
ax[0].set_xlim(left=0)
ax[0].set_ylim(bottom=0)
ax[0].set_xlabel("z/D", fontsize=fontsize)
ax[0].set_ylabel("$\\hat{\\tau} = \\tau / \\tau_{ult}$", fontsize=fontsize)
# ax[0].set_title("Shaft Resistance t-z Curve")
ax[0].grid()
ax[0].legend()

ax[1].plot(z_over_D_Q, pQ, label="all soils")
ax[1].set_xlim(left=0)
ax[1].set_ylim(bottom=0)
ax[1].set_xlabel("z/D", fontsize=fontsize)
ax[1].set_ylabel("$\\hat{Q} = Q / Q_{ult}$", fontsize=fontsize)
# ax[1].set_title("End Resistance Q-z curve")
ax[1].grid()

plt.tight_layout()
plt.show()
#%%