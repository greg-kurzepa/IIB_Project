#%%
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import packaged._pile_and_soil as _pile_and_soil
import packaged._model_springs as _model_springs

K = 100 # Pile to soil stiffness ratio
L_over_D_list = np.logspace(0.7, 2, num=20, base=10) # L/D ratios to use
D = 0.6
E_pile = 35e9
P = 1.8e6
N=100

# to use _model_springs for a purely linear stress-strain soil relationship, need to do 2 things:
# 1. define an elastic soil that has tau_ult=1 and q_ult=1
# 2. supply a soil_constitutive_func to solve_springs3 that computes strain from u,d and returns the stress
class ElasticSoil:
    def __init__(self, base_depth):
        self.gamma_d = np.nan
        self.gamma_sat = np.nan
        self.base_depth = base_depth

        self.layer_type = "elastic"
        
    def tau_ult(self, eff_stress):
        return 1
    
    def q_ult(self, eff_stress):
        return 1
f_tau = lambda u, strain : E_pile / K * strain
# f_Q = lambda d_tip, strain_tip : E_pile / K * strain_tip
f_Q = lambda d_tip, strain_tip : 0

settlements = []
for L_over_D in tqdm(L_over_D_list):
    # Define pile and soil
    pile = _pile_and_soil.Pile(R=D/2, L=D*L_over_D, E=E_pile)
    layer1 = ElasticSoil(base_depth = D*L_over_D)
    soil = _pile_and_soil.Soil([layer1])

    # Solve
    res1 = _model_springs.solve_springs3(pile, soil, P=P, z_w=0, N=N,
                                         tau_over_tau_ult_func=f_tau, Q_over_Q_ult_func=f_Q)
    res2 = _model_springs.solve_elastic(pile, soil_E=E_pile/K, P=P, N=N)
    
    # for debugging. plot the force and strain and displacement profiles in 3 subplots
    fig, ax = plt.subplots(5, 1, sharex=True)
    z = np.linspace(0, D*L_over_D, N)
    z_midpoints = 0.5 * (z[:-1] + z[1:])

    ax[0].plot(z, res1[0])
    ax[0].set_title("Force profile")
    ax[1].plot(z, res1[1])
    ax[1].set_title("Strain profile")
    ax[2].plot(z, res1[2])
    ax[2].set_title("Displacement profile")

    ax[3].plot(z_midpoints, res1[3])
    ax[4].plot(z_midpoints, res1[4])
    print(res1[3], res1[4])

    plt.show()

    settlements.append(res1[2][0]) # surface settlement
settlements = np.array(settlements)

I_q = settlements * D * E_pile / (P * K)

#%%

plt.plot(L_over_D_list, I_q)
plt.xscale("log")
plt.show()

#%%