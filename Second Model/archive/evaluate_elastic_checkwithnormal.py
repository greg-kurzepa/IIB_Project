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

settlements = []
for L_over_D in tqdm(L_over_D_list):
    # Define pile and soil
    pile = _pile_and_soil.Pile(R=D/2, L=D*L_over_D, E=E_pile)
    layer1 = _pile_and_soil.SandLayer(gamma_d=15e3, gamma_sat=19e3, N_q=8, beta=0.214, shaft_friction_limit=47.8e3, end_bearing_limit=1.9e6, base_depth=D*L_over_D)
    soil = _pile_and_soil.Soil([layer1])

    # Solve
    res1 = _model_springs.solve_springs3(pile, soil, P=P, z_w=0, N=100)
    settlements.append(res1[2][0]) # surface settlement
settlements = np.array(settlements)

I_q = settlements * D / P

#%%

plt.plot(L_over_D_list, I_q)
plt.xscale("log")
plt.show()

#%%