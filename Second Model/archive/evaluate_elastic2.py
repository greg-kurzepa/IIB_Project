#%%
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import packaged._pile_and_soil as _pile_and_soil
import packaged._model_springs as _model_springs

L_over_D_arr = np.logspace(0.7,2,10)
D = 1
E_pile = 100
E_soil = 1
P = 1
N = 100 # number of nodes (= number of elements + 1)

I = []
for L_over_D in tqdm(L_over_D_arr):
    dz = D*L_over_D / (N-1)
    dz_halfel = 0.5 * dz
    pile = _pile_and_soil.Pile(R=D/2, L=D*L_over_D, E=E_pile)

    d, u, res = _model_springs.solve_elastic(pile, E_soil=E_soil, P=P, N=N)

    d0 = d[0]
    I.append(d0 * D * E_soil / P)
# %%

plt.plot(L_over_D_arr, I)
plt.xscale("log")
plt.show()
# %%
