#%%
import numpy as np
import pymc as pm
import pandas as pd
import copy
from tqdm import tqdm
import matplotlib.pyplot as plt

import packaged._pile_and_soil as _pile_and_soil
import packaged._model_springs as _model_springs

# Pile
pile = _pile_and_soil.Pile(R=0.3, L=30, E=35e9)

# Define the other variables (loading, number of elements, model noise variance)
P = 3.6e6 # top axial load
N = 100 # number of nodes along pile
z = np.linspace(0, pile.L, N)
z_midpoints = 0.5 * (z[:-1] + z[1:])

# according to API, beta for soils should be scaled by 1.25 for plugged piles
plug_factor=1.25

# in this test, there are two soil layers. each has its own uniform set of ground truth parameters
# Soil 1
layer1 = _pile_and_soil.SandLayer(gamma_d=15e3, e=0.689, N_q=8, beta=plug_factor*0.214, shaft_friction_limit=47.8e3, end_bearing_limit=1.9e6, base_depth=12.5)
layer2 = _pile_and_soil.SandLayer(gamma_d=17e3, e=0.441, N_q=40, beta=plug_factor*0.46, shaft_friction_limit=96e3, end_bearing_limit=10e6, base_depth=pile.L)

#%% --------------------------------------------------------------------------------------------
# Now we will perform a sensitivity analysis on the model parameters
# Vary each parameter by +-10% and see how the output changes

vary_factor = 1.1

# Don't vary the parameter
soil = _pile_and_soil.Soil([layer1, layer2])
res0 = _model_springs.solve_springs4(pile, soil, P, 3, N)

# Vary the parameter upwards
layer1.gamma_d *= vary_factor
layer2.gamma_d *= vary_factor
res1 = _model_springs.solve_springs4(pile, soil, P, 3, N)

# Vary the parameter downwards
layer1.gamma_d /= vary_factor**2
layer2.gamma_d /= vary_factor**2
res2 = _model_springs.solve_springs4(pile, soil, P, 3, N)

#%% --------------------------------------------------------------------------------------------
# Plot the results

plt.title("gamma_d")
plt.plot(res0[0], z, label="Original", linewidth=2)
plt.plot(res1[0], z, label="Increase by 10%", linestyle="--")
plt.plot(res2[0], z, label="Decrease by 10%", linestyle="--")
plt.xlim(left=0)
plt.gca().invert_yaxis()
plt.xlabel("F")
plt.ylabel("z")
plt.legend()
plt.grid()
plt.show()
#%%