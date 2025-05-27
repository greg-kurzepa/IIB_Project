#%%
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import solver

# analytic linear elastic solution
def f_analytic(z, P, pile, lambda_k, Omega,):
    return P / (pile.E * pile.A * lambda_k) * (((1 + Omega*np.tanh(lambda_k*pile.L))/(Omega+np.tanh(lambda_k*pile.L)))*np.cosh(lambda_k*z) - np.sinh(lambda_k*z))

# N = 100
# L_over_D_arr = np.logspace(1,2,10)
# D = 1
# E_pile = 100
# E_soil = 1
# P = 1
# Kb = 0 # spring stiffness at the bottom

N = 100
L_over_D_arr = np.logspace(1,2,10)
D = 0.6
E_pile = 35e9
E_soil = E_pile/100
P = 1.8e6
Kb = 0 # spring stiffness at the bottom

class Pile:
    def __init__(self, R, L, E):
        self.R = R # the outer radius of the pile
        self.D = 2*R
        self.L = L # the length of the pile
        self.A = np.pi*self.R**2 # non-hollow
        self.C = 2*np.pi*self.R

        # constitutive model parameters
        self.E = E

I_analytic = []
I_solved_springs = []
zeros_analytic = []
zeros_solved_springs = []
for idx, L_over_D in (enumerate(L_over_D_arr)):
    pile = Pile(R=D/2, L=D*L_over_D, E=E_pile)
    z = np.linspace(0, pile.L, N)
    z_midpoints = 0.5 * (z[:-1] + z[1:])
    dz = pile.L / (N-1)
    dz_halfel = 0.5 * dz
    lambda_k = np.sqrt(pile.C * E_soil / (pile.A * pile.E))
    Omega = Kb / (lambda_k * pile.E * pile.A)

    # compute analytic linear elastic solution
    u = f_analytic(z_midpoints, P, pile, lambda_k, Omega)
    d = f_analytic(z, P, pile, lambda_k, Omega)
    I_analytic.append(d[0] * D * E_soil / P)
    zeros_analytic.append(solver.f_simultaneous_linearelastic(np.concatenate((d, u)), dz, N, pile, E_soil, P, Kb))

    # compute numerical linear elastic solution using solve_springs4 function
    d2, u2, zeros2, _ = solver.solve_linearelastic(pile, np.array([E_soil]), np.array([pile.L]), P, N)
    I_solved_springs.append(d2[0] * D * E_soil / P)
    zeros_solved_springs.append(zeros2)

#%%
plt.plot(L_over_D_arr, I_analytic, label="analytic", alpha=0.75)
plt.plot(L_over_D_arr, I_solved_springs, label="solved springs", linestyle="--", alpha=0.75)
plt.xlabel("$L/D$")
plt.ylabel("$I = S D E_s/P$")
plt.xscale("log")
plt.legend()
plt.show()
# %%