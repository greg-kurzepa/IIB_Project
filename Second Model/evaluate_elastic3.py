#%%
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import packaged._pile_and_soil as _pile_and_soil
import packaged._model_springs as _model_springs

# analytic linear elastic solution
def f_analytic(z, P, pile, lambda_k, Omega,):
    return P / (pile.E * pile.A * lambda_k) * (((1 + Omega*np.tanh(lambda_k*pile.L))/(Omega+np.tanh(lambda_k*pile.L)))*np.cosh(lambda_k*z) - np.sinh(lambda_k*z))

N = 100
L_over_D_arr = np.logspace(1,2,10)
D = 1
E_pile = 100
E_soil = 1
P = 1
Kb = 0 # spring stiffness at the bottom

class ElasticSoil:
    def __init__(self, base_depth):
        self.gamma_d = np.nan
        self.gamma_sat = np.nan
        self.base_depth = base_depth
        self.shaft_friction_limit = np.inf
        self.end_bearing_limit = np.inf

        self.layer_type = "elastic"
        
    def tau_ult(self, eff_stress):
        return 1
    
    def q_ult(self, eff_stress):
        return 1
f_tau = lambda u : E_soil * u
f_Q = lambda d_tip : 0

I_analytic = []
I_solved = []
I_solved_springs = []
zeros_analytic = []
zeros_solved = []
zeros_solved_springs = []
for idx, L_over_D in (enumerate(L_over_D_arr)):
    pile = _pile_and_soil.Pile(R=D/2, L=D*L_over_D, E=E_pile)
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
    zeros_analytic.append(_model_springs.f_simultaneous_elastic(np.concatenate((d, u)), dz, N, pile, E_soil, P))

    # compute numerical linear elastic solution using custom function
    d_solved, u_solved, res = _model_springs.solve_elastic(pile, E_soil=E_soil, P=P, N=N)
    # print(f"Solved iteration {idx}. ier: {res[2]}. Message: {res[3]}")
    I_solved.append(d_solved[0] * D * E_soil / P)
    zeros_solved.append(_model_springs.f_simultaneous_elastic(np.concatenate((d_solved, u_solved)), dz, N, pile, E_soil, P))

    # compute numerical linear elastic solution using solve_springs4 function
    soil = _pile_and_soil.Soil([ElasticSoil(base_depth = D*L_over_D)])
    res2 = _model_springs.solve_springs4(pile, soil, P, 0, N, tau_over_tau_ult_func=f_tau, Q_over_Q_ult_func=f_Q)
    I_solved_springs.append(res2[2][0] * D * E_soil / P)
    zeros_solved_springs.append(res2[4])

#%%
plt.plot(L_over_D_arr, I_analytic, label="analytic", alpha=0.75)
plt.plot(L_over_D_arr, I_solved, label="solved", alpha=0.75)
plt.plot(L_over_D_arr, I_solved_springs, label="solved springs", linestyle="--", alpha=0.75)
plt.xlabel("$L/D$")
plt.ylabel("$I = S D E_s/P$")
plt.xscale("log")
plt.legend()
plt.show()
# %%