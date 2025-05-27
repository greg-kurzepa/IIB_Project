#%%
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import solver

# analytic linear elastic solution
def f_analytic(z, P, pile, lambda_k, Omega):
    return P / (pile.E * pile.A * lambda_k) * (((1 + Omega*np.tanh(lambda_k*pile.L))/(Omega+np.tanh(lambda_k*pile.L)))*np.cosh(lambda_k*z) - np.sinh(lambda_k*z))
def strain_analytic(z, P, pile, lambda_k, Omega):
    print("Omega", Omega)
    print(f"C1: {(1 + Omega*np.tanh(lambda_k*pile.L))/(Omega+np.tanh(lambda_k*pile.L))}")
    print(f"C1, zero Omega: {1/(np.tanh(lambda_k*pile.L))}")
    return P / (pile.E * pile.A) * (((1 + Omega*np.tanh(lambda_k*pile.L))/(Omega+np.tanh(lambda_k*pile.L)))*np.sinh(lambda_k*z) - np.cosh(lambda_k*z))

def strain(u, d, dz_halfel):
    strain_top = (d[:-1] - u) / dz_halfel
    strain_tip = (u[-1] - d[-1]) / dz_halfel
    strain = np.append(strain_top, strain_tip)
    return strain

#-------------------------------------------------------
# Single soil layer

N = 100
L_over_D_arr = np.logspace(1,2,10)
D = 0.6
E_pile = 35e9
E_soil = 2.64e7
P = 1.8e6

L_over_D = 50

pile = solver.Pile(R=D/2, L=D*L_over_D, E=E_pile)
z = np.linspace(0, pile.L, N)
z_midpoints = 0.5 * (z[:-1] + z[1:])
dz = pile.L / (N-1)
dz_halfel = 0.5 * dz
lambda_k = np.sqrt(pile.C * E_soil / (pile.A * pile.E))
Kb = 73.26e6 # spring stiffness at the bottom, FOR NOW same as soil
Omega = Kb / (lambda_k * pile.E * pile.A)

#%%
analytic_I = []
analytic_zeros = []
springs_I = []
springs_zeros = []

L_over_D_arr = np.logspace(1, 2, 50)
E_soil_arr = [E_pile/100, E_pile/1000, E_pile/10000]
for E_soil in E_soil_arr:
    _analytic_I = []
    _analytic_zeros = []
    _springs_I = []
    _springs_zeros = []

    for L_over_D in tqdm(L_over_D_arr):
        pile.L = D * L_over_D
        
        # compute analytic linear elastic solution
        u = f_analytic(z_midpoints, P, pile, lambda_k, Omega)
        d = f_analytic(z, P, pile, lambda_k, Omega)
        F_analytic = - pile.A * pile.E * strain_analytic(z, P, pile, lambda_k, Omega)
        I_analytic = d[0] * D * E_soil / P
        zeros_analytic = solver.f_simultaneous_linearelastic(np.concatenate((d, u)), dz, N, pile, E_soil, P, Kb)
        
        _analytic_I.append(I_analytic)
        _analytic_zeros.append(zeros_analytic)

        # compute numerical linear elastic solution using solve_springs4 function
        d2, u2, zeros2, _ = solver.solve_linearelastic(pile, np.array([E_soil]), np.array([pile.L]), P, Kb, N)
        F_springs = pile.A * pile.E * strain(u2, d2, dz_halfel)
        I_solved_springs = d2[0] * D * E_soil / P

        _springs_I.append(I_solved_springs)
        _springs_zeros.append(zeros2)

    analytic_I.append(_analytic_I)
    analytic_zeros.append(_analytic_zeros)
    springs_I.append(_springs_I)
    springs_zeros.append(_springs_zeros)

#%%
fontsize = 16
for E_s in E_soil_list:
    plt.plot(L_over_D_arr, analytic_I, label='analytic')
    plt.plot(L_over_D_arr, springs_I, label='springs', linestyle='--')
plt.xscale('log')
plt.xlabel('$L/D$', fontsize=fontsize)
plt.ylabel('$I = S D E_s/P$', fontsize=fontsize)
plt.grid()
plt.xlim(left=10)
plt.ylim(bottom=0)
plt.legend()
plt.show()

#%%