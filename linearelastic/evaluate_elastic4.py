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
# E_soil = 2.64e7
P = 4.6e6

L_over_D = 50

pile = solver.Pile(R=D/2, L=D*L_over_D, E=E_pile)
z = np.linspace(0, pile.L, N)
z_midpoints = 0.5 * (z[:-1] + z[1:])
dz = pile.L / (N-1)
dz_halfel = 0.5 * dz
# lambda_k = np.sqrt(pile.C * E_soil / (pile.A * pile.E))
# Kb = 73.26e6 # spring stiffness at the bottom, FOR NOW same as soil
Kb = 589e6
# Omega = Kb / (lambda_k * pile.E * pile.A)

# # compute analytic linear elastic solution
# u = f_analytic(z_midpoints, P, pile, lambda_k, Omega)
# d = f_analytic(z, P, pile, lambda_k, Omega)
# F_analytic = - pile.A * pile.E * strain_analytic(z, P, pile, lambda_k, Omega)
# I_analytic = d[0] * D * E_soil / P
# zeros_analytic = solver.f_simultaneous_linearelastic(np.concatenate((d, u)), dz, N, pile, E_soil, P, Kb)

# # compute numerical linear elastic solution using solve_springs4 function
# d2, u2, zeros2, _ = solver.solve_linearelastic(pile, np.array([E_soil]), np.array([pile.L]), P, Kb, N)
# F_springs = pile.A * pile.E * strain(u2, d2, dz_halfel)
# I_solved_springs = d2[0] * D * E_soil / P

#%%

# E_soil_list = np.array([0.1*E_soil, 0.5*E_soil])
E_soil_list = np.array([14.94e6, 30e6])
lambda_k_list = np.sqrt(pile.C * E_soil_list / (pile.A * pile.E))
Omega = Kb / (lambda_k_list[1] * pile.E * pile.A)
base_depths = np.array([pile.L/3, pile.L])

u1, strain1 = solver.solve_linearelastic_analytic(z, P, pile, lambda_k_list, Omega, base_depths)
F_analytic = - pile.A * pile.E * strain1

d2, u2, zeros2, _ = solver.solve_linearelastic(pile, E_soil_list, base_depths, P, Kb, N)
F_springs = pile.A * pile.E * strain(u2, d2, dz_halfel)

plt.plot(z, F_analytic, label='analytic')
# plt.plot(z, F_springs, label='springs', linestyle="--")
plt.xlabel('z (m)')
plt.ylabel('F (N)')
# ax[1].title(f"2-layer F_analytic vs F_springs, L_over_D = {L_over_D}")
plt.ylim(bottom=0)
plt.xlim(0, 30)
plt.grid()
# plt.legend()

plt.tight_layout()
plt.show()

#%%

import pandas as pd
forces = pd.DataFrame({
    "z": z,
    "F_analytic": F_analytic,
})
forces.to_csv("forces_analytic_4,6.csv", index=False)

#%%