import numpy as np
import scipy
import scipy.optimize
    
class Pile:
    def __init__(self, R, L, E):
        self.R = R # the outer radius of the pile
        self.D = 2*R
        self.L = L # the length of the pile
        self.A = np.pi*self.R**2 # non-hollow
        self.C = 2*np.pi*self.R

        # constitutive model parameters
        self.E = E

def get_u(z, C1, C2, lambda_k):
    return C1 * np.exp(lambda_k * z) + C2 * np.exp(-lambda_k * z)
def get_dudz(z, C1, C2, lambda_k):
    return C1 * lambda_k * np.exp(lambda_k * z) - C2 * lambda_k * np.exp(-lambda_k * z)
def solve_linearelastic_analytic(z, P, pile, lambda_k_list, Omega, base_depths):
    # only works for 2 layers.

    if type(Omega) == np.ndarray:
        Omega = Omega.item()

    lambda_vec = np.array([lambda_k_list[0], lambda_k_list[0], lambda_k_list[1], lambda_k_list[1]])

    top = np.array([1, -1, 0, 0])
    alphas = np.array([1, 1, -1, -1]) * np.exp( base_depths[0] * lambda_vec * np.array([1, -1, 1, -1]) )
    betas = alphas * lambda_vec * np.array([1, -1, 1, -1])
    bottom = np.array([0, 0, np.exp(2 * pile.L * lambda_k_list[1]) * (1 + Omega), Omega - 1])
    A = np.vstack((top, alphas, betas, bottom))

    b = np.array([-P/(pile.E * pile.A * lambda_k_list[0]), 0, 0, 0])

    # print(f"A: {A}, b: {b}")
    constants = np.linalg.solve(A, b)

    u = np.where(
        z <= base_depths[0],
        get_u(z, constants[0], constants[1], lambda_k_list[0]),
        get_u(z, constants[2], constants[3], lambda_k_list[1])
    )
    dudz = np.where(
        z <= base_depths[0],
        get_dudz(z, constants[0], constants[1], lambda_k_list[0]),
        get_dudz(z, constants[2], constants[3], lambda_k_list[1])
    )
    return u, dudz

def f_simultaneous_linearelastic(x, dz, N, pile, E_soil_profile, P, Kb):
        d = x[:N]
        u = x[N:]

        dz_halfel = dz/2
        strain_top = (d[:-1] - u) / dz_halfel
        strain_bottom = (u - d[1:]) / dz_halfel

        F_tip_over_AE = Kb/(pile.A * pile.E) * d[-1]
        # F_tip = pile.A * E_soil * strain[-1]
        F_top_excluding_tip_over_AE = strain_top
        F_top_over_AE = np.append(F_top_excluding_tip_over_AE, F_tip_over_AE)
        
        F_bottom_excluding_head_over_AE = strain_bottom
        F_bottom_over_AE = np.insert(F_bottom_excluding_head_over_AE, 0, P/(pile.A * pile.E))

        S_over_AE = (pile.C/pile.A) * dz * (E_soil_profile/pile.E) * u

        zeros_1 = F_bottom_over_AE - F_top_over_AE
        # zeros_2 = F_top + S - np.insert(F_top[:-1], 0, P)
        zeros_2 = F_top_over_AE[1:] + S_over_AE - F_bottom_over_AE[:-1]

        return np.concatenate((zeros_1, zeros_2))

def solve_linearelastic(pile, E_soil_list, base_depths, P, Kb, N=100, tol=1e-8, outtol=1e-3):
    z = np.linspace(0, pile.L, N)
    z_midpoints = 0.5 * (z[:-1] + z[1:])
    dz = pile.L / (N-1)
    dz_halfel = 0.5 * dz

    E_soil_profile = E_soil_list[np.searchsorted(base_depths, z_midpoints)]

    d_initial = np.zeros_like(z)
    u_initial = np.zeros_like(z_midpoints)
    initial = np.concatenate((d_initial, u_initial))
    
    res, infodict, ier, mesg = scipy.optimize.fsolve(f_simultaneous_linearelastic, initial, xtol=tol, full_output=True, args=(dz, N, pile, E_soil_profile, P, Kb))
    # check if the solver converged
    zeros = f_simultaneous_linearelastic(res, dz, N, pile, E_soil_profile, P, Kb)
    if any(abs(zeros) > outtol):
        print(f"WARNING: Elastic solver did not converge, max fsolve error was {np.max(abs(zeros)):.4e}. ier: {ier}, mesg: {mesg}")

    d = res[:N]
    u = res[N:]

    return d, u, zeros, res