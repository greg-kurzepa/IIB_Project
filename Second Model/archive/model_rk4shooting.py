#%%

import numpy as np
import scipy
import scipy.integrate

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
# import seaborn as sns
# import seaborn.objects as so

#%%

class Pile:
    def __init__(self, R, L, W):
        self.R = R # the outer radius of the pile
        self.L = L # the length of the pile
        self.W = W # the total weight of the pile
        self.A = np.pi*self.R**2 # FOR NOW non-hollow
        self.C = 2*np.pi*self.R

def g(pile, z, alpha, gamma, N_c, s_u0, rho):
    # base bearing load
    BEARING = np.pi * pile.R**2 * ( gamma*pile.L + N_c*( s_u0 + rho*pile.L ) )

    # total vertical shear force below z
    SHEAR = 2*np.pi*pile.R*alpha * ( 0.5*rho*( pile.L**2 - z**2 ) + s_u0*( pile.L - z ) )

    # total pile weight below z
    W_FRAC = pile.W * (1 - z/pile.L)

    # vertical force in steel at that point
    F = BEARING + SHEAR - W_FRAC
    return F

#%%

m_to_in = 39.3701

# True soil parameter values (used to test the MCMC fit)
alpha = 0.4 # shear utilisation, typically 0.3-0.8
gamma = 20e3 # unit weight of soil, typically 20-22 kn/m3 for London Clay
N_c = 9 # bearing utlisation, typically 9
s_u0 = 30e3 # surface undrained shear strength, representative of London Clay
rho = 4.8e3 # rate of increase of s_u with depth, representative of London Clay
sigma_n = 5e3 # model noise variance

# concrete parameters
pile = Pile(0.15, 10, 2275) # https://ukrstarline.ua/en/reinforced-concrete-products/reinforced-concrete-pile-driven/reinforced-concrete-pile-driven-c-10030-10
E = 20e9 # youngs modulus

P = 200e3 # top axial load
N = 101 # number of nodes along pile
z = np.linspace(0, pile.L, N)
dz = pile.L / (N-1) # length of one element
Q_ult = N_c * (s_u0 + rho * pile.L) # ultimate end bearing pressure

# known boundary conditions
disp_bottom = 0 # at z=L
strain_top = - P / (pile.A*E) # at z=0

# guessed strain boundary condition at z=L for shooting method
# using the soil-limit failure model. we know the shape is quadratic so need evaluate only 3 points.
F_bottom = g(pile, pile.L, alpha, gamma, N_c, s_u0, rho) # evaluate force at z=L
F_bottom_scaled = F_bottom * P/g(pile, 0, alpha, gamma, N_c, s_u0, rho) # evaluate force at z=0 and scale F_bottom by the ratio of this to actual top load
strain_guess_bottom = - F_bottom_scaled / (pile.A*E)

#%%

def shear_transfer(pile, disp):
    # values to interpolate from
    z_over_D = np.array([0, 0.0016, 0.0031, 0.0057, 0.0080, 0.0100, 0.0200, np.inf]) / m_to_in
    tau_over_tau_ult = np.array([0, 0.3, 0.5, 0.75, 0.9, 1, 0.9, 0.9])
    
    return(np.interp(0.5*disp/pile.R, z_over_D, tau_over_tau_ult))

def bearing_transfer(pile, disp):
    # values to interpolate from
    z_over_D = np.array([0, 0.002, 0.013, 0.042, 0.073, 0.1, np.inf]) / m_to_in
    Q_over_Q_ult = np.array([0, 0.25, 0.5, 0.75, 0.9, 1, 1])

    return(np.interp(0.5*disp/pile.R, z_over_D, Q_over_Q_ult))

# runge-kutta function. dy/dz = f(z,y)
# y = (u, du/dz)
def diffeq(z, y):
    return np.vstack((
        y[1], 
        np.append(
            # note, will want to make alpha calculation better as per guide
            # all but the last node are soil side shear. the last node is soil bearing shear.
            alpha * (s_u0 + rho * z[:-1]) * pile.C * shear_transfer(pile, y[0][:-1]) / (pile.A*E),
            Q_ult * bearing_transfer(pile, y[0][-1]) / (pile.A*E*dz)
        )
    ))

# residual definition. i.e. returns zero when solution matches boundary conditions.
def bc(ya, yb):
    return np.array([
        yb[0], # the value of u at z=L will be set to zero
        ya[1] - strain_top # the value of strain, du/dz, at z=0 will be set to strain_top
    ])

soil_failure_F = g(pile, z, alpha, gamma, N_c, s_u0, rho)
soil_failure_F *= (P / soil_failure_F[0])
strain_guess = - soil_failure_F / (pile.A*E)
disp_guess = disp_bottom + np.cumsum(strain_guess * dz)
y_guess = np.stack([disp_guess, strain_guess])

# df_guess = pd.DataFrame({"z":z, "F_guess":soil_failure_F, "u_guess":disp_guess})
# sns.relplot(
#     data=df_guess,
#     kind="line",
#     x="z", y="F_guess"
# )

# Solve boundary value problem
result = scipy.integrate.solve_bvp(diffeq, bc, z, y_guess, tol=1e-8)
u = result.sol(z)[0]
strain = result.sol(z)[1]
F = - result.sol(z)[1] * pile.A*E
# df = pd.DataFrame({"z":z, "u":u, "strain":strain, "F":F})

#%%

# Plot!

print(F)

plt.plot(z, F)
plt.xlabel("z")
plt.ylabel("F")
plt.grid()
plt.show()
# %%
