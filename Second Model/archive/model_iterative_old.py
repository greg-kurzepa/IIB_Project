import numpy as np
import matplotlib.pyplot as plt
import arviz as az

class Pile:
    def __init__(self, R, L, W):
        self.R = R # the outer radius of the pile
        self.L = L # the length of the pile
        self.W = W # the total weight of the pile
        self.A = np.pi*self.R**2 # FOR NOW non-hollow

# True soil parameter values (used to test the MCMC fit)
alpha = 0.4 # shear utilisation, typically 0.3-0.8
gamma = 20e3 # unit weight of soil, typically 20-22 kn/m3 for London Clay
N_c = 9 # bearing utlisation, typically 9
s_u0 = 30e3 # surface undrained shear strength, representative of London Clay
rho = 4.8e3 # rate of increase of s_u with depth, representative of London Clay
sigma_n = 5e3 # model noise variance

# concrete parameters
pile = Pile(0.15, 10, 2275) # https://ukrstarline.ua/en/reinforced-concrete-products/reinforced-concrete-pile-driven/reinforced-concrete-pile-driven-c-10030-10
E = 20e9

N = 101 # number of nodes down pipe (= number of stresses, = #strains + 1, #s_u + 1)

z = np.linspace(0, pile.L, N)
dz = pile.L / (N-1)
z_midpoints = (z[1:] + z[:-1]) / 2
s_u = rho*z_midpoints + s_u0
tau_ult = alpha * s_u # note, will want to make alpha calculation better as per guide

# soil-limit based model, used for a rough initial guess
def g(pile, z, alpha, gamma, N_c, s_u0, rho):
    # base bearing load
    BEARING = np.pi * pile.R**2 * ( gamma*pile.L + N_c*( s_u0 + rho*pile.L ) )
    print(BEARING)

    # total vertical shear force below z
    SHEAR = 2*np.pi*pile.R*alpha * ( 0.5*rho*( pile.L**2 - z**2 ) + s_u0*( pile.L - z ) )

    # total pile weight below z
    W_FRAC = pile.W * (1 - z/pile.L)

    # vertical force in steel at that point
    F = BEARING + SHEAR - W_FRAC
    return F

# for clay, API (American Petroleum Institute) 2002
def shear_transfer_clay_API(pile, u):
    # values to interpolate from
    z_over_D = np.array([0, 0.0016, 0.0031, 0.0057, 0.0080, 0.0100, 0.0200, np.inf])
    tau_over_tau_ult = np.array([0, 0.3, 0.5, 0.75, 0.9, 1, 0.9])
    
    return(np.interp(0.5*u/pile.R), z_over_D, tau_over_tau_ult)

shear_transfer = shear_transfer_clay_API

# intial guess for axial force at each point
F = g(pile, z, alpha, gamma, N_c, s_u0, rho)/2 # make load on top half of failure load
P = F[0] # load on top stays constant

n_its = 3
for i in range(n_its):
    strain = ( (F[1:] + F[:-1]) / 2 ) / (pile.A*E) # avg force / AE
    u = np.cumsum(strain * dz) # wrong; we need a boundary condition!
    S = shear_transfer(pile, u) * tau_ult * 2*np.pi*pile.R*dz # shear force in the element
    F = 