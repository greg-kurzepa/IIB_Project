import numpy as np

m_to_in = 39.3701

class Pile:
    def __init__(self, R, L, W, E):
        self.R = R # the outer radius of the pile
        self.D = 2*R
        self.L = L # the length of the pile
        self.W = W # the total weight of the pile
        self.A = np.pi*self.R**2 # FOR NOW non-hollow
        self.C = 2*np.pi*self.R
        self.E = E

class Soil:
    def __init__(self, alpha=0.4, gamma=20e3, N_c=9, s_u0=30e3, rho=4.8e3, sigma_n=5e3):
        self.alpha = alpha
        self.gamma = gamma
        self.N_c = N_c
        self.s_u0 = s_u0
        self.rho = rho
        self.sigma_n = sigma_n

class System:
    def __init__(self, pile, soil):
        self.p = pile
        self.s = soil

    def soil_limit_model(self, z):
        # base bearing load
        BEARING = np.pi * self.p.R**2 * ( self.s.gamma*self.p.L + self.s.N_c*( self.s.s_u0 + self.s.rho*self.p.L ) )

        # total vertical shear force below z
        SHEAR = 2*np.pi*self.p.R*self.s.alpha * ( 0.5*self.s.rho*( self.p.L**2 - z**2 ) + self.s.s_u0*( self.p.L - z ) )

        # total pile weight below z
        W_FRAC = self.p.W * (1 - z/self.p.L)

        # vertical force in steel at that point
        F = BEARING + SHEAR - W_FRAC
        if max(W_FRAC / (BEARING+SHEAR)) > 0.05: print("Warning: W_FRAC is a significant proportion of the total force.")
        return F
    
    def g(self, disp):
        disp_over_D = disp / self.p.D
        
        # values to interpolate from
        z_over_D = np.array([0, 0.0016, 0.0031, 0.0057, 0.0080, 0.0100, 0.0200, np.inf]) / m_to_in
        tau_over_tau_ult = np.array([0, 0.3, 0.5, 0.75, 0.9, 1, 0.9, 0.9])

        # return np.interp(disp_over_D, z_over_D, tau_over_tau_ult)
        return np.sign(disp_over_D)*np.interp(np.abs(disp_over_D), z_over_D, tau_over_tau_ult)

    def h(self, disp):
        disp_over_D = disp / self.p.D

        # values to interpolate from
        z_over_D = np.array([-np.inf, 0, 0.002, 0.013, 0.042, 0.073, 0.1, np.inf]) / m_to_in
        Q_over_Q_ult = np.array([0, 0, 0.25, 0.5, 0.75, 0.9, 1, 1])

        # return np.interp(disp_over_D, z_over_D, Q_over_Q_ult)
        return disp_over_D*np.interp(disp_over_D, z_over_D, Q_over_Q_ult)