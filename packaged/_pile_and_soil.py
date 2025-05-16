import numpy as np
import scipy

gamma_w = 9.81e3

class Pile:
    def __init__(self, R, L, f_ck=50, alpha_e=1.0, G_F0=0.065, reinforcement_ratio=0.04):
        """_summary_

        Args:
            R (float): _description_
            L (float): _description_
            f_ck (float, optional): Characteristic compressive strength of concrete. See FIB 2008
            alpha_e (float, optional): Scaling factor for elastic modulus based on type of aggregate. See FIB 2008
            G_F0 (float, optional): Fracture energy of concrete. See FIB 2008. 0.065N/mm for riverbed gravel, 0.106N/mm for crushed basalt.
        """

        self.R = R # the outer radius of the pile
        self.D = 2*R
        self.L = L # the length of the pile
        self.A = np.pi*self.R**2 # non-hollow
        self.C = 2*np.pi*self.R

        # constitutive model parameters
        self.steel_E = 210e9 # Pa, Young's modulus of steel
        self.reinforcement_ratio = reinforcement_ratio # steel area as proportion of pile area
        self.E_c0 = 20.5e3 # MPa, this value never changes
        self.f_ck = f_ck # in MPa
        self.alpha_e = alpha_e
        self.G_F0 = G_F0 # in N/mm

        self.f_cm = self.f_ck + 8 # in MPa
        self.E_ci = self.E_c0 * alpha_e * (self.f_cm / 10)**(1/3) # in MPa
        self.f_ctm = 2.64 * (np.log(self.f_cm / 10) - 0.1) # in MPa
        self.G_F = G_F0 * np.log(1 + self.f_cm / 10)

        self.microcracking_strain = 0.9 * self.f_ctm / self.E_ci # strain at which stress reaches 0.9 * f_ctm
        self.crack_strain = 0.15
        print(f"microcracking strain: {self.microcracking_strain:.2e}")
        assert self.microcracking_strain < self.crack_strain, "microcracking strain should be less than crack strain of 0.15"

        self.w1 = self.G_F / self.f_ctm
        self.w2 = 5 * self.G_F / self.f_ctm

    def F_over_AEs(self, strain, d): # vectorised!
        # return self.E * strain

        steel_stress_over_es = strain

        # using only LHS of fig. 4.7 (fib 2008)
        concrete_stress_MPa = np.piecewise(
            strain,
            condlist = [
                strain > -self.microcracking_strain,
                (strain <= -self.microcracking_strain) & (strain > -self.crack_strain),
                strain <= -self.crack_strain
            ],
            funclist = [
                lambda strain: self.E_ci * strain,
                lambda strain: np.interp(strain, [-self.microcracking_strain, -self.crack_strain], [-0.9*self.f_ctm, -self.f_ctm]),
                lambda strain: 0 * strain
            ]
        )

        concrete_stress_over_Es = concrete_stress_MPa / (self.steel_E / 1e6)

        return self.reinforcement_ratio * steel_stress_over_es + (1 - self.reinforcement_ratio) * concrete_stress_over_Es
    
    def __repr__(self):
        return f"Pile(R={self.R}, L={self.L}, f_ck={self.f_ck}, alpha_e={self.alpha_e}, G_F0={self.G_F0}, reinforcement_ratio={self.reinforcement_ratio})"

class ClayLayer:
    def __init__(self, gamma_d, e, N_c, psi, base_depth):
        self.gamma_d = gamma_d # dry unit weight
        self.e = e # void ratio
        self.N_c = N_c # bearing capacity factor, should be 9 for clays
        self.psi = psi # normalised undrained shear strength (correlated with OCR, see R8.4)
        self.shaft_pressure_limit = np.inf
        self.end_pressure_limit = np.inf

        self.base_depth = base_depth # depth at which this layer ends and another begins

        self.alpha = 0.5 * self.psi**(-0.5) if self.psi <= 1.0 else 0.5 * self.psi**(-0.25)
        assert self.alpha > 0 and self.alpha <= 1

        self.layer_type = "clay"

    @property
    def gamma_sat(self):
        return self.gamma_d + gamma_w * self.e/(1 + self.e)

    def tau_ult(self, eff_stress):
        s_u = self.psi * eff_stress
        return self.alpha * s_u
    
    def q_ult(self, eff_stress):
        s_u = self.psi * eff_stress
        return self.N_c * s_u
    
    def __repr__(self):
        return f"ClayLayer(gamma_d={self.gamma_d}, e={self.e}, N_c={self.N_c}, psi={self.psi}, base_depth={self.base_depth})"

class SandLayer:
    def __init__(self, gamma_d, e, N_q, beta, shaft_pressure_limit, end_pressure_limit, base_depth):
        self.gamma_d = gamma_d # dry unit weight
        self.e = e # void ratio
        self.N_q = N_q # bearing capacity factor
        self.beta = beta # shaft friction factor
        self.shaft_pressure_limit = shaft_pressure_limit # in Pa
        self.end_pressure_limit = end_pressure_limit # in Pa

        self.base_depth = base_depth # depth at which this layer ends and another begins

        self.layer_type = "sand"

    @property
    def gamma_sat(self):
        return self.gamma_d + gamma_w * self.e/(1 + self.e)

    def tau_ult(self, eff_stress):
        return self.beta * eff_stress
    
    def q_ult(self, eff_stress):
        return self.N_q * eff_stress
    
    def __repr__(self):
        return f"SandLayer(gamma_d={self.gamma_d}, e={self.e}, N_q={self.N_q}, beta={self.beta}, shaft_pressure_limit={self.shaft_pressure_limit}, end_pressure_limit={self.end_pressure_limit}, base_depth={self.base_depth})"

class Soil:
    """It stores the layers in the soil from top to bottom."""
    def __init__(self, layers=None):
        if layers is not None:
            self.layers = layers
        else:
            self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def find_layer_id_at_depth(self, z):
        # accounts for the case when one layer might have no depth, i.e. two layers at the same coordinate. in this case ignore the layer that has no depth.
        for i, layer in enumerate(self.layers):
            if z < layer.base_depth:
                return i
            
        raise ValueError(f"the passed z, {z} is deeper than the base depth of the deepest layer, {self.layers[-1].base_depth}.")