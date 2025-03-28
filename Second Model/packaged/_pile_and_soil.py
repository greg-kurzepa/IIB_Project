import numpy as np
import scipy

gamma_w = 9.81e3

class Pile:
    def __init__(self, R, L, E, W=None):
        self.R = R # the outer radius of the pile
        self.D = 2*R
        self.L = L # the length of the pile
        self.W = W # the total weight of the pile
        self.A = np.pi*self.R**2 # FOR NOW non-hollow
        self.C = 2*np.pi*self.R
        self.E = E

    def stress_from_strain(self, strain):
        return self.E * strain

    def strain_from_stress_pick_lowest(self, stress):
        return stress / self.E

class ClayLayer:
    def __init__(self, gamma_d, e, N_c, psi, base_depth):
        self.gamma_d = gamma_d # dry unit weight
        self.e = e # void ratio
        self.N_c = N_c # bearing capacity factor, should be 9 for clays
        self.psi = psi # normalised undrained shear strength (correlated with OCR, see R8.4)
        self.shaft_friction_limit = np.inf
        self.end_bearing_limit = np.inf

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

class SandLayer:
    def __init__(self, gamma_d, e, N_q, beta, shaft_friction_limit, end_bearing_limit, base_depth):
        self.gamma_d = gamma_d # dry unit weight
        self.e = e # void ratio
        self.N_q = N_q # bearing capacity factor
        self.beta = beta # shaft friction factor
        self.shaft_friction_limit = shaft_friction_limit # in Pa
        self.end_bearing_limit = end_bearing_limit # in Pa

        self.base_depth = base_depth # depth at which this layer ends and another begins

        self.layer_type = "sand"

    @property
    def gamma_sat(self):
        return self.gamma_d + gamma_w * self.e/(1 + self.e)

    def tau_ult(self, eff_stress):
        return self.beta * eff_stress
    
    def q_ult(self, eff_stress):
        return self.N_q * eff_stress

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