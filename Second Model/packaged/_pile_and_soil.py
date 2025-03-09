import numpy as np
import scipy

class Pile:
    def __init__(self, R, L, W, E):
        self.R = R # the outer radius of the pile
        self.D = 2*R
        self.L = L # the length of the pile
        self.W = W # the total weight of the pile
        self.A = np.pi*self.R**2 # FOR NOW non-hollow
        self.C = 2*np.pi*self.R
        self.E = E

class SoilLayer:
    # a set of parameters and priors I have used before is
    # alpha=0.4, gamma=20e3, N_c=9, s_u0=30e3, rho=4.8e3, sigma_n=5e3, base_depth=pile.L

    def __init__(self, alpha, gamma, N_c, s_u0, rho, base_depth):
        self.alpha = alpha              # shear utilisation, typically 0.3-0.8
        self.gamma = gamma              # unit weight of soil, typically 20-22 kn/m3 for London Clay
        self.N_c = N_c                  # bearing utlisation, typically 9
        self.s_u0 = s_u0                # surface undrained shear strength
        self.rho = rho                  # rate of increase of s_u with depth
        self.base_depth = base_depth    # depth at which this layer ends and another begins

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