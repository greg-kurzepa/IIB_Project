import jax
import jax.numpy as jnp
import jaxopt

N = 100
def solve_springs(pile_params, soil_params):
    # pile_params is a 1d array
    # soil_params is a 2d array, each row is a layer and each column is a layer property
    # layer properties: soil_type, 