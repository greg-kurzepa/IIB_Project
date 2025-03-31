#%%
import jax
import jax.numpy as jnp
import jaxopt
import numpy as np

def f1(x): 
    return 2*x

def f2(x):
    return 3*x

def forward(unit_weights, layer_depths, N, L, threshold=2.5):
    # generate scaled unit weight profile
    z = jnp.linspace(0, L, N)
    idxs = jnp.array(N * layer_depths / L, dtype=jnp.int32)
    idxs = jnp.diff(idxs, prepend=0)
    profile = jnp.repeat(unit_weights, idxs, total_repeat_length=N)
    scaled_profile = jax.lax.select(profile < threshold, f1(profile), f2(profile))

    return scaled_profile

forward_jit = jax.jit(forward, static_argnames=("N", "L", "threshold"))

#%%
N = 100
L = 10.0
sigma = 0.1
unit_weights = jnp.array([2.0, 3.0])
layer_depths = jnp.array([5.0, L])
print(forward_jit(unit_weights, layer_depths, N, L))
# %%
