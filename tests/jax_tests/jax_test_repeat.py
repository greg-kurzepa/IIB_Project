#%%
import jax
import jax.numpy as jnp
import jax.scipy.optimize as opt

# test whether repeat works with gradients
def f_repeat(depths, params):
    N=100
    x = jnp.linspace(0, L, N)
    idxs = jnp.round(jnp.array(N * depths / L)).astype(jnp.int32)
    idxs = jnp.diff(idxs, prepend=0)

    r = jnp.repeat(params, idxs, total_repeat_length=N)
    return jnp.sum(r / L)

def f_repeat_print(depths, params):
    N=100
    x = jnp.linspace(0, L, N)
    idxs = jnp.array(N * depths / L, dtype=jnp.int32)
    idxs = jnp.diff(idxs, prepend=0)

    r = jnp.repeat(params, idxs, total_repeat_length=N)
    print(r.shape)
    
    return jnp.sum(r / L)

L = 10.0
vg = jax.value_and_grad(jax.jit(f_repeat), argnums=(0, 1))
params = jnp.array([2.0,3.0])
depths = jnp.array([5.0,L])
g = vg(depths, params)
print(g)

f_repeat_print(depths, params)

# gradient DOES NOT WORK with depths parameter.
# this is fine for me, since the depths will be known.
# BUT this means I cannot support layer inference, and I must supply the layer depths as a separate input.
# %%
