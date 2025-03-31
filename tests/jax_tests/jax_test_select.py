#%%
import jax
import jax.numpy as jnp
import jax.scipy.optimize as opt

def f1(x):
    return 2*x

def f2(x):
    return 3*x

def solve(params, depths, N=100, L=10.0, threshold=2.5):
    x = jnp.linspace(0, L, N)
    idxs = jnp.array(N * depths / L, dtype=jnp.int32)
    idxs = jnp.diff(idxs, prepend=0)
    profile = jnp.repeat(params, idxs)

    scaled_profile = jax.lax.select(profile < threshold, f1(profile), f2(profile))
    return jnp.sum(scaled_profile / L)

def solve_print(params, depths, N=100, L=10.0, threshold=2.5):
    x = jnp.linspace(0, L, N)
    idxs = jnp.array(N * depths / L, dtype=jnp.int32)
    idxs = jnp.diff(idxs, prepend=0)
    profile = jnp.repeat(params, idxs)

    scaled_profile = jax.lax.select(profile < threshold, f1(profile), f2(profile))
    print(profile, scaled_profile)
    return jnp.sum(scaled_profile / L)

#%%

L = 10.0
N = 100
vg = jax.value_and_grad(solve, argnums=0)
params = jnp.array([2.0,3.0])
depths = jnp.array([5.0,L])
g = vg(params, depths, N, L)
print(g)

solve_print(params, depths, N, L)